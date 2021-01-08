/**
 * @brief
 * intersect
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/hash.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

namespace intersect_internal {

  struct StateInfo {
    // the state_idx01 in a_fsas_.
    int32_t a_fsas_state_idx01;
    // the state_idx01 in b_fsas_.
    int32_t b_fsas_state_idx01;
  };

  struct ArcInfo {
    int32_t src_ostate;  // source state-index which is index into states_.
    int32_t dest_ostate;  // dest state-index which is index into states_.
    int32_t a_arc_idx012;  // The idx012 of the source arc in a_fsas_.
    int32_t b_arc_idx012;  // The idx012 of the source arc in b_fsas_.
    // Note: other fields, e.g. the label and score, can be worked
    // out from the arc-indexes.
  };


/*
static std::ostream &operator<<(std::ostream &os, const StateInfo &s) {
  os << "StateInfo{" << s.a_fsas_state_idx01 << ","
     << s.b_fsas_state_idx01 << "}";
  return os;
}

static std::ostream &operator<<(std::ostream &os, const ArcInfo &a) {
  os << "ArcInfo{" << a.a_fsas_arc_idx012 << "," << a.arc_loglike << ","
     << a.u.dest_a_fsas_state_idx01 << "," << a.end_loglike << "}";
  return os;
}
*/

}  // namespace intersect_internal

using namespace intersect_internal;  // NOLINT

// Caution: this is really a .cu file.  It contains mixed host and device code.

/*
   Intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  This version does only forward-backward
   pruning in the backward pass; the forward pass does no pruning.

   Can use either different decoding graphs (one per acoustic sequence) or a
   shared graph.

   How to use this object:
       Construct it
       Call Intersect()
       Call FormatOutput()
*/
class DeviceIntersector {
 public:
  /**
     This object does intersection on device (the general case, but without treating
     epsilons specially)

       @param [in] a_fsas  An FsaVec (3 axes), must be valid.  Caution: in future,
                           we may require that it be arc-sorted.
       @param [in] b_fsas  An FsaVec (3 axes), must be valid.
       @param [in] b_to_a_map  Map from fsa-index in b_fsas to the index of the FSA
                           in a_fsas that we want to intersect it with.

     Does not fully check its args (see wrapping code).  After constructing this object,
     call Intersect() and then FormatOutput().
   */
  DeviceIntersector(FsaVec &a_fsas, FsaVec &b_fsas,
                    const Array1<int32_t> &b_to_a_map):
      c_(a_fsas.Context()),
      a_fsas_(a_fsas),
      b_fsas_(b_fsas),
      b_to_a_map_(b_to_a_map),
      b_state_bits_(2 + HighestBitSet(b_fsas_.TotSize(1))),
      key_bits_(b_state_bits_ + 2 + HighestBitSet(a_fsas_.shape.MaxSize(1))) {

    if (key_bits_ < 32)  // TEMP!!
      key_bits_ = 32;

    // We may want to tune this hash size eventually.
    // Note: the hash size
    int32_t hash_size = 4 * RoundUpToNearestPowerOfTwo(b_fsas.NumElements()),
        min_hash_size = 1 << 16;
    if (hash_size < min_hash_size)
      hash_size = min_hash_size;
    // caution: also use hash_size in FirstIter() as default size of various arrays.
    state_pair_to_state_ = Hash(c_, hash_size);

    K2_CHECK(c_->IsCompatible(*b_fsas.Context()));
    K2_CHECK(c_->IsCompatible(*b_to_a_map.Context()));
  }


  void FirstIter() {
    int32_t initial_size = state_pair_to_state_.NumBuckets();
    arcs_row_ids_ = Array1<int32_t>(c_, initial_size);
    arcs_row_ids_.Resize(0, true);
    arcs_ = Array1<ArcInfo>(c_, initial_size);
    arcs_.Resize(0, true);

    int32_t num_fsas = b_fsas_.Dim0();

    states_ = Array1<StateInfo>(c_, initial_size);

    Renumbering renumber_initial_states(c_, num_fsas);

    char *keep_initial_states = renumber_initial_states.Keep().Data();

    const int32_t *b_fsas_row_splits1_data = b_fsas_.RowSplits(1).Data(),
                          *b_to_a_map_data = b_to_a_map_.Data(),
                 *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();

    K2_EVAL(c_, num_fsas, lambda_set_keep, (int32_t i) -> void {
        int nonempty_b = b_fsas_row_splits1_data[i+1] > b_fsas_row_splits1_data[i],
                   i_a = b_to_a_map_data[i],
            nonempty_a = a_fsas_row_splits1_data[i_a+1] > a_fsas_row_splits1_data[i_a];
        keep_initial_states[i] = (char)(nonempty_a & nonempty_b);
      });
    int32_t num_initial_states = renumber_initial_states.New2Old().Dim();

    states_.Resize(num_initial_states, true);
    final_states_ = Array1<StateInfo>(c_, num_initial_states);

    StateInfo *states_data = states_.Data(),
        *final_states_data = final_states_.Data();
    const int32_t *new2old_data = renumber_initial_states.New2Old().Data();
    K2_EVAL(c_, num_initial_states, lambda_set_state_info, (int32_t new_i) -> void {
        int32_t b_idx0 = new2old_data[new_i],
               b_idx01 = b_fsas_row_splits1_data[b_idx0],
                a_idx0 = b_to_a_map_data[b_idx0],
               a_idx01 = a_fsas_row_splits1_data[a_idx0];
        StateInfo info;
        info.a_fsas_state_idx01 = a_idx01;
        info.b_fsas_state_idx01 = b_idx01;
        states_data[new_i] = info;

        // now set final-state info.
        info.a_fsas_state_idx01 = a_fsas_row_splits1_data[a_idx0 + 1] - 1;
        info.b_fsas_state_idx01 = b_fsas_row_splits1_data[b_idx0 + 1] - 1;
        final_states_data[new_i] = info;
      });

    iter_to_state_row_splits_cpu_.reserve(128);
    iter_to_state_row_splits_cpu_.push_back(0);
    iter_to_state_row_splits_cpu_.push_back(num_initial_states);
  }


  /*
    Adds the StateInfo for the final-states to the states_ array.
  */
  void LastIter() {
    int32_t num_final_states = final_states_.Dim();
    int32_t cur_num_states = states_.Dim(),
            tot_num_states = cur_num_states + num_final_states;
    states_.Resize(tot_num_states);
    Array1<StateInfo> dest = states_.Arange(cur_num_states,
                                            tot_num_states);
    Assign(final_states_, &dest);
    K2_CHECK_EQ(cur_num_states, iter_to_state_row_splits_cpu_.back());  // Remove this line.
    iter_to_state_row_splits_cpu_.push_back(tot_num_states);
  }

  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput(). */
  void Intersect() {
    NVTX_RANGE(K2_FUNC);

    FirstIter();
    Forward();
    LastIter();
  }


  /*
    Creates and returns a ragged array indexed [fsa][state][arc],
    containing the result of intersection.  (Note: we don't guarantee that
    all states are coaccessible (i.e. can reach the end); if that might be
    an issue in your case, you can call Connect() afterward.

         @param [out] arc_map_a_out  If non-NULL, the map from (arc-index of
                                  returned FsaVec) to (arc-index in a_fsas_)
                                  will be written to here.
         @param [out] arc_map_b_out  If non-NULL, the map from (arc-index of
                                  returned FsaVec) to (arc-index in b_fsas_)
                                  will be written to here.
         @return  Returns a FsaVec that is the composed result.  It may
                  contain states and/or arcs that are not co-accessible.
   */
  FsaVec FormatOutput(Array1<int32_t> *arc_map_a_out,
                      Array1<int32_t> *arc_map_b_out) {
    NVTX_RANGE(K2_FUNC);

    int32_t num_states = iter_to_state_row_splits_cpu_.back(),
             num_iters = iter_to_state_row_splits_cpu_.size() - 1,
              num_fsas = b_fsas_.Dim0();
    Array1<int32_t> row_splits1(c_, iter_to_state_row_splits_cpu_),
        row_ids1(c_, num_states);
    RowSplitsToRowIds(row_splits1, &row_ids1);

    const int32_t *b_fsas_row_ids1_data = b_fsas_.RowIds(1).Data();
    int32_t *row_ids1_data = row_ids1.Data();
    StateInfo *states_data = states_.Data();
    K2_CHECK_EQ(num_states, states_.Dim());

    /*
      currently, row_ids1 maps from state-index (in states_) to iteration
      index 0 <= t < num_iters.  We next modify it so it maps from state to
      a number that encodes (iter, FSA-index), i.e. we modify it from
           iter  ->  iter * num_fsas + fsa_idx0.
      Later we'll reorder the rows so that each FSA has all its states
      together.
    */
    K2_EVAL(c_, num_states, lambda_modify_row_ids, (int32_t i) -> void {
        int32_t iter = row_ids1_data[i];
        StateInfo info = states_data[i];
        // note: the FSA-index of the output is the same as that in b_fsas_,
        // but not necessarily in a_fsas_, thanks to b_to_a_map_.
        int32_t fsa_idx0 = b_fsas_row_ids1_data[info.b_fsas_state_idx01],
              new_row_id = iter * num_fsas + fsa_idx0;
        K2_CHECK_LT(static_cast<uint32_t>(fsa_idx0),
                    static_cast<uint32_t>(num_fsas));
        row_ids1_data[i] = new_row_id;
      });

    Array1<int32_t> row_ids2(row_ids1),  // we'll later interpret this as the 2nd
                                         // level's row-ids.
        row_splits2(c_, num_iters * num_fsas + 1);
    RowIdsToRowSplits(row_ids2, &row_splits2);

    // We'll use 'fsaiter_new2old' to effectively transpose two axes, the
    // iteration and FSA axes.  We want the FSA to be the more-slowly-varying
    // index, so we can have all states for FSA 0 first.
    Array1<int32_t> fsaiter_new2old(c_, num_iters * num_fsas);
    int32_t *fsaiter_new2old_data = fsaiter_new2old.Data();
    K2_EVAL(c_, num_iters * num_fsas, lambda_set_reordering, (int32_t i) -> void {
        int32_t fsa_idx = i / num_iters,
               iter_idx = i % num_iters;
        int32_t old_i = iter_idx * num_fsas + fsa_idx;
        fsaiter_new2old_data[i] = old_i;
      });

    Array1<int32_t> &row_ids3(arcs_row_ids_);
    Array1<int32_t> row_splits3(c_, num_states + 1);
    RowIdsToRowSplits(row_ids3, &row_splits3);

    RaggedShape layer2 = RaggedShape2(&row_splits2, &row_ids2, -1),
                layer3 = RaggedShape2(&row_splits3, &row_ids3, -1);

    Array1<int32_t> states_new2old, arcs_new2old;
    RaggedShape layer2_new = Index(layer2, fsaiter_new2old,
                                   &states_new2old),
                layer3_new = Index(layer3, states_new2old,
                                   &arcs_new2old);

    RaggedShape layer1_new = RegularRaggedShape(c_, num_fsas, num_iters);

    // We remove axis 1, which represents 'iteration-index' (this is not
    // something the user needs to know or care about).
    RaggedShape temp = ComposeRaggedShapes3(layer1_new, layer2_new, layer3_new);
    RaggedShape ans_shape = RemoveAxis(temp, 1);

    int32_t num_arcs = arcs_.Dim();
    K2_CHECK_EQ(ans_shape.NumElements(), num_arcs);

    Array1<Arc> ans_values(c_, num_arcs);

    int32_t *arc_map_a_data = nullptr,
            *arc_map_b_data = nullptr;
    if (arc_map_a_out) {
      *arc_map_a_out = Array1<int32_t>(c_, num_arcs);
      arc_map_a_data = arc_map_a_out->Data();
    }
    if (arc_map_b_out) {
      *arc_map_b_out = Array1<int32_t>(c_, num_arcs);
      arc_map_b_data = arc_map_b_out->Data();
    }

    Array1<int32_t> states_old2new = InvertPermutation(states_new2old);

    ArcInfo *arc_info_data = arcs_.Data();
    const Arc *a_arcs_data = a_fsas_.values.Data(),
              *b_arcs_data = b_fsas_.values.Data();
    Arc *arcs_out_data = ans_values.Data();
    const int32_t *arcs_new2old_data = arcs_new2old.Data(),
                *states_new2old_data = states_new2old.Data(),
                *states_old2new_data = states_old2new.Data();

    const int32_t *ans_shape_row_ids2 = ans_shape.RowIds(2).Data(),
                  *ans_shape_row_ids1 = ans_shape.RowIds(1).Data(),
               *ans_shape_row_splits1 = ans_shape.RowSplits(1).Data();

    // arc_idx012 here is w.r.t. ans_shape that currently has axes, indexed
    // [fsa][state][arc].
    K2_EVAL(c_, num_arcs, lambda_set_output_data, (int32_t new_arc_idx012) -> void {
        int32_t new_src_state_idx01 = ans_shape_row_ids2[new_arc_idx012],
                     old_arc_idx012 = arcs_new2old_data[new_arc_idx012],
                old_src_state_idx01 = states_new2old_data[new_src_state_idx01];

        ArcInfo info = arc_info_data[old_arc_idx012];
        K2_CHECK_EQ(old_src_state_idx01, info.src_ostate);
        int32_t fsa_idx0 = ans_shape_row_ids1[new_src_state_idx01];
        int32_t dest_state_idx01;
        if (info.dest_ostate >= 0) {
          dest_state_idx01 = states_old2new_data[info.dest_ostate];
        } else {
          dest_state_idx01 = ans_shape_row_splits1[fsa_idx0 + 1] - 1;
        }
        int32_t fsa_idx0x = ans_shape_row_splits1[fsa_idx0],
          dest_state_idx1 = dest_state_idx01 - fsa_idx0x,
           src_state_idx1 = new_src_state_idx01 - fsa_idx0x;

        Arc a_arc = a_arcs_data[info.a_arc_idx012],
            b_arc = b_arcs_data[info.b_arc_idx012];
        if (arc_map_a_data) arc_map_a_data[new_arc_idx012] = info.a_arc_idx012;
        if (arc_map_b_data) arc_map_b_data[new_arc_idx012] = info.b_arc_idx012;

        Arc out_arc;
        out_arc.src_state = src_state_idx1;
        out_arc.dest_state = dest_state_idx1;
        K2_CHECK_EQ(a_arc.label, b_arc.label);
        out_arc.label = a_arc.label;
        out_arc.score = a_arc.score + b_arc.score;
        arcs_out_data[new_arc_idx012] = out_arc;
      });

    return Ragged<Arc>(ans_shape, ans_values);
  }

  void Forward() {
    NVTX_RANGE(K2_FUNC);
    for (int32_t t = 0; ; t++) {
      NVTX_RANGE("LoopT");

      if (states_.Dim() * 4 > state_pair_to_state_.NumBuckets()) {
        // enlarge hash..
        state_pair_to_state_.Resize(state_pair_to_state_.NumBuckets() * 2,
                                    key_bits_);
      }

      K2_CHECK_EQ(t + 2, int32_t(iter_to_state_row_splits_cpu_.size()));

      int32_t state_begin = iter_to_state_row_splits_cpu_[t],
          state_end = iter_to_state_row_splits_cpu_[t + 1],
          num_states = state_end - state_begin;

      if (num_states == 0) {
        // It saves a little processing later to remove the last, empty,
        // iteration-index.
        iter_to_state_row_splits_cpu_.pop_back();
        break;  // Nothing left to process.
      }

      // We need to process output-states numbered state_begin..state_end-1.

      // Row 0 of num_arcs will contain the num_arcs leaving each state
      // in b in this batch; row 1 will contain (num_arcs in a * num_arcs in b).
      // If the total of row 1 is small enough and we're using the device,
      // we'll process all pairs of arcs; otherwise we'll do a logarithmic
      // search.
      Array2<int32_t> num_arcs(c_, 2, num_states + 1);

      auto num_arcs_acc = num_arcs.Accessor();
      StateInfo *states_data = states_.Data();
      const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data(),
          *b_fsas_row_splits2_data = b_fsas_.RowSplits(2).Data();

      K2_EVAL(c_, num_states, lambda_find_num_arcs, (int32_t i) -> void {
        int32_t state_idx = state_begin + i;
        StateInfo info = states_data[state_idx];
        int32_t b_fsas_state_idx01 = info.b_fsas_state_idx01,
            b_start_arc = b_fsas_row_splits2_data[b_fsas_state_idx01],
            b_end_arc =  b_fsas_row_splits2_data[b_fsas_state_idx01 + 1],
            b_num_arcs = b_end_arc - b_start_arc;
        num_arcs_acc(0, i) = b_num_arcs;
        int32_t a_fsas_state_idx01 = info.a_fsas_state_idx01,
            a_start_arc = a_fsas_row_splits2_data[a_fsas_state_idx01],
            a_end_arc =  a_fsas_row_splits2_data[a_fsas_state_idx01 + 1],
            a_num_arcs = a_end_arc - a_start_arc;
        num_arcs_acc(1, i) = b_num_arcs * a_num_arcs;
        });

      Array1<int32_t> row_splits_ab = num_arcs.Row(1),
                         num_arcs_b = num_arcs.Row(0);
      ExclusiveSum(row_splits_ab, &row_splits_ab);

      // tot_ab is total of (num-arcs from state a * num-arcs from state b).
      int32_t tot_ab = row_splits_ab[num_states],
              cutoff = 1 << 30;  // Eventually I'll make cutoff smaller, like 16384,
                           // and implement the other branch.

      const Arc *a_arcs_data = a_fsas_.values.Data(),
          *b_arcs_data = b_fsas_.values.Data();

      int32_t key_bits = key_bits_, b_state_bits = b_state_bits_,
          value_bits = 64 - key_bits;

      // `value_max` is the limit for how large values in the hash can be.
      uint64_t value_max = ((uint64_t)1) << value_bits;
      auto state_pair_to_state_acc =
          state_pair_to_state_.GetGenericAccessor(key_bits);

      K2_CHECK_GT(value_max, (uint64_t)tot_ab) << "Problem size too large "
          "for hash table... redesign or reduce problem size.";

      if (tot_ab < cutoff) {
        Array1<int32_t> row_ids_ab(c_, tot_ab);
        RowSplitsToRowIds(row_splits_ab, &row_ids_ab);

        const int32_t *row_ids_ab_data = row_ids_ab.Data(),
                   *row_splits_ab_data = row_splits_ab.Data(),
                      *num_arcs_b_data = num_arcs_b.Data();

        const int32_t *b_fsas_row_ids1_data = b_fsas_.RowIds(1).Data();

        // arcs_newstates_renumbering serves two purposes:
        //  - we'll keep some subset of the `tot_ab` arcs.
        //  - some subset of the dest-states of those arcs will be "new" dest-states
        //    that need to be assigned a state-id.
        // To avoid sequential kernels for computing Old2New() and computing New2Old(),
        // we combine those two renumberings into one.
        Renumbering arcs_newstates_renumbering(c_, tot_ab * 2);
        char *keep_arc_data = arcs_newstates_renumbering.Keep().Data(),
            *new_dest_state_data = keep_arc_data + tot_ab;
        const int32_t *a_fsas_row_splits2 = a_fsas_.RowSplits(2).Data(),
                      *b_fsas_row_splits2 = b_fsas_.RowSplits(2).Data();

        K2_EVAL(c_, tot_ab, lambda_set_keep_arc_newstate, (int32_t i) -> void {
          // state_i is the index into the block of ostates that we're
          // processing, the actual state index is state_i + state_begin.
          int32_t state_i = row_ids_ab_data[i],
              // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
              arc_pair_idx = i - row_splits_ab_data[state_i],
              state_idx = state_i + state_begin;
          StateInfo sinfo = states_data[state_idx];
          int32_t num_arcs_b = num_arcs_b_data[state_i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_splits2[sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_splits2[sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          // Not treating epsilons specially here, see documentation for
          // IntersectDevice() in [currently] fsa_algo.h.
          int keep_arc = (a_arcs_data[a_arc_idx012].label ==
                          b_arcs_data[b_arc_idx012].label);
          keep_arc_data[i] = (char)keep_arc;
          int new_dest_state = 0;
          if (keep_arc && a_arcs_data[a_arc_idx012].label != -1) {
            // investigate whether the dest-state is new (not currently allocated
            // a state-id).  We don't allocate ids for the final-state, so skip this
            // if label is -1.

            int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + sinfo.b_fsas_state_idx01 -
                                     b_arcs_data[b_arc_idx012].src_state,
                a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state;
            uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits) |
                   b_dest_state_idx01, hash_value = i;
            // If it was successfully inserted, then this arc is assigned
            // responsibility for creating the state-id for its destination
            // state.
            if (state_pair_to_state_acc.Insert(hash_key, hash_value)) {
              new_dest_state = 1;
            }
          }
          new_dest_state_data[i] = (char)new_dest_state;
          });

        // When reading the code below, remember this code is a little unusual
        // because we have combined the renumberings for arcs and new-states
        // into one.
        int32_t num_kept_arcs = arcs_newstates_renumbering.Old2New(true)[tot_ab],
                 num_kept_tot = arcs_newstates_renumbering.New2Old().Dim(),
              num_kept_states = num_kept_tot - num_kept_arcs;

        int32_t next_state_end = state_end + num_kept_states;
        iter_to_state_row_splits_cpu_.push_back(next_state_end);
        states_.Resize(next_state_end);  // Note: this Resize() won't actually reallocate each time.
        states_data = states_.Data();  // In case it changed (unlikely)

        Array1<int32_t> states_new2old =
            arcs_newstates_renumbering.New2Old().Arange(num_kept_arcs, num_kept_tot);
        const int32_t *states_new2old_data = states_new2old.Data(),
            *b_to_a_map_data = b_to_a_map_.Data(),
            *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();

        // set new elements of `states_data`, setting up the StateInfo on the next
        // frame and setting the state indexes in the hash (to be looked up when
        // creating the arcs.
        K2_EVAL(c_, num_kept_states, lambda_set_states_data, (int32_t i) -> void {
          // the reason for the "- tot_ab" is that this was in the second half of
          // the array of 'kept' of size tot_ab * 2.
          int32_t arc_i = states_new2old_data[i] - tot_ab;

          // The code below repeats what we did when processing arcs in the
          // previous lambda (now just for a small subset of arcs).

          // src_state_i is the index into the block of ostates that we're
          // processing, the actual state index is state_i + state_begin.
          int32_t src_state_i = row_ids_ab_data[arc_i],
              // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
              arc_pair_idx = arc_i - row_splits_ab_data[src_state_i],
             src_state_idx = src_state_i + state_begin;
          StateInfo src_sinfo = states_data[src_state_idx];
          int32_t num_arcs_b = num_arcs_b_data[src_state_i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_splits2[src_sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_splits2[src_sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          Arc b_arc = b_arcs_data[b_arc_idx012],
              a_arc = a_arcs_data[a_arc_idx012];
          K2_DCHECK_EQ(a_arc.label, b_arc.label);

          int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + src_sinfo.b_fsas_state_idx01 -
                                     b_arcs_data[b_arc_idx012].src_state,
                b_fsa_idx0 = b_fsas_row_ids1_data[b_dest_state_idx01],
              a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state,
              a_dest_state_idx01 = a_fsas_row_splits1_data[b_to_a_map_data[b_fsa_idx0]] +
                    a_dest_state_idx1;
          uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits) |
              b_dest_state_idx01;
          uint64_t value, *key_value_location;
          bool ans = state_pair_to_state_acc.Find(hash_key, &value,
                                                  &key_value_location);
          K2_CHECK(ans);
          K2_CHECK_EQ(value, arc_i);
          int32_t dest_state_idx = state_end + i;
          state_pair_to_state_acc.SetValue(key_value_location, hash_key,
                                             (uint64_t)dest_state_idx);

          StateInfo dest_sinfo;
          dest_sinfo.a_fsas_state_idx01 = a_dest_state_idx01;
          dest_sinfo.b_fsas_state_idx01 = b_dest_state_idx01;
          states_data[dest_state_idx] = dest_sinfo;
          });


        int32_t old_num_arcs = arcs_.Dim(),
            new_num_arcs = old_num_arcs + num_kept_arcs;
        if (static_cast<uint64_t>(tot_ab) >= value_max ||
            static_cast<uint64_t>(next_state_end) >= value_max) {
          K2_LOG(FATAL) << "Problem size is too large for this code: b_state_bits="
                        << b_state_bits_ << ", key_bits=" << key_bits_
                        << ", value_bits=" << value_bits
                        << ", value_max=" << value_max
                        << ", tot_ab=" << new_num_arcs
                        << ", next_state_end=" << next_state_end;
        }

        arcs_.Resize(new_num_arcs);
        arcs_row_ids_.Resize(new_num_arcs);
        ArcInfo *arcs_data = arcs_.Data();
        int32_t *arcs_row_ids_data = arcs_row_ids_.Data();

        const int32_t *arcs_new2old_data =
            arcs_newstates_renumbering.New2Old().Data();

        K2_EVAL(c_, num_kept_arcs, lambda_set_arc_info, (int32_t new_arc_i) -> void {
          // 0 <= old_arc_i < tot_ab.
          int32_t old_arc_i = arcs_new2old_data[new_arc_i];

          // The code below repeats what we did when processing arcs in the
          // previous lambdas (we do this for all arcs that were kept).

          // src_state_i is the index into the block of ostates that we're
          // processing, the actual state index is src_state_i + state_begin.
          int32_t src_state_i = row_ids_ab_data[old_arc_i];
          // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
          int32_t arc_pair_idx = old_arc_i - row_splits_ab_data[src_state_i],
              src_state_idx = src_state_i + state_begin;
          StateInfo src_sinfo = states_data[src_state_idx];
          int32_t num_arcs_b = num_arcs_b_data[src_state_i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_splits2[src_sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_splits2[src_sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          Arc b_arc = b_arcs_data[b_arc_idx012],
              a_arc = a_arcs_data[a_arc_idx012];
          K2_DCHECK_EQ(a_arc.label, b_arc.label);

          int32_t dest_state_idx = -1;
          if (a_arc.label != -1) {
            int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + src_sinfo.b_fsas_state_idx01 -
                                     b_arcs_data[b_arc_idx012].src_state,
              a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state;
            uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits) +
                b_dest_state_idx01;

            uint64_t value;
            bool ans = state_pair_to_state_acc.Find(hash_key, &value);
            dest_state_idx = static_cast<uint32_t>(value);
          }  // else leave it at -1, it's a final-state and we allocate their
             // state-ids at the end.

          ArcInfo info;
          info.src_ostate = src_state_idx;
          info.dest_ostate = dest_state_idx;
          info.a_arc_idx012 = a_arc_idx012;
          info.b_arc_idx012 = b_arc_idx012;
          arcs_data[old_num_arcs + new_arc_i] = info;
          arcs_row_ids_data[old_num_arcs + new_arc_i] = src_state_idx;
          });

      } else {
        ExclusiveSum(num_arcs, &num_arcs, 1);  // sum
        // Plan to implement binary search here at some point, to get arc ranges...
        K2_LOG(FATAL) << "Not implemented yet, see code..";
      }
    }
  }

  ~DeviceIntersector() {
    // Prevent crash in destructor of hash (at exit, it still contains values, by design).
    state_pair_to_state_.Destroy();
  }


  ContextPtr c_;
  FsaVec a_fsas_;  // a_fsas_: decoding graphs, with same Dim0() as
                    // b_fsas_. Note: a_fsas_ has 3 axes.

  FsaVec b_fsas_;

  // map from fsa-index in b_fsas_ to the fsa-index in a_fsas_ that we want to
  // intersect it with.
  Array1<int32_t> b_to_a_map_;

  // iter_to_state_row_splits_cpu_, which is resized on each iteration of the
  // algorithm, is a row-splits array that maps from iteration index to
  // state_idx (index into states_).
  std::vector<int32_t> iter_to_state_row_splits_cpu_;

  // states_ is a resizable array of StateInfo that conceptually is the elements
  // of a ragged array indexed [iter][state], with row_splits1 ==
  // iter_to_state_row_splits_cpu_.
  Array1<StateInfo> states_;

  // final_states_ is an array of StateInfo, of dimension <= b_fsas_.Dim0(),
  // that contains the final state-pairs of each composed FSA that has initial
  // state-pairs.  These will be added to the end of states_ after composition
  // has finished.
  Array1<StateInfo> final_states_;

  // arcs_ is a resizable array of ArcInfo that conceptually is the elements
  // of a ragged array indexed [iter][state][arc], with row_splits1 == iter_to_state_row_splits_cpu_
  // and row_ids2 == arcs_row_ids_.
  Array1<ArcInfo> arcs_;

  // arcs_row_ids_, which always maintained as having the same size as `arcs_`,
  // maps from the output arc to the corresponding ostate index that the arc
  // leaves from (index into states_).  Actually this may be redu
  Array1<int32_t> arcs_row_ids_;



  // The hash maps from state-pair, as:
  //   state_pair = (a_fsas_state_idx1 << b_state_bits_) + b_fsas_state_idx01
  //
  // The number of bits in the key (max bits set in `state_pair`) is
  // key_bits_ == b_state_bits_ + HighestBitSet(a_fsas_.MaxSize(1)).
  // The number of bits in the value is 64 minus this; we'll crash if
  // the number of states ends up being too large to store in this
  // value.
  int32_t b_state_bits_;  // == HighestBitSet(b_fsas_.TotSize(1)).
  int32_t key_bits_;  // b_state_bits_ + HighestBitSet(a_fsas_.MaxSize(1)).


  Hash state_pair_to_state_;
};



FsaVec IntersectDevice(FsaVec &a_fsas, int32_t properties_a,
                     FsaVec &b_fsas, int32_t properties_b,
                     const Array1<int32_t> &b_to_a_map,
                     Array1<int32_t> *arc_map_a,
                     Array1<int32_t> *arc_map_b) {
  NVTX_RANGE("IntersectDevice");
  K2_CHECK_NE(properties_a & kFsaPropertiesValid, 0);
  K2_CHECK_NE(properties_b & kFsaPropertiesValid, 0);
  K2_CHECK_EQ(a_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_to_a_map.Dim(), b_fsas.Dim0());
  K2_CHECK_LT(static_cast<uint32_t>(MaxValue(b_to_a_map)),
              static_cast<uint32_t>(a_fsas.Dim0()));

  DeviceIntersector intersector(a_fsas, b_fsas, b_to_a_map);
  intersector.Intersect();
  return intersector.FormatOutput(arc_map_a, arc_map_b);
}
}  // namespace k2
