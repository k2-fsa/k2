/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef K2_WITH_CUDA
#include <cooperative_groups.h>
#endif

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
os << "ArcInfo{" << a.a_arc_idx012 << "," << a.b_arc_idx012 << "}";
  return os;
}
*/


}  // namespace intersect_internal

using namespace intersect_internal;  // NOLINT

/*
   Intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.

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
       @param [in] sorted_match_a  If true, the arcs of a_fsas arcs must be sorted
                           by label (checked by calling code via properties), and
                           we'll use a matching approach that requires this.

     Does not fully check its args (see wrapping code).  After constructing this object,
     call Intersect() and then FormatOutput().
   */
  DeviceIntersector(FsaVec &a_fsas, FsaVec &b_fsas,
                    const Array1<int32_t> &b_to_a_map,
                    bool sorted_match_a):
      c_(a_fsas.Context()),
      a_fsas_(a_fsas),
      sorted_match_a_(sorted_match_a),
      b_fsas_(b_fsas),
      b_to_a_map_(b_to_a_map),
      a_states_multiple_(b_fsas_.TotSize(1) | 1) {
    NVTX_RANGE(K2_FUNC);
    int32_t num_key_bits = NumBitsNeededFor(a_fsas_.shape.MaxSize(1) *
                                            (int64_t)a_states_multiple_);
    // in future the accessor for num_key_bits==32 may be more efficient, and
    // there's no point leaving >32 bits for the value since our arrays don't
    // support that and also we can't have more values than the #keys
    // since the values are consecutive.
    if (num_key_bits < 32)
      num_key_bits = 32;

    // We may want to tune this default hash size eventually.
    // We will expand the hash as needed.
    int32_t hash_size = 4 * RoundUpToNearestPowerOfTwo(b_fsas.NumElements()),
        min_hash_size = 1 << 16;
    if (hash_size < min_hash_size)
      hash_size = min_hash_size;
    // caution: also use hash_size in FirstIter() as default size of various arrays.
    int32_t num_value_bits = std::max<int32_t>(NumBitsNeededFor(hash_size - 1),
                                               64 - num_key_bits);
    state_pair_to_state_ = Hash(c_, hash_size, num_key_bits,
                                num_value_bits);


    K2_CHECK(c_->IsCompatible(*b_fsas.Context()));
    K2_CHECK(c_->IsCompatible(*b_to_a_map.Context()));
  }


  void FirstIter() {
    NVTX_RANGE(K2_FUNC);
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
    NVTX_RANGE(K2_FUNC);
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
    FirstIter();
    if (sorted_match_a_)
      ForwardSortedA();
    else
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
    int32_t num_key_bits = state_pair_to_state_.NumKeyBits(),
        num_value_bits = state_pair_to_state_.NumValueBits();
    if (num_key_bits + num_value_bits == 64) {
      return FormatOutputTpl<Hash::GenericAccessor>(arc_map_a_out,
                                                    arc_map_b_out);
    } else {
      return FormatOutputTpl<Hash::PackedAccessor>(arc_map_a_out,
                                                   arc_map_b_out);
    }
  }

  template <typename AccessorT>
  FsaVec FormatOutputTpl(Array1<int32_t> *arc_map_a_out,
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
        K2_DCHECK_LT(static_cast<uint32_t>(fsa_idx0),
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
    RaggedShape layer2_new = Index(layer2, 0, fsaiter_new2old,
                                   &states_new2old),
                layer3_new = Index(layer3, 0, states_new2old,
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

    const int32_t *b_fsas_row_ids2_data = b_fsas_.RowIds(2).Data();

    int32_t a_states_multiple = a_states_multiple_;
    AccessorT state_pair_to_state_acc =
        state_pair_to_state_.GetAccessor<AccessorT>();

    // arc_idx012 here is w.r.t. ans_shape that currently has axes indexed
    // [fsa][state][arc].
    K2_EVAL(c_, num_arcs, lambda_set_output_data, (int32_t new_arc_idx012) -> void {
        int32_t new_src_state_idx01 = ans_shape_row_ids2[new_arc_idx012],
              old_arc_idx012 = arcs_new2old_data[new_arc_idx012];

        ArcInfo info = arc_info_data[old_arc_idx012];
        int32_t fsa_idx0 = ans_shape_row_ids1[new_src_state_idx01];
        Arc a_arc = a_arcs_data[info.a_arc_idx012],
            b_arc = b_arcs_data[info.b_arc_idx012];
        if (arc_map_a_data) arc_map_a_data[new_arc_idx012] = info.a_arc_idx012;
        if (arc_map_b_data) arc_map_b_data[new_arc_idx012] = info.b_arc_idx012;

        int32_t new_dest_state_idx01;  // index of the dest_state w.r.t
                                       // ans_shape
        if (a_arc.label == -1) {
          new_dest_state_idx01 = ans_shape_row_splits1[fsa_idx0 + 1] - 1;
        } else {
          // first work out old_dest_state_idx01, which is the index (into
          // states_) of the dest-state.
          int32_t b_src_state_idx01 = b_fsas_row_ids2_data[info.b_arc_idx012],
              b_dest_state_idx01 = b_src_state_idx01 + b_arc.dest_state - b_arc.src_state,
              a_dest_state_idx1 = a_arc.dest_state;
          uint64_t hash_key = (((uint64_t)a_dest_state_idx1) * a_states_multiple) +
              b_dest_state_idx01;
          uint64_t value = 0;
          bool ans = state_pair_to_state_acc.Find(hash_key, &value);
          K2_CHECK_EQ(ans, true);
          int32_t old_dest_state_idx01 = static_cast<uint32_t>(value);
          new_dest_state_idx01 = states_old2new_data[old_dest_state_idx01];
        }
        int32_t fsa_idx0x = ans_shape_row_splits1[fsa_idx0],
            dest_state_idx1 = new_dest_state_idx01 - fsa_idx0x,
            src_state_idx1 = new_src_state_idx01 - fsa_idx0x;

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
          cutoff = 1 << 30;  // Eventually I'll make cutoff smaller, like
                             // 16384, and implement the other branch.

      if (tot_ab > cutoff) {
        K2_LOG(FATAL) << "Problem size is too large for un-sorted intersection, "
            "please make sure one input is arc-sorted and use sorted_match_a=true.";
      }
      // The following is a bound on how big we might need the hash to be, assuming
      // all arc-pairs match, which of course they won't, but it's safe.  For large
      // problems you should be using sorted_match_a=true.
      PossiblyResizeHash(4 * (states_.Dim() + tot_ab),
                         states_.Dim() + tot_ab);

      int32_t num_key_bits = state_pair_to_state_.NumKeyBits(),
          num_value_bits = state_pair_to_state_.NumValueBits();
      if (num_key_bits == 32 && num_value_bits == 32) {
        ForwardOneIter<Hash::Accessor<32> >(t, tot_ab, num_arcs_b,
                                            row_splits_ab);
      } else if (num_key_bits + num_value_bits == 64) {
        ForwardOneIter<Hash::GenericAccessor>(t, tot_ab, num_arcs_b,
                                              row_splits_ab);
      } else {
        ForwardOneIter<Hash::PackedAccessor>(t, tot_ab, num_arcs_b,
                                             row_splits_ab);
      }
    }
  }
  /*
    This is a piece of the code in Forward() that was broken out because it
    needs to be templated on the hash accessor type.  It does the last
    part of the intersection algorithm for a single iteration.
        @param [in] t   The iteration of the algorithm, dictates the
                    batch of states we need to process.
        @param [in] tot_ab  The total number of arcs we need to process,
                    which equals the sum of
                    (total number of arcs leaving state in a) *
                    (total number of arcs leaving state in b) for each
                    state-pair (a,b) that we need to process.
        @param [in] num_arcs_b  An array indexed by an index i such
                    that (i + state_begin) is an index into states_,
                    that gives the number of arcs leaving the "b"
                    state in that state-pair (from b_fsas_).  The
                    last element of this array is undefined.
        @param [in] row_splits_ab  The exclusive-sum of the products
                    (total number of arcs leaving state in a) *
                    (total number of arcs leaving state in b) for each
                    state-pair (a,b) that we need to process; dimension
                    is 1 + (num states that we need to process).
  */
  template <typename HashAccessorT>
  void ForwardOneIter(int32_t t, int32_t tot_ab,
                      const Array1<int32_t> &num_arcs_b,
                      const Array1<int32_t> &row_splits_ab) {
    NVTX_RANGE(K2_FUNC);
    int32_t state_begin = iter_to_state_row_splits_cpu_[t],
        state_end = iter_to_state_row_splits_cpu_[t + 1];

    const Arc *a_arcs_data = a_fsas_.values.Data(),
        *b_arcs_data = b_fsas_.values.Data();

    int32_t key_bits = state_pair_to_state_.NumKeyBits(),
        a_states_multiple = a_states_multiple_,
        value_bits = state_pair_to_state_.NumValueBits();

    // `value_max` is the limit for how large values in the hash can be.
    uint64_t value_max = ((uint64_t)1) << value_bits;
    HashAccessorT state_pair_to_state_acc =
        state_pair_to_state_.GetAccessor<HashAccessorT>();

    // Note: we can actually resolve the next failure fairly easily now;
    // we'll do it when needed.
    K2_CHECK_GT(value_max, (uint64_t)tot_ab) << "Problem size too large "
        "for hash table... redesign or reduce problem size.";

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
    StateInfo *states_data = states_.Data();
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
          uint64_t hash_key = (((uint64_t)a_dest_state_idx1) * a_states_multiple) +
              b_dest_state_idx01, hash_value = i;
          // If it was successfully inserted, then this arc is assigned
          // responsibility for creating the state-id for its destination
          // state.
          // The value `hash_value` that we insert into the hash is temporary,
          // and will be replaced below with the index into states_.
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
    states_data = states_.Data();   // In case it changed (unlikely)

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
        uint64_t hash_key = (((uint64_t)a_dest_state_idx1) * a_states_multiple) +
            b_dest_state_idx01;
        uint64_t value, *key_value_location = nullptr;
        bool ans = state_pair_to_state_acc.Find(hash_key, &value,
                                                &key_value_location);
        K2_DCHECK(ans);
        K2_DCHECK_EQ(value, (uint64_t)arc_i);
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
      K2_LOG(FATAL) << "Problem size is too large for this code: a_states_multiple="
                    << a_states_multiple_ << ", key_bits=" << key_bits
                    << ", value_bits=" << value_bits
                    << ", value_max=" << value_max
                    << ", tot_ab=" << tot_ab
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

        //int32_t dest_state_idx = -1;
        if (a_arc.label != -1) {
          int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
              b_dest_state_idx01 = b_dest_state_idx1 + src_sinfo.b_fsas_state_idx01 -
              b_arcs_data[b_arc_idx012].src_state,
              a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state;
          uint64_t hash_key = (((uint64_t)a_dest_state_idx1) * a_states_multiple) +
              b_dest_state_idx01;

          uint64_t value = 0;
          bool ans = state_pair_to_state_acc.Find(hash_key, &value);
          // dest_state_idx = static_cast<uint32_t>(value);
        }  // else leave dest_state_idx at -1; it's a final-state and we
        // allocate their state-ids at the end.

        // Actually we no longer need dest_state_idx, it will be obtained
        // directly from the hash when we format the output.
        ArcInfo info;
        info.a_arc_idx012 = a_arc_idx012;
        info.b_arc_idx012 = b_arc_idx012;
        arcs_data[old_num_arcs + new_arc_i] = info;
        arcs_row_ids_data[old_num_arcs + new_arc_i] = src_state_idx;
      });
  }

  /*
    This function ensures that the hash `state_pair_to_state_` has an array
    with at least `min_num_buckets` buckets, and NumValueBits() large enough to contain
    at least `min_supported_values` values

    The number of bits allocated for the key will not be changed (this was
    set to the required value in the constructor).

      @param [in] min_num_buckets  The minimum number of buckets required;
                    the actual number chosen will be a power of 2 that is >=
                    min_num_buckets.  CAUTION: this number should be
                    considerably larger than the maximum number of key/value
                    pairs you might want to store in the hash (or it will get
                    too full).
      @param [in] min_supported_values  The user declares that the
                    hash must have enough bits allocated to values that
                    it can store values 0 <= v < min_supported_values.
                    In general this requires that
                    min_supported_values < (1 << state_pair_to_state_.NumValueBits()),
                    the strictly-less-than being necessary because (1<<num_value_bits)-1 is not
                    allowed as a value if (1<<num_key_bits)-1 is allowed as a key,
                    which condition we are too lazy to check.
   */
  void PossiblyResizeHash(int32_t min_num_buckets,
                          int32_t min_supported_values) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GE(min_num_buckets, 0);
    int32_t cur_num_buckets = state_pair_to_state_.NumBuckets(),
        cur_num_key_bits = state_pair_to_state_.NumKeyBits(),
        cur_num_value_bits = state_pair_to_state_.NumValueBits(),
        num_buckets = std::max<int32_t>(
            RoundUpToNearestPowerOfTwo(min_num_buckets),
            cur_num_buckets),
        num_value_bits = std::max<int32_t>(
            NumBitsNeededFor(min_supported_values),
            cur_num_value_bits);
    if (num_value_bits != cur_num_value_bits ||
        num_buckets != cur_num_buckets) {
      state_pair_to_state_.Resize(num_buckets,
                                  cur_num_key_bits,
                                  num_value_bits);
    }
  }

  void ForwardSortedA() {
    NVTX_RANGE(K2_FUNC);
    for (int32_t t = 0; ; t++) {
      K2_CHECK_EQ(t + 2, int32_t(iter_to_state_row_splits_cpu_.size()));
      int32_t state_begin = iter_to_state_row_splits_cpu_[t],
          state_end = iter_to_state_row_splits_cpu_[t + 1],
          num_states = state_end - state_begin;
      if (num_states == 0) {
        // It saves a little processing later to remove the last, empty,
        // iteration-index.
        iter_to_state_row_splits_cpu_.pop_back();
        break;
      }
      // We need to process output-states numbered state_begin..state_end-1.
      // num_arcs_b will contain the number of arcs leaving the state in b_fsas_,
      // i.e. the state with index StateInfo::b_fsas_state_idx01.
      Array1<int32_t> num_arcs_b(c_, num_states + 1);
      int32_t *num_arcs_b_data = num_arcs_b.Data();

      StateInfo *states_data = states_.Data();
      const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data(),
          *b_fsas_row_splits2_data = b_fsas_.RowSplits(2).Data();

      K2_EVAL(c_, num_states, lambda_find_num_arcs_b, (int32_t i) -> void {
        int32_t state_idx = state_begin + i;
        StateInfo info = states_data[state_idx];
        int32_t b_fsas_state_idx01 = info.b_fsas_state_idx01,
            b_start_arc = b_fsas_row_splits2_data[b_fsas_state_idx01],
            b_end_arc =  b_fsas_row_splits2_data[b_fsas_state_idx01 + 1],
            b_num_arcs = b_end_arc - b_start_arc;
        num_arcs_b_data[i] = b_num_arcs;
        });

      ExclusiveSum(num_arcs_b, &num_arcs_b);
      int32_t num_b_arcs = num_arcs_b.Back();

      Array1<int32_t> b_arc_to_state(c_, num_b_arcs);
      RowSplitsToRowIds(num_arcs_b, &b_arc_to_state);
      int32_t *b_arc_to_state_data = b_arc_to_state.Data();

      /*
        We now know, for each state-pair we need to
        process, the total number of arcs leaving the state in b.
        We need to figure out the range of matching arcs leaving
        the state in a.
      */
      Array1<int32_t> first_matching_a_arc_idx012(c_, num_b_arcs);
      int32_t *first_matching_a_arc_idx012_data =
          first_matching_a_arc_idx012.Data();
      // The + 1 is because we'll do an exclusive sum.
      Array1<int32_t> num_matching_a_arcs(c_, num_b_arcs + 1);
      int32_t *num_matching_a_arcs_data = num_matching_a_arcs.Data();

      const Arc *a_arcs_data = a_fsas_.values.Data(),
          *b_arcs_data = b_fsas_.values.Data();
      int32_t key_bits = state_pair_to_state_.NumKeyBits(),
          value_bits = state_pair_to_state_.NumValueBits();

      if (c_->GetDeviceType() == kCuda) {
#ifdef K2_WITH_CUDA
        namespace cg = cooperative_groups;
        constexpr int log_thread_group_size = 2,
            thread_group_size = (1 << log_thread_group_size);  // 4
        static_assert(thread_group_size > 1, "Bad thread_group_size");
        // the "* 2" below is because pairs of thread groups handle the
        // (beginning, end) of ranges of arcs in a_fsas_; and we need
        // these groups to be within the same warp so we can sync them.
        static_assert(thread_group_size * 2 <= 32,
                      "thread_group_size too large");

        auto lambda_find_ranges = [=] __device__(
            cg::thread_block_tile<thread_group_size> g,  // or auto g..
            int32_t *shared_data,  // points to shared data for this block of
                                   // threads
            int32_t idx01_doubled) -> void {

          // thread_group_type is 0 if we're finding the beginning of the range
          // of matching arcs, and 1 if we're finding the end of the range of
          // matching arcs.
          // 0 <= idx01 < num_b_arcs is an index into the list of arcs we're
          // processing; the array's shape has (row_splits,row_ids) ==
          // (num_arcs_b, b_arc_to-state).
          int32_t arc_idx01 = idx01_doubled / 2,
              thread_group_type = idx01_doubled % 2;

          // the idx01 is into the list of arcs in b that we're processing.
          // 0 <= state_idx0 < num_states.
          int32_t state_idx0 = b_arc_to_state_data[arc_idx01],
              arc_idx1x = num_arcs_b_data[state_idx0],
              arc_idx1 = arc_idx01 - arc_idx1x;
          // state_idx is an index into states_.
          int32_t state_idx = state_begin + state_idx0;
          StateInfo info = states_data[state_idx];
          int32_t b_begin_arc_idx01x = b_fsas_row_splits2_data[info.b_fsas_state_idx01],
              b_arc_idx012 = b_begin_arc_idx01x + arc_idx1;
          // ignore the apparent name mismatch setting b_arc_idx012 above;
          // arc_idx1 is an idx1 w.r.t. a different array than b_fsas_.
          K2_DCHECK_LT(b_arc_idx012, b_fsas_row_splits2_data[info.b_fsas_state_idx01+1]);

          int32_t a_begin_arc_idx012 = a_fsas_row_splits2_data[info.a_fsas_state_idx01],
              a_end_arc_idx012 = a_fsas_row_splits2_data[info.a_fsas_state_idx01 + 1];

          int32_t thread_idx = g.thread_rank(),
              num_threads = g.size();  // = thread_group_size.

          // We convert to uint64_t so we can add 1 without wrapping around;
          // this way, even-numbered thread groups (i.e. groups of size
          // thread_group_size) find the beginning of the range of arcs in a,
          // and odd-numbered thread groups find the end of the range of arcs.
          uint64_t label = static_cast<uint64_t>(static_cast<uint32_t>(
                                        b_arcs_data[b_arc_idx012].label)) +
                           static_cast<uint64_t>(thread_group_type);

          // We are now searching for the lowest arc-index i in the range
          // a_begin_arc_idx012 <= i <= a_end_arc_idx012, where
          // arcs_data[i].label >= `label`, where we treat the labels of arcs
          // indexed i >= a_end_arc_idx012 as infinitely large.
          int32_t range_len = a_end_arc_idx012 - a_begin_arc_idx012,  // > 0
              log_range_len = 31 - __clz(range_len | 1),
                  num_iters = 1 + log_range_len / log_thread_group_size;

          // suppose log_thread_group_size=2, thread_group_size=4.
          // Then:
          //  0 <= range_len < 4  -> num_iters is 1
          //  4 <= range_len < 16  -> num_iters is 2
          // Note: at 4 and 16, we need num_iters to be (2,3)
          // respectively because a_end_arc_idx012 is a value we need
          // to include in the search.


          // "per_thread_range" is the length of the interval of arcs that each thread
          // 0 <= thread_idx < num_threads is currently responsible for.
          // At this point, the group of threads is searching an interval
          // [interval_start ... interval_start+(per_thread_range*thread_group_size)].
          // for the lowest index i such that
          //   (i>= a_end_arc_idx012 ? UINT64_MAX : (uint32_t)a_arcs_data[i]) >= label
          // and such an i must exist in this range because the range includes a_end_arc_idx012
          // (checked in K2_DCHECK below)
          int32_t per_thread_range =  1 << ((num_iters - 1) * log_thread_group_size); // > 0
          int32_t interval_start = a_begin_arc_idx012;

          K2_DCHECK_GT(interval_start + per_thread_range * thread_group_size,
                       a_end_arc_idx012);

          while (per_thread_range > 0) {
            // this_thread_start is the beginning of the range of arcs that this
            // thread is responsible for searching.
            int32_t this_thread_start = interval_start +
                (thread_idx * per_thread_range),
                this_thread_last = this_thread_start + per_thread_range - 1;
            // last_label is the label on the last arc in the range that this
            // thread is responsible for.  We ensure that the range of arcs
            // we are searching (which, remember, includes a_end_arc_idx012)
            // always have at least one arc whose label (taken as +infty
            // for out-of-range arcs) is >= `label`.  So `last_label` for
            // the last thread will always be >= `label`.
            uint64_t last_label = (this_thread_last >= a_end_arc_idx012 ?
                                    static_cast<uint64_t>(-1) :
                                    static_cast<uint64_t>(static_cast<uint32_t>(
                                        a_arcs_data[this_thread_last].label))),
                prev_last_label = g.shfl_up(last_label, 1);
            // Note: prev_last_label is the last_label for the previous thread,
            // and it's a don't-care value which will be ignored if this
            // thread_idx == 0.

            // Exactly one thread in the group will satisfy the following
            // conditions.  Note: for the last thread in the thread group,
            // the condition "label < end_label" will always be true, because
            // label < UINT64_MAX.
            if ((thread_idx == 0 || prev_last_label < label) &&
                last_label >= label) {
              *shared_data = this_thread_start;
            }
            g.sync();
            interval_start = *shared_data;  // broadcast to all threads..
            per_thread_range = per_thread_range >> log_thread_group_size;
          }
          // OK, now all threads in the group should share the variable
          // `interval_start`.  We construct a thread_block_tile of double
          // the size, so we can broadcast the lower and upper bounds of
          // the range of matching arcs in a (look above for "thread_group_type"
          // for more explanation).
          cg::thread_block_tile<thread_group_size*2>
            g_double = cg::tiled_partition<thread_group_size*2>(cg::this_thread_block());
          int32_t lower_bound, upper_bound;
          if (thread_idx == 0) {  // only the 1st thread from each of the 2 groups
                                  // participates.
            lower_bound = g_double.shfl(interval_start, 0);
            upper_bound = g_double.shfl(interval_start, thread_group_size);

            if (g_double.thread_rank() == 0) {  // equiv. to:
                                                // (thread_group_type == 0)
              /*
              K2_DCHECK_LE(lower_bound, upper_bound);
              K2_DCHECK_LE(a_begin_arc_idx012, lower_bound);
              K2_DCHECK_LE(upper_bound, a_end_arc_idx012);
              K2_DCHECK(lower_bound == a_end_arc_idx012 ||
                        a_arcs_data[lower_bound].label >= label);
              K2_DCHECK(lower_bound == a_begin_arc_idx012 ||
                        a_arcs_data[lower_bound - 1].label < label);
              K2_DCHECK(upper_bound == a_end_arc_idx012 ||
                uint32_t(a_arcs_data[upper_bound].label) > uint32_t(label)); */
              if (upper_bound != a_begin_arc_idx012) {
                K2_DCHECK_LE(uint32_t(a_arcs_data[upper_bound - 1].label),
                             uint32_t(label));
              }
              first_matching_a_arc_idx012_data[arc_idx01] = lower_bound;
            } else {
              // g_double.thread_rank() == thread_group_size
              num_matching_a_arcs_data[arc_idx01] = upper_bound - lower_bound;
            }
          }
        };
        EvalGroupDevice<thread_group_size, int32_t>(
            c_, num_b_arcs * 2, lambda_find_ranges);
#else
        K2_LOG(FATAL) << "Unreachable code";
#endif
      } else {
        // Use regular binary search.
        K2_EVAL(c_, num_b_arcs, lambda_find_ranges_cpu, (int32_t arc_idx01) -> void {
            // the idx01 is into the list of arcs in b that we're processing..
            // 0 <= state_idx0 < num_states.
            // state_idx is an index into states_.
            int32_t state_idx0 = b_arc_to_state_data[arc_idx01],
                arc_idx1x = num_arcs_b_data[state_idx0],
                arc_idx1 = arc_idx01 - arc_idx1x,
                state_idx = state_begin + state_idx0;
            StateInfo info = states_data[state_idx];
            int32_t
                b_begin_arc_idx01x = b_fsas_row_splits2_data[info.b_fsas_state_idx01],
                b_arc_idx012 = b_begin_arc_idx01x + arc_idx1;
            // ignore the apparent name mismatch setting b_arc_idx012 above;
            // arc_idx1 is an idx1 w.r.t. a different array than b_fsas_.
            K2_DCHECK_LT(b_arc_idx012, b_fsas_row_splits2_data[info.b_fsas_state_idx01 + 1]);
            int32_t a_begin_arc_idx012 = a_fsas_row_splits2_data[info.a_fsas_state_idx01],
                a_end_arc_idx012 = a_fsas_row_splits2_data[info.a_fsas_state_idx01 + 1];
            uint32_t label = static_cast<uint32_t>(b_arcs_data[b_arc_idx012].label);

            int32_t begin = a_begin_arc_idx012,
                end = a_end_arc_idx012;
            // We are looking for the first index begin <= i < end such that
            //     a_arcs[i].label >= label.
            while (begin < end) {
              int32_t mid = (begin + end) / 2;
              assert(mid < end);  // temp?
              uint32_t a_label = uint32_t(a_arcs_data[mid].label);
              if (a_label < label) {
                begin = mid + 1;
              } else {
                end = mid;
              }
            }
            if (begin < a_end_arc_idx012) {
              K2_CHECK_GE((uint32_t)a_arcs_data[begin].label, label);
            }
            if (begin - 1 > a_begin_arc_idx012) {
              K2_CHECK_LT((uint32_t)a_arcs_data[begin-1].label, label);
            }

            // "range_begin" is the "begin" of the possibly-empty range of arc-indexes
            // in a that matches `label`
            int32_t range_begin = begin, range_end = begin;
            // The following linear search will probably be faster than
            // logarithmic search in the normal case where there are not many
            // matching arcs.  In the unusual case where there are many matching
            // arcs per state, it won't dominate the running time of the entire
            // algorithm.
            while (range_end < a_end_arc_idx012 &&
                   uint32_t(a_arcs_data[range_end].label) == label)
              range_end++;
            first_matching_a_arc_idx012_data[arc_idx01] = range_begin;
            num_matching_a_arcs_data[arc_idx01] = range_end - range_begin;
          });
      }

      ExclusiveSum(num_matching_a_arcs, &num_matching_a_arcs);
      int32_t tot_matched_arcs = num_matching_a_arcs.Back();

      {
        int32_t max_possible_states = states_.Dim() + tot_matched_arcs;
        PossiblyResizeHash(4 * max_possible_states,
                           max_possible_states);
      }


      int32_t num_key_bits = state_pair_to_state_.NumKeyBits(),
          num_value_bits = state_pair_to_state_.NumValueBits();
      if (num_key_bits == 32 && num_value_bits == 32) {
        ForwardSortedAOneIter<Hash::Accessor<32> >(
            t, num_arcs_b, b_arc_to_state,
            num_matching_a_arcs, first_matching_a_arc_idx012,
            tot_matched_arcs);
      } else if (num_key_bits + num_value_bits == 64) {
        ForwardSortedAOneIter<Hash::GenericAccessor>(
            t, num_arcs_b, b_arc_to_state,
            num_matching_a_arcs, first_matching_a_arc_idx012,
            tot_matched_arcs);
      } else {
        ForwardSortedAOneIter<Hash::PackedAccessor>(
            t, num_arcs_b, b_arc_to_state,
            num_matching_a_arcs, first_matching_a_arc_idx012,
            tot_matched_arcs);
      }
    }
  }

  /*
    This is some code that was broken out of ForwardSortedA() because it needed
    to be templated on the hash accessor type.  Does the last part of a single
    iteration of the algorithm.

      @param [in] t    The iteration index >= 0, representing the batch of
                       states that we are processing arcs leaving from.
      @param [in] num_arcs_b_row_splits   An array of shape equal to
                       1 + num_states, where num_states is the number of
                       states we're processing on this iteration (see the
                       variable in the code), with is the exclusive-sum
                       of the number of arcs leaving the states in b
                       (from "b" members of the state-pair we're processing)
      @param [in] b_arc_to_state   The result of turning `num_arcs_b_row_splits`
                       into a row-ids array.  Each element corresponds to an arc
                       in b_fsas_ that we are processing.
      @param [in] matching_a_arcs_row_splits   An array of size b_arc_to_state.Dim() + 1,
                       which is the exclusive sum of the number of matching arcs
                       in a_fsas_ for a particular arc in b_fsas_ that we are
                       processing.
      @param [in] first_matching_a_arc_idx012  An array of size b_arc_to-state.Dim(),
                       giving the index of the first matching arc in a_fsas_ that
                       matches the corresponding arc in b_fsas_.
      @param [in] tot_matched_arcs  Must equal matching_a_arcs_row_splits.Back()
   */
  template <typename HashAccessorT>
  void ForwardSortedAOneIter(
      int32_t t,
      const Array1<int32_t> &num_arcs_b_row_splits,
      const Array1<int32_t> &b_arc_to_state,
      const Array1<int32_t> &matching_a_arcs_row_splits,
      const Array1<int32_t> &first_matching_a_arc_idx012,
      int32_t tot_matched_arcs) {
      NVTX_RANGE(K2_FUNC);

      HashAccessorT state_pair_to_state_acc =
          state_pair_to_state_.GetAccessor<HashAccessorT>();
      const Arc *a_arcs_data = a_fsas_.values.Data(),
          *b_arcs_data = b_fsas_.values.Data();
      int32_t state_begin = iter_to_state_row_splits_cpu_[t],
          state_end = iter_to_state_row_splits_cpu_[t + 1],
          a_states_multiple = a_states_multiple_;

      Array1<int32_t> matched_arc_to_b_arc(c_, tot_matched_arcs);
      RowSplitsToRowIds(matching_a_arcs_row_splits, &matched_arc_to_b_arc);
      const int32_t *matched_arc_to_b_arc_data = matched_arc_to_b_arc.Data(),
          *b_arc_to_state_data = b_arc_to_state.Data(),
          *num_arcs_b_row_splits_data = num_arcs_b_row_splits.Data();

      Renumbering new_state_renumbering(c_, tot_matched_arcs);
      // We'll write '1' where the arc-pair leads to a new state (exactly one
      // such arc will have a '1', for each newly produced state.
      char *new_state_renumbering_keep_data = new_state_renumbering.Keep().Data();

      int32_t old_num_arcs = arcs_.Dim(),
          new_num_arcs = old_num_arcs + tot_matched_arcs;

      arcs_.Resize(new_num_arcs);
      arcs_row_ids_.Resize(new_num_arcs);
      ArcInfo *new_arcs_data = arcs_.Data() + old_num_arcs;
      int32_t *new_arcs_row_ids_data = arcs_row_ids_.Data() + old_num_arcs;

      // `hash_keys_value_locations` will be written to only for
      // arcs that are responsible for creating a new state; it points to
      // the key/value location in the hash corresponding to that new
      // state, to which we'll later write the state_id (idx into states_).
      Array1<uint64_t*> hash_key_value_locations(c_, tot_matched_arcs);
      uint64_t **hash_key_value_locations_data = hash_key_value_locations.Data();

      // We'll write to a_state_idx01_temp only for arcs that are
      // responsible for creating new destination-state (i.e. we'll write at
      // the same locations as hash_key_value_locations, where
      // new_state_renumbering_keep_data == 1).  It's the idx01 of the
      // dest-state, in a, of the arc.
      Array1<int32_t> a_dest_state_idx01_temp(c_, tot_matched_arcs);
      int32_t *a_dest_state_idx01_temp_data = a_dest_state_idx01_temp.Data();
      const int32_t
          *matching_a_arcs_row_splits_data = matching_a_arcs_row_splits.Data(),
          *first_matching_a_arc_idx012_data = first_matching_a_arc_idx012.Data(),
          *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data(),
          *b_fsas_row_splits2_data = b_fsas_.RowSplits(2).Data();


      StateInfo *states_data = states_.Data();
      K2_EVAL(c_, tot_matched_arcs, lambda_set_arcs_and_new_state, (int32_t idx012) -> void {
          // `idx012` is into an ragged tensor that we haven't physically
          // constructed, containing the new arcs we are adding on this frame;
          // its shape's 1st layer is formed by (num_arcs_b, b_arc_to_state),
          // and its 2nd layer is formed by (matching_a_arcs_row_splits,
          // matched_arc_to_b_arc).
          int32_t b_arc_idx01 = matched_arc_to_b_arc_data[idx012],
              matched_arc_idx01x = matching_a_arcs_row_splits_data[b_arc_idx01],
              matched_arc_idx2 = idx012 - matched_arc_idx01x,
              state_idx0 = b_arc_to_state_data[b_arc_idx01],
              b_arc_idx0x = num_arcs_b_row_splits_data[state_idx0],
              b_arc_idx1 = b_arc_idx01 - b_arc_idx0x;

          int32_t state_idx = state_begin + state_idx0; // into states_
          StateInfo sinfo = states_data[state_idx];
          int32_t b_fsas_state_idx01 = sinfo.b_fsas_state_idx01,
              b_begin_arc_idx01x = b_fsas_row_splits2_data[b_fsas_state_idx01],
              b_arc_idx012 = b_begin_arc_idx01x + b_arc_idx1;
          // ignore the apparent name mismatch setting b_arc_idx012; arc_idx1
          // is an idx1 w.r.t. a different array than b_fsas_.
          int32_t first_matching_a_arc_idx012 =
              first_matching_a_arc_idx012_data[b_arc_idx01],
              a_arc_idx012 = first_matching_a_arc_idx012 + matched_arc_idx2;

          Arc b_arc = b_arcs_data[b_arc_idx012],
              a_arc = a_arcs_data[a_arc_idx012];
          K2_CHECK_EQ(b_arc.label, a_arc.label);

          char new_dest_state = 0;
          // int32_t dest_state_idx = -1;
          if (a_arcs_data[a_arc_idx012].label != -1) {
            // investigate whether the dest-state is new (not currently
            // allocated a state-id).  We don't allocate state-ids for the
            // final-state yet, so skip this if label is -1.
            int32_t b_dest_state_idx1 = b_arc.dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 +
                   sinfo.b_fsas_state_idx01 - b_arc.src_state,
                a_dest_state_idx1 = a_arc.dest_state;
            uint64_t hash_key = (((uint64_t)a_dest_state_idx1) * a_states_multiple) +
                b_dest_state_idx01,
                hash_value = 0,  // actually it's a don't-care.
                *hash_key_value_location = nullptr;
            // If it was successfully inserted, then this arc is assigned
            // responsibility for creating the state-id for its destination
            // state.  We'll assign the value below in
            // lambda_allocate_new_state_ids.
            if (state_pair_to_state_acc.Insert(hash_key, hash_value, nullptr,
                                               &hash_key_value_location)) {
              hash_key_value_locations_data[idx012] = hash_key_value_location;
              int32_t a_dest_state_idx01 = a_dest_state_idx1 +
                  (sinfo.a_fsas_state_idx01 - a_arc.src_state);
              a_dest_state_idx01_temp_data[idx012] = a_dest_state_idx01;
              new_dest_state = (char)1;
            }
          }
          ArcInfo arc_info;
          arc_info.a_arc_idx012 = a_arc_idx012;
          arc_info.b_arc_idx012 = b_arc_idx012;
          new_arcs_data[idx012] = arc_info;
          new_arcs_row_ids_data[idx012] = state_idx;
          new_state_renumbering_keep_data[idx012] = new_dest_state;
        });

      int32_t num_new_states = new_state_renumbering.New2Old().Dim();

      const int32_t *new_state_renumbering_new2old_data =
          new_state_renumbering.New2Old().Data();
      K2_DCHECK_EQ(states_.Dim(), state_end);
      int32_t next_state_end = state_end + num_new_states;
      iter_to_state_row_splits_cpu_.push_back(next_state_end);
      states_.Resize(next_state_end);  // Note: this Resize() won't actually
                                       // reallocate each time.
      K2_CHECK_EQ(uint64_t(next_state_end) >>
                  state_pair_to_state_.NumValueBits(), 0);
      int32_t num_kept_key_bits = 64 - state_pair_to_state_.NumValueBits();

      states_data = states_.Data();  // In case it changed (unlikely)
      const int32_t *b_fsas_row_ids1_data = b_fsas_.RowIds(1).Data(),
          *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();

      // The next lambda modifies state_pair_to_state_, replacing the temporary
      // values in the hash with the newly allocated state-ids.
      K2_EVAL(c_, num_new_states, lambda_allocate_new_state_ids, (int32_t i) -> void {
          int32_t new_state_idx = state_end + i;
          // `arc_idx` below is the index into the matched arcs on this frame,
          // with 0 <= new_arc_idx < tot_matched_arcs.
          int32_t new_arc_idx = new_state_renumbering_new2old_data[i];
          uint64_t *hash_key_value_location = hash_key_value_locations_data[new_arc_idx];
          // The next assertion depends on knowledge of the implementation of
          // the hash.  If in future we change details of the hash
          // implementation and it fails, it can be removed.
          // We're checking that we inserted `hash_value = 0` above.
          K2_DCHECK_EQ(*hash_key_value_location >> num_kept_key_bits, 0);
          uint64_t key = state_pair_to_state_acc.SetValue(hash_key_value_location,
                                                          new_state_idx);
          uint32_t b_state_idx01 = key % a_states_multiple,
              a_state_idx01 = a_dest_state_idx01_temp_data[new_arc_idx];
          // a_state_idx01 is not stored in `key`, because we store it
          // as the idx1.
          StateInfo info { (int32_t)a_state_idx01, (int32_t)b_state_idx01 };
          states_data[new_state_idx] = info;
        });
  }


  ~DeviceIntersector() {
    // Prevent crash in destructor of hash (at exit, it still contains values, by design).
    state_pair_to_state_.Destroy();
  }


  ContextPtr c_;
  FsaVec a_fsas_;  // a_fsas_: decoding graphs
                   // Note: a_fsas_ has 3 axes.
  bool sorted_match_a_;  // If true, we'll require a_fsas_ to be arc-sorted; and
                         // we'll use a matching approach that won't blow up in
                         // memory or time when a_fsas_ has states with very
                         // high out-degree.

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
  // leaves from (index into states_).  Actually this may be redundant.
  Array1<int32_t> arcs_row_ids_;

  // The hash maps from state-pair, as:
  //   state_pair = (a_fsas_state_idx1 * a_states_multiple) + b_fsas_state_idx01
  // to indexes into the states_ array (numbered 0,1,2,...), or to -1
  // in cases where the state-pair is a pair of final-states.
  //
  // We name the values in the hash, which, as we mentioned, are indexes into
  // the states_ array, as`output_state_idx01`; the shape of the ragged array
  // which this is an index into, is given by
  // row_splits==iter_to_state_row_splits_cpu_.
  //
  // We ensure that a_states_multiple_ >= b_fsas_.TotSize(1) in order to ensure
  // uniqueness of the hashed values; and we also make sure a_states_multiple_
  // is odd, which ensures the states in a_fsas_ also affect the low bits of the
  // hash value.
  int32_t a_states_multiple_;

  // This hash will also contain -1 as values in cases where the dest-state is a
  // final-state (these are allocated right at the beginning); and inside of
  // Forward() and ForwardSortedA() it will also contain temporary quantities
  // for newly created states, while we are working out the newly created
  // state-ids.
  Hash state_pair_to_state_;
};



FsaVec IntersectDevice(FsaVec &a_fsas, int32_t properties_a,
                       FsaVec &b_fsas, int32_t properties_b,
                       const Array1<int32_t> &b_to_a_map,
                       Array1<int32_t> *arc_map_a,
                       Array1<int32_t> *arc_map_b,
                       bool sorted_match_a) {
  NVTX_RANGE("IntersectDevice");
  K2_CHECK_NE(properties_a & kFsaPropertiesValid, 0);
  K2_CHECK_NE(properties_b & kFsaPropertiesValid, 0);
  if (sorted_match_a && ((properties_a & kFsaPropertiesArcSorted) == 0)) {
    K2_LOG(FATAL) << "If you provide sorted_match_a=true, a_fsas "
        "must be arc-sorted, but (according to the properties) "
        "it is not.";
  }
  K2_CHECK_EQ(a_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_to_a_map.Dim(), b_fsas.Dim0());
  K2_CHECK_LT(static_cast<uint32_t>(MaxValue(b_to_a_map)),
              static_cast<uint32_t>(a_fsas.Dim0()));

  DeviceIntersector intersector(a_fsas, b_fsas, b_to_a_map,
                                sorted_match_a);
  intersector.Intersect();
  return intersector.FormatOutput(arc_map_a, arc_map_b);
}
}  // namespace k2
