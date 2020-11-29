/**
 * @brief
 * compose
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
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

namespace intersect_internal {

/* Information associated with a state active on a particular frame..  */
struct StateInfo {
  /* abs_state_id is the state-index in a_fsas_.  Note: the ind0 in here
     won't necessarily match the ind0 within FrameInfo::state if
     a_fsas_stride_ == 0. */
  int32_t a_fsas_state_idx01;

  /* Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
     FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
     represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
     use max not log-sum).  You can take the pruned lattice and rescore it if
     you want log-sum.  */
  int32_t forward_loglike;

  /* Note: this `backward_loglike` is the best score of any path from here to
     the end, minus the best path in the overall FSA, i.e. it's the backward
     score you get if, at the final-state, you set backward_loglike ==
     forward_loglike. So backward_loglike + OrderedIntToFloat(forward_loglike)
     <= 0, and you can treat it somewhat like a posterior (except they don't sum
     to one as we're using max, not log-add).
  */
  float backward_loglike;
};



/*
static std::ostream &operator<<(std::ostream &os, const StateInfo &s) {
  os << "StateInfo{" << s.a_fsas_state_idx01 << ","
     << OrderedIntToFloat(s.forward_loglike) << "," << s.backward_loglike
     << "}";
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
*/
class MultiGraphDenseIntersect {
 public:
  /**
     Intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Dim0() == 1 in which
                           case the graph is shared.
       @param [in] b_fsas  The neural-net output, with each frame containing the
                           log-likes of each phone.  A series of sequences of
                           (in general) different length.  MUST BE SORTED BY
                           DECREASING LENGTH, or it is an error.
                           (Calling code should be able to ensure this.)
       @param [in] output_beam    Beam >0 for pruning output, i.e. arcs that are
                           not on a path within `output_beam` of the best path
                           will not be retained.
   */
  MultiGraphDenseIntersect(FsaVec &a_fsas,
                           DenseFsaVec &b_fsas,
                           float output_beam)
      : a_fsas_(a_fsas),
        b_fsas_(b_fsas),
<<<<<<< HEAD
        output_beam_(output_beam) {
    NVTX_RANGE(__func__);
=======
        search_beam_(search_beam),
        output_beam_(output_beam),
        min_active_(min_active),
        max_active_(max_active),
        dynamic_beams_(a_fsas.Context(), b_fsas.shape.Dim0(), search_beam) {
    NVTX_RANGE(K2_FUNC);
>>>>>>> upstream/master
    c_ = GetContext(a_fsas.shape, b_fsas.shape);

    K2_CHECK(a_fsas_.Dim0() == b_fsas_.shape.Dim0());
    num_fsas_ = a_fsas_.Dim0();
    K2_CHECK_GT(num_fsas_, 0);
    K2_CHECK(b_fsas.scores.IsContiguous());
    K2_CHECK_GT(output_beam, 0);
    // Set up carcs_
    InitCompressedArcs();

    {
      Array1<int32_t> dest_states = GetDestStates(a_fsas_, true);
      incoming_arcs_ = GetIncomingArcs(a_fsas_, dest_states);
    }

    {
      int32_t axis = 0, num_srcs = 2;
      RaggedShape *vec1[2] = { &incoming_arcs_.shape, &a_fsas_.shape };
      forward_then_backward_shape_ = Append(axis, num_srcs, vec1);
      RaggedShape *vec2[2] = { &a_fsas_.shape, &incoming_arcs_.shape };
      backward_then_forward_shape_ = Append(axis, num_srcs, vec2);
    }

<<<<<<< HEAD
=======
  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput(). */
  void Intersect() {
    /*
      T is the largest number of (frames+1) of neural net output, or the largest
      number of frames of log-likelihoods we count the final frame with (0,
      -inf, -inf..) that is used for the final-arcc.  The largest number of
      states in the fsas represented by b_fsas equals T+1 (e.g. 1 frame would
      require 2 states, because that 1 frame is the arc from state 0 to state
      1).  So the #states is 2 greater than the actual number of frames in the
      neural-net output.
    */
    NVTX_RANGE(K2_FUNC);
    int32_t T = b_fsas_.shape.MaxSize(1), num_fsas = b_fsas_.shape.Dim0();
>>>>>>> upstream/master

    int32_t num_arcs = a_fsas_.NumElements();
    // arc_scores_ will be used for forward and backward computations
    // simultaneously.
    arc_scores_ = Array1<float>(c_, num_arcs * 2);

    // set up fsa_info_
    InitFsaInfo();

    // set up steps_, which contains a bunch of meta-information about the steps of the algorithm.
    InitSteps();

    int32_t num_seqs = b_fsas.shape.Dim0();

    { // check that b_fsas are in order of decreasing length.
      Array1<int32_t> r = b_fsas.shape.RowSplits(1).To(GetCpuContext());
      int32_t *r_data = r.Data();
      int32_t prev_t = r_data[1] - r_data[0];
      for (int32_t i = 1; i + 1 < r.Dim(); i++) {
        int32_t this_t = r_data[i+1] - r_data[i];
        if (this_t < prev_t)
          K2_LOG(FATAL) << "Sequences (DenseFsaVec) must be in sorted from greatest to least.";
        prev_t = this_t;
      }
      T_ = r_data[1] - r_data[0];  // longest first, so T_ is the length of the
                                   // longest sequence.
    }

    int32_t num_states = a_fsas_.TotSize(1);
    // this is the largest array size we'll be dealing with.
    size_t product = ((size_t)(T_+1) * (size_t)num_states);
    K2_CHECK_EQ((1+product), (size_t)(int32_t)(1+product)) <<
        "Problem size is too large for this algorithm; try reducing minibatch size.";
  }

<<<<<<< HEAD
  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput(). */
  void Intersect() {
    NVTX_RANGE(__func__);
    // TODO.
    K2_LOG(FATAL) << "Not implemented";
  }

=======
  // Return FrameInfo for 1st frame, with `states` set but `arcs` not set.
  std::unique_ptr<FrameInfo> InitialFrameInfo() {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = b_fsas_.shape.Dim0();
    std::unique_ptr<FrameInfo> ans = std::make_unique<FrameInfo>();

    if (a_fsas_.Dim0() == 1) {
      int32_t start_states_per_seq = (a_fsas_.shape.TotSize(1) > 0),  // 0 or 1
          num_start_states = num_fsas * start_states_per_seq;
      ans->states = Ragged<StateInfo>(
          RegularRaggedShape(c_, num_fsas, start_states_per_seq),
          Array1<StateInfo>(c_, num_start_states));
      StateInfo *states_data = ans->states.values.Data();
      auto lambda_set_states = [=] __host__ __device__(int32_t i) -> void {
        StateInfo info;
        info.a_fsas_state_idx01 = 0;  // start state of a_fsas_
        info.forward_loglike = FloatToOrderedInt(0.0);
        states_data[i] = info;
      };
      Eval(c_, num_start_states, lambda_set_states);
    } else {
      Ragged<int32_t> start_states = GetStartStates(a_fsas_);
      ans->states =
          Ragged<StateInfo>(start_states.shape,
                            Array1<StateInfo>(c_, start_states.NumElements()));
      StateInfo *ans_states_values_data = ans->states.values.Data();
      const int32_t *start_states_values_data = start_states.values.Data(),
                    *start_states_row_ids1_data =
                        start_states.shape.RowIds(1).Data();
      auto lambda_set_state_info =
          [=] __host__ __device__(int32_t states_idx01) -> void {
        StateInfo info;
        info.a_fsas_state_idx01 = start_states_values_data[states_idx01];
        info.forward_loglike = FloatToOrderedInt(0.0);
        ans_states_values_data[states_idx01] = info;
      };
      Eval(c_, start_states.NumElements(), lambda_set_state_info);
    }
    return ans;
  }

  void FormatOutput(FsaVec *ofsa, Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b) {
    NVTX_RANGE(K2_FUNC);
    ContextPtr c_cpu = GetCpuContext();
    int32_t T = b_fsas_.shape.MaxSize(1);

    int32_t *oshapeu_row_ids3 = oshape_unpruned_.RowIds(3).Data(),
            *oshapeu_row_ids2 = oshape_unpruned_.RowIds(2).Data(),
            *oshapeu_row_ids1 = oshape_unpruned_.RowIds(1).Data(),
            *oshapeu_row_splits3 = oshape_unpruned_.RowSplits(3).Data(),
            *oshapeu_row_splits2 = oshape_unpruned_.RowSplits(2).Data(),
            *oshapeu_row_splits1 = oshape_unpruned_.RowSplits(1).Data();

    int32_t *oshapep_row_ids3 = oshape_pruned_.RowIds(3).Data(),
            *oshapep_row_ids2 = oshape_pruned_.RowIds(2).Data(),
            *oshapep_row_ids1 = oshape_pruned_.RowIds(1).Data(),
            *oshapep_row_splits3 = oshape_pruned_.RowSplits(3).Data(),
            *oshapep_row_splits2 = oshape_pruned_.RowSplits(2).Data(),
            *oshapep_row_splits1 = oshape_pruned_.RowSplits(1).Data();

    // the 0123 and 012 express what type of indexes they are, see comment at
    // top of utils.h
    int32_t *new2old_arc_map0123 = renumber_output_arcs_.New2Old().Data(),
            *old2new_state_map012 = renumber_output_states_.Old2New().Data();

    Array1<ArcInfo *> ai_data_ptrs(c_cpu, T + 1);
    Array1<int32_t *> arcs_row_splits1_ptrs(c_cpu, T + 1);
    Array1<int32_t *> arcs_row_splits2_ptrs(c_cpu, T + 1);

    for (int32_t t = 0; t <= T; t++) {
      ai_data_ptrs.Data()[t] = frames_[t]->arcs.values.Data();
      arcs_row_splits1_ptrs.Data()[t] =
          frames_[t]->arcs.shape.RowSplits(1).Data();
      arcs_row_splits2_ptrs.Data()[t] =
          frames_[t]->arcs.shape.RowSplits(2).Data();
    }
    // transfer to GPU if we're using a GPU
    ai_data_ptrs = ai_data_ptrs.To(c_);
    arcs_row_splits1_ptrs = arcs_row_splits1_ptrs.To(c_);
    arcs_row_splits2_ptrs = arcs_row_splits2_ptrs.To(c_);
    ArcInfo **ai_data_ptrs_data = ai_data_ptrs.Data();
    int32_t **arcs_row_splits1_ptrs_data = arcs_row_splits1_ptrs.Data(),
            **arcs_row_splits2_ptrs_data = arcs_row_splits2_ptrs.Data();

    int32_t tot_arcs_pruned = oshape_pruned_.TotSize(3);
    *arc_map_a = Array1<int32_t>(c_, tot_arcs_pruned);
    *arc_map_b = Array1<int32_t>(c_, tot_arcs_pruned);
    int32_t *arc_map_a_data = arc_map_a->Data(),
            *arc_map_b_data = arc_map_b->Data();
    Array1<Arc> arcs_out(c_, tot_arcs_pruned);
    Arc *arcs_out_data = arcs_out.Data();
    const Arc *a_fsas_arcs = a_fsas_.values.Data();
    int32_t b_fsas_num_cols = b_fsas_.scores.Dim1();
    const int32_t *b_fsas_row_ids1 = b_fsas_.shape.RowIds(1).Data();
    const int32_t *b_fsas_row_splits1 = b_fsas_.shape.RowSplits(1).Data();

    auto lambda_format_arc_data =
        [=] __host__ __device__(int32_t pruned_idx0123) -> void {
      int32_t unpruned_idx0123 = new2old_arc_map0123[pruned_idx0123];
      int32_t unpruned_idx012 = oshapeu_row_ids3[unpruned_idx0123],
              unpruned_idx01 = oshapeu_row_ids2[unpruned_idx012],
              unpruned_idx01x = oshapeu_row_splits2[unpruned_idx01],
              unpruned_idx01xx = oshapeu_row_splits3[unpruned_idx01x],
              unpruned_idx23 = unpruned_idx0123 - unpruned_idx01xx,
              unpruned_idx0 = oshapeu_row_ids1[unpruned_idx01],  // fsa-id
          unpruned_idx0x = oshapeu_row_splits1[unpruned_idx0],
              // unpruned_idx0xx = oshapeu_row_splits2[unpruned_idx0x],
          unpruned_idx1 = unpruned_idx01 - unpruned_idx0x,  // t
          unpruned_idx01_next_t = unpruned_idx01 + 1,
              unpruned_idx01x_next_t =
                  oshapeu_row_splits2[unpruned_idx01_next_t];

      int32_t t = unpruned_idx1;
      int32_t *arcs_row_splits1_data = arcs_row_splits1_ptrs_data[t],
              *arcs_row_splits2_data = arcs_row_splits2_ptrs_data[t],
              arcs_idx0x = arcs_row_splits1_data[unpruned_idx0],
              arcs_idx0xx = arcs_row_splits2_data[arcs_idx0x];
      // below: axes 2,3 of the unpruned layout coincide with axes 1,2 of
      // 'arcs'; these are state and arc indexes (within this frame
      // of this FSA).
      int32_t arcs_idx012 = arcs_idx0xx + unpruned_idx23;
      ArcInfo *ai_data = ai_data_ptrs_data[t];
      ArcInfo arc_info = ai_data[arcs_idx012];

      // we call it ind2 because the state-index is axis 2 of oshape.
      int32_t unpruned_dest_state_idx2 = arc_info.u.dest_info_state_idx1,
              unpruned_dest_state_idx012 =
                  unpruned_idx01x_next_t + unpruned_dest_state_idx2,
              pruned_dest_state_idx012 =
                  old2new_state_map012[unpruned_dest_state_idx012],
              pruned_dest_state_idx01 =
                  oshapep_row_ids2[pruned_dest_state_idx012],
              pruned_dest_state_idx0 =
                  oshapep_row_ids1[pruned_dest_state_idx01],
              pruned_dest_state_idx0x =
                  oshapep_row_splits1[pruned_dest_state_idx0],
              pruned_dest_state_idx0xx =
                  oshapep_row_splits2[pruned_dest_state_idx0x],
              pruned_dest_state_idx12 =
                  pruned_dest_state_idx012 - pruned_dest_state_idx0xx;

      // note: the src-state and dest-state have the same ind0 which is the
      // FSA-id.
      int32_t pruned_src_state_idx012 = old2new_state_map012[unpruned_idx012],
              pruned_src_state_idx12 =
                  pruned_src_state_idx012 - pruned_dest_state_idx0xx;

      Arc arc;
      // The numbering for the dest-state in the output Arc is the numbering
      // *within the FSA*, and we ignore the time index (1) because that index
      // will be removed as the FSA format has no notion of time; that's why we
      // use the indx12.

      arc_map_a_data[pruned_idx0123] = arc_info.a_fsas_arc_idx012;

      arc.src_state = pruned_src_state_idx12;
      arc.dest_state = pruned_dest_state_idx12;
      arc.label = a_fsas_arcs[arc_info.a_fsas_arc_idx012].label;
      K2_CHECK_LE(static_cast<uint32_t>(arc.label + 1),
                  static_cast<uint32_t>(b_fsas_num_cols))
          << "label out of range";
      int32_t fsa_id = unpruned_idx0, b_fsas_idx0x = b_fsas_row_splits1[fsa_id],
              b_fsas_idx01 = b_fsas_idx0x + t, b_fsas_idx2 = (arc.label + 1),
              b_fsas_arc_idx012 = b_fsas_idx01 * b_fsas_num_cols + b_fsas_idx2;
      arc.score = arc_info.arc_loglike;
      arc_map_b_data[pruned_idx0123] = b_fsas_arc_idx012;
      arcs_out_data[pruned_idx0123] = arc;
    };
    Eval(c_, tot_arcs_pruned, lambda_format_arc_data);
>>>>>>> upstream/master

  void InitCompressedArcs() {
    K2_LOG(FATAL) << "Not implemented";
  }

<<<<<<< HEAD
  void InitFsaInfo() {
    K2_LOG(FATAL) << "Not implemented";
=======
  /*
    Computes pruning cutoffs for this frame: these are the cutoffs for the arc
    "forward score", one per FSA.  This is a dynamic process involving
    dynamic_beams_ which are updated on each frame (they start off at
    search_beam_).

       @param [in] arc_end_scores  The "forward log-probs" (scores) at the
                    end of each arc, i.e. its contribution to the following
                    state.  Is a tensor indexed [fsa_id][state][arc]; we
                    will get rid of the [state] dim, combining it with the
                    [arc] dim, so it's just [fsa_id][arc]
                    It is conceptually unchanged by this operation but non-const
                    because row-ids of its shape may need to be generated.
       @return      Returns a vector of log-likelihood cutoffs, one per FSA (the
                    cutoff will be -infinity for FSAs that don't have any active
                    states).  The cutoffs will be of the form: the best score
                    for any arc, minus the dynamic beam.  See the code for how
                    the dynamic beam is adjusted; it will approach
    'search_beam_' as long as the number of active states in each FSA is between
    min_active and max_active.
  */
  Array1<float> GetPruningCutoffs(Ragged<float> &arc_end_scores) {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = arc_end_scores.shape.Dim0();

    // get the maximum score from each sub-list (i.e. each FSA, on this frame).
    // Note: can probably do this with a cub Reduce operation using an operator
    // that has side effects (that notices when it's operating across a
    // boundary).
    // the max will be -infinity for any FSA-id that doesn't have any active
    // states (e.g. because that stream has finished).
    // Casting to ragged2 just considers the top 2 indexes, ignoring the 3rd.
    // i.e. it's indexed by [fsa_id][state].
    Ragged<float> end_scores_per_fsa = arc_end_scores.RemoveAxis(1);
    Array1<float> max_per_fsa(c_, end_scores_per_fsa.Dim0());
    MaxPerSublist(end_scores_per_fsa, -std::numeric_limits<float>::infinity(),
                  &max_per_fsa);
    const int32_t *arc_end_scores_row_splits1_data =
        arc_end_scores.RowSplits(1).Data();
    const float *max_per_fsa_data = max_per_fsa.Data();
    float *dynamic_beams_data = dynamic_beams_.Data();

    float default_beam = search_beam_, max_active = max_active_,
          min_active = min_active_;
    K2_CHECK_LT(min_active, max_active);

    Array1<float> cutoffs(c_, num_fsas);
    float *cutoffs_data = cutoffs.Data();

    auto lambda_set_beam_and_cutoffs =
        [=] __host__ __device__(int32_t i) -> void {
      float best_loglike = max_per_fsa_data[i],
            dynamic_beam = dynamic_beams_data[i];
      int32_t active_states = arc_end_scores_row_splits1_data[i + 1] -
                              arc_end_scores_row_splits1_data[i];
      if (active_states <= max_active) {
        // Not constrained by max_active...
        if (active_states >= min_active || active_states == 0) {
          // Neither the max_active nor min_active constraints
          // apply.  Gradually approach 'beam'
          // (Also approach 'beam' if active_states == 0; we might as
          // well, since there is nothing to prune here).
          dynamic_beam = 0.8 * dynamic_beam + 0.2 * default_beam;
        } else {
          // We violated the min_active constraint -> increase beam
          if (dynamic_beam < default_beam) dynamic_beam = default_beam;
          // gradually make the beam larger as long
          // as we are below min_active
          dynamic_beam *= 1.25;
        }
      } else {
        // We violated the max_active constraint -> decrease beam
        if (dynamic_beam > default_beam) dynamic_beam = default_beam;
        // Decrease the beam as long as we have more than
        // max_active active states.
        dynamic_beam *= 0.9;
      }
      dynamic_beams_data[i] = dynamic_beam;
      cutoffs_data[i] = best_loglike - dynamic_beam;
    };
    Eval(c_, num_fsas, lambda_set_beam_and_cutoffs);

    return cutoffs;
>>>>>>> upstream/master
  }

  /*
    InitSteps() sets up steps_; it works out metadata and allocates memory, but
    does not do any of the actual computation.
   */
<<<<<<< HEAD
  void InitSteps() {
    NVTX_RANGE(__func__);
    // This vector, of length num_fsas_, tells us how many copies of (the states
    // of the i'th decoding graph) we have.  It equals (the length of the sequence
    // of log-likes in n_fsas_) + 1.  It is monotonically decreasing (thanks to
    // how we require the FSAs to be sorted).
    Array1<int32_t> num_copies_per_fsa(c_, num_fsas_);
    const int32_t *b_row_splits_data = b_fsas_.shape.RowSplits(1).Data();
    int32_t *num_copies_per_fsa_data = num_copies_per_fsa.Data();
    auto lambda_set_num_copies = [=]  __host__ __device__ (int32_t i) -> void {
      num_copies_per_fsa_data[i] = 1 + b_row_splits_data[i + 1] - b_row_splits_data[i];
=======
  Ragged<ArcInfo> GetArcs(int32_t t, FrameInfo *cur_frame) {
    NVTX_RANGE(K2_FUNC);
    Ragged<StateInfo> &states = cur_frame->states;
    const StateInfo *state_values = states.values.Data();

    // in a_fsas_ (the decoding graphs), maps from state_idx01 to arc_idx01x.
    const int32_t *fsa_arc_splits = a_fsas_.shape.RowSplits(2).Data();

    int32_t num_states = states.values.Dim();
    Array1<int32_t> num_arcs(c_, num_states + 1);
    int32_t *num_arcs_data = num_arcs.Data();
    auto num_arcs_lambda =
        [=] __host__ __device__(int32_t state_idx01) -> void {
      int32_t a_fsas_state_idx01 = state_values[state_idx01].a_fsas_state_idx01,
              a_fsas_arc_idx01x = fsa_arc_splits[a_fsas_state_idx01],
              a_fsas_arc_idx01x_next = fsa_arc_splits[a_fsas_state_idx01 + 1],
              a_fsas_num_arcs = a_fsas_arc_idx01x_next - a_fsas_arc_idx01x;
      num_arcs_data[state_idx01] = a_fsas_num_arcs;
>>>>>>> upstream/master
    };
    Eval(c_, num_fsas_, lambda_set_num_copies);

<<<<<<< HEAD
    std::vector<int32_t> range(num_fsas_ + 1);
    // fill with num_fsas_, num_fsas_ + 1, num_fsas_ + 2, ... num_fsas_ * 2.
    std::iota(range.begin(), range.end(), num_fsas_);
    std::vector<RaggedShape> bf_shape_prefixes = GetPrefixes(backward_then_forward_shape_,
                                                             range),
                             fb_shape_prefixes = GetPrefixes(forward_then_backward_shape_,
                                                             range);


    ContextPtr c_cpu = GetCpuContext();
=======
  /*
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* object for the next frame.

      @param [in] t   Time-step that we are processing arcs leaving from;
                   will be called with t=0, t=1, ...
      @param [in] cur_frame  FrameInfo object for the states corresponding to
                   time t; will have its 'states' member set up but not its
                   'arcs' member (this function will create that).
     @return  Returns FrameInfo object corresponding to time t+1; will have its
             'states' member set up but not its 'arcs' member.
   */
  std::unique_ptr<FrameInfo> PropagateForward(int32_t t, FrameInfo *cur_frame) {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = NumFsas();
    // Ragged<StateInfo> &states = cur_frame->states;
    // arc_info has 3 axes: fsa_id, state, arc.
    Ragged<ArcInfo> arc_info = GetArcs(t, cur_frame);

    ArcInfo *ai_data = arc_info.values.Data();
    Array1<float> ai_data_array1(c_, arc_info.values.Dim());
    float *ai_data_array1_data = ai_data_array1.Data();
    auto lambda_set_ai_data = [=] __host__ __device__(int32_t i) -> void {
      ai_data_array1_data[i] = ai_data[i].end_loglike;
    };
    Eval(c_, ai_data_array1.Dim(), lambda_set_ai_data);
    Ragged<float> ai_loglikes(arc_info.shape, ai_data_array1);

    // `cutoffs` is of dimension num_fsas.
    Array1<float> cutoffs = GetPruningCutoffs(ai_loglikes);
    float *cutoffs_data = cutoffs.Data();

    // write certain indexes ( into ai.values) to state_map_.Data().  Keeps
    // track of the active states and will allow us to assign a numbering to
    // them.
    int32_t *ai_row_ids1 = arc_info.shape.RowIds(1).Data(),
            *ai_row_ids2 = arc_info.shape.RowIds(2).Data();
    auto state_map_acc = state_map_.Accessor();
    // We use a separate state_map vector per FSA we're processing if a_fsas_
    // only has one FSA (in this case it means we're sharing the FSA among
    // potentially multiple streams).
    // But if there is >1 FSA in a_fsas_, then a_fsas_ is indexed by the idx01
    // into a_fsas_ and we don't need a stride.
    // Modifying the accessor is perhaps a little bit of a hack.
    if (a_fsas_.shape.Dim0() > 1) state_map_acc.elem_stride0 = 0;

    {
      NVTX_RANGE("LambdaSetStateMap");
      auto lambda_set_state_map =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
        int32_t dest_state_idx01 =
            ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
        float end_loglike = ai_data[arc_idx012].end_loglike,
              cutoff = cutoffs_data[fsa_id];
        if (end_loglike > cutoff) {
          // The following is a race condition as multiple threads may write to
          // the same location, but it doesn't matter, the point is to assign
          // one of the indexes.
          state_map_acc(fsa_id, dest_state_idx01) = arc_idx012;
        }
      };
      Eval(c_, arc_info.NumElements(), lambda_set_state_map);
    }

    // renumber_states will be a renumbering that dictates which of the arcs in
    // 'ai' correspond to unique states.  Only one arc for each dest-state is
    // kept (it doesn't matter which one).
    Renumbering renumber_states(c_, arc_info.NumElements());

    renumber_states.Keep() = 0;
    char *keep_this_state_data = renumber_states.Keep().Data();

    if (a_fsas_.shape.Dim0() > 1) {
      NVTX_RANGE("LambdaSetKeepA");
      int32_t *state_map_data = state_map_.Data();
      auto lambda_set_keep =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t dest_state = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
        int32_t j = state_map_data[dest_state];
        if (j != -1) {  // the dest-state was kept..
          // caution: keep_this_state_data is indexed by *arc*
          if (j == arc_idx012)  // this arc 'won' the data race..
            keep_this_state_data[arc_idx012] = 1;
        }
      };
      Eval(c_, arc_info.NumElements(), lambda_set_keep);
    } else {
      NVTX_RANGE("LambdaSetKeepB");
      auto lambda_set_keep =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                dest_state = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
        int j = state_map_acc(fsa_id, dest_state);
        if (j != -1) {  // the dest-state was kept..
          // caution: keep_this_state_data is indexed by *arc*
          if (j == arc_idx012)  // this arc 'won' the data race..
            keep_this_state_data[arc_idx012] = 1;
        }
      };
      Eval(c_, arc_info.NumElements(), lambda_set_keep);
    }

    int32_t num_states = renumber_states.NumNewElems();
    int32_t *state_reorder_data = renumber_states.Old2New().Data();

    // state_to_fsa_id maps from an index into the next frame's
    // FrameInfo::states.values() vector to the sequence-id (fsa_id) associated
    // with it.  It should be non-decreasing.
    Array1<int32_t> state_to_fsa_id(c_, num_states);
    {  // This block sets 'state_to_fsa_id'.
      NVTX_RANGE("LambdaSetStateToFsaId");
      int32_t *state_to_fsa_id_data = state_to_fsa_id.Data();
      auto lambda_state_to_fsa_id =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                this_state_j = state_reorder_data[arc_idx012],
                next_state_j = state_reorder_data[arc_idx012 + 1];
        if (next_state_j > this_state_j) {
          state_to_fsa_id_data[this_state_j] = fsa_id;
        }
      };
      Eval(c_, arc_info.NumElements(), lambda_state_to_fsa_id);
>>>>>>> upstream/master

    // This vector, of length T_ + 1, tells us, for each frame 0 <= t <= T, how
    // many FSAs have a copy of their decoding-graph states alive on this
    // time-index.  It equals InvertMonotonicDecreasing(num_copies_per_fsa_)
    // and it is also the case that InvertMonotonicDecreasing(num_fsas_per_t_)
    // == num_copies_per_fsa_.
    Array1<int32_t> num_fsas_per_t = InvertMonotonicDecreasing(
        num_copies_per_fsa),
                num_fsas_per_t_cpu = num_fsas_per_t.To(c_cpu);

    Array1<int32_t> a_fsas_row_splits1_cpu = a_fsas_.RowSplits(1).To(c_cpu),
                  a_fsas_row_splits12_cpu = a_fsas_.RowSplits(2)[
                      a_fsas_.RowSplits(1)].To(c_cpu);
    int32_t tot_arcs = a_fsas_.NumElements(),
          tot_states = a_fsas_.TotSize(1);

    std::vector<Step> steps(T_ + 1);

    for (int32_t t = 0; t <= T_; t++) {
      Step &step = steps[t];
      step.forward_t = t;
      step.backward_t = T_ - t;
      step.forward_before_backward = (step.forward_t <= step.backward_t);
      int32_t nf = step.forward_num_fsas = num_fsas_per_t_cpu[step.forward_t],
              nb = step.backward_num_fsas = num_fsas_per_t_cpu[step.backward_t];


      int32_t num_arcs_forward = a_fsas_row_splits12_cpu[nf],
             num_arcs_backward = a_fsas_row_splits12_cpu[nb],
            num_states_forward = a_fsas_row_splits1_cpu[nf],
           num_states_backward = a_fsas_row_splits1_cpu[nb];

      if (step.forward_before_backward) {
        // fb_shape_prefixes[nb] is incoming_arcs_.shape appended with the first
        // `nb` FSAs of a_fsas_.shape.  Note: for purposes of allocation (and
        // reduction of arcs->states) we assume that for the forward pass all
        // the FSAs are active; this may not really be true, but it keeps the
        // shapes regular.
        step.arc_scores = Ragged<float>(fb_shape_prefixes[nb],
                                        arc_scores_.Arange(0, tot_arcs +
                                                           num_arcs_backward));
        step.forward_arc_scores = arc_scores_.Arange(0, tot_arcs);
        step.backward_arc_scores = arc_scores_.Arange(tot_arcs,
                                                      tot_arcs + num_arcs_backward);
        step.state_scores = Array1<float>(c_, tot_states + num_states_backward);
        step.forward_state_scores = step.state_scores.Arange(0, tot_states);
        step.backward_state_scores = step.state_scores.Arange(0, num_states_backward);
      } else {
        step.arc_scores = Ragged<float>(bf_shape_prefixes[nb],
                                        arc_scores_.Arange(0, tot_arcs +
                                                           num_arcs_forward));
        step.backward_arc_scores = arc_scores_.Arange(0, tot_arcs);
        step.forward_arc_scores = arc_scores_.Arange(tot_arcs,
                                                     tot_arcs + num_arcs_forward);
        step.state_scores = Array1<float>(c_, tot_states + num_states_forward);
        step.backward_state_scores = step.state_scores.Arange(0, tot_states);
        step.forward_state_scores = step.state_scores.Arange(0, num_states_forward);
      }
    }
  }

<<<<<<< HEAD
=======
    {
      NVTX_RANGE("LambdaModifyStateMap");
      // Modify the elements of `state_map` to refer to the indexes into
      // `ans->states` / `kept_states_data`, rather than the indexes into
      // ai_data. This will decrease some of the values in `state_map`, in
      // general.
      auto lambda_modify_state_map =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
        int32_t dest_state_idx = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
        int32_t this_j = state_reorder_data[arc_idx012],
                next_j = state_reorder_data[arc_idx012 + 1];
        if (next_j > this_j) {
          K2_CHECK_EQ(state_map_acc(fsa_id, dest_state_idx), arc_idx012);
          // Note: this_j is an idx01 into ans->states.  previously
          // state_map_data cotained an arc_idx012 (of the entering arc that won
          // the race).
          state_map_acc(fsa_id, dest_state_idx) = this_j;
        }
      };
      Eval(c_, arc_info.NumElements(), lambda_modify_state_map);
    }
>>>>>>> upstream/master


<<<<<<< HEAD
  void DoStep0() {
    NVTX_RANGE(__func__);
    // Run step zero of the computation: this initializes the forward probabilities on
    // frame 0, and the backward probabilities on the last frame for each sequence.
    std::vector<float*> backward_state_scores_vec(T_ + 1);
    int32_t tot_states = a_fsas_.TotSize(1);
    for (int32_t t = 0; t <= T_; t++) {
      int32_t bt = steps_[t].backward_t;
      backward_state_scores_vec[bt] = steps_[t].backward_state_scores.Data();
=======
    {
      NVTX_RANGE("LambdaSetArcsAndStates");
      auto lambda_set_arcs_and_states =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];

        ArcInfo &info = ai_data[arc_idx012];

        int32_t dest_a_fsas_state_idx01 = info.u.dest_a_fsas_state_idx01;
        // state_idx01 is the index into ans->states, of the destination state.
        // Note: multiple arcs may enter this state, which is why we had to set
        // that in a separate kernel (lambda_modify_state_map).
        int32_t state_idx01 = state_map_acc(fsa_id, dest_a_fsas_state_idx01);
        info.u.dest_info_state_idx01 = state_idx01;

        // state_idx01 will be -1 for states that were pruned out because
        // they were outside the beam.
        if (state_idx01 < 0) return;
        // multiple threads may write the same value to the address written to
        // in the next line.
        kept_states_data[state_idx01].a_fsas_state_idx01 =
            dest_a_fsas_state_idx01;
        int32_t end_loglike_int = FloatToOrderedInt(info.end_loglike);
        // Set the forward log-like of the dest state to the largest of any of
        // those of the incoming arcs.  Note: we initialized this in
        // lambda_init_loglike above.
        atomicMax(&(kept_states_data[state_idx01].forward_loglike),
                  end_loglike_int);
      };
      Eval(c_, arc_info.NumElements(), lambda_set_arcs_and_states);
    }
    {
      NVTX_RANGE("LambdaResetStateMap");
      const int32_t *next_states_row_ids1 = ans->states.shape.RowIds(1).Data();
      auto lambda_reset_state_map =
          [=] __host__ __device__(int32_t state_idx01) -> void {
        int32_t a_fsas_state_idx01 =
                    kept_states_data[state_idx01].a_fsas_state_idx01,
                fsa_idx0 = next_states_row_ids1[state_idx01];
        K2_CHECK_EQ(state_map_acc(fsa_idx0, a_fsas_state_idx01), state_idx01);
        // We're resetting state_map to its original clean condition.
        state_map_acc(fsa_idx0, a_fsas_state_idx01) = -1;
      };
      Eval(c_, ans->states.NumElements(), lambda_reset_state_map);
>>>>>>> upstream/master
    }
    backward_state_scores_ = Array1<float*>(c_, backward_state_scores_vec);
    float **backward_state_scores_data = backward_state_scores_.Data();
    float *forward_scores_t0 = steps_[0].forward_state_scores.Data();
    int32_t *a_fsas_row_ids1 = a_fsas_.RowIds(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    const float minus_inf = -std::numeric_limits<float>::infinity();
    auto lambda_init_state_scores = [=] __host__ __device__ (int32_t state_idx01) -> void {
      int32_t fsa_idx0 = a_fsas_row_ids1[state_idx01];
      FsaInfo this_info = fsa_info_data[fsa_idx0];
      int32_t state_idx0x = this_info.state_offset,
          state_idx1 = state_idx01 - state_idx0x;
      float start_loglike = (state_idx1 == 0 ? 0 : minus_inf),
         end_loglike = (state_idx1 + 1 == this_info.num_states ? 0 :
                      minus_inf);
      float *backward_state_scores_last_frame = backward_state_scores_data[this_info.T];
      forward_scores_t0[state_idx01] = start_loglike;
      backward_state_scores_last_frame[state_idx01] = end_loglike;
    };
    Eval(c_, tot_states, lambda_init_state_scores);
  }

  /* Called for 1 <= t <= T_, does one step of propagation (does forward and
     backward simultaneously, for different time steps) */
  void DoStep(int32_t t) {
    NVTX_RANGE(__func__);
    Step &step = steps_[t], &prev_step = steps_[t-1];

    int32_t forward_num_fsas = step.forward_num_fsas,
           backward_num_fsas = step.backward_num_fsas;
    float *forward_arc_scores_data = step.forward_arc_scores.Data(),
         *backward_arc_scores_data = step.backward_arc_scores.Data(),
     *prev_forward_state_scores_data = prev_step.forward_state_scores.Data(),
  *next_backward_state_scores_data = prev_step.backward_state_scores.Data();

    // the frame from which we need to read scores is actually forward_t - 1,
    // e.g. if we are writing the state-probs on frame t=1 we need to read the
    // scores on t=1.
    int32_t forward_scores_t = step.forward_t - 1,
           backward_state_scores_t = step.backward_t;


    CompressedArc *carcs_data = carcs_.Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    float *scores_data = b_fsas_.scores.Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();

    auto lambda_set_arc_scores = [=] __host__ __device__ (int32_t arc_idx012) -> void {
      CompressedArc carc = carcs_data[arc_idx012];
      int32_t fsa_idx = carc.fsa_idx;
      FsaInfo fsa_info = fsa_info_data[fsa_idx];
      // First, forward pass.  We read from the src_state of the arc
      if (fsa_idx < forward_num_fsas) {
        int32_t src_state_idx1 = carc.src_state,
               src_state_idx01 = fsa_info.state_offset + src_state_idx1;
        float src_prob = prev_forward_state_scores_data[src_state_idx01];
        float arc_end_prob = src_prob + carc.score +
                             scores_data[carc.label_plus_one +
                                         fsa_info.scores_offset +
                                         scores_stride * forward_scores_t];
        // For the forward pass we write the arc-end probs out-of-order
        // w.r.t. the regular ordering of the arcs, so we can more easily
        // do the reduction at the destination states.
        forward_arc_scores_data[carc.incoming_arc_idx012] = arc_end_prob;
      }
      if (fsa_idx < backward_num_fsas) {
        int32_t dest_state_idx1 = carc.dest_state,
               dest_state_idx01 = fsa_info.state_offset + dest_state_idx1;
        float dest_prob = next_backward_state_scores_data[dest_state_idx01];
        float arc_begin_prob = dest_prob + carc.score +
                             scores_data[carc.label_plus_one +
                                         fsa_info.scores_offset +
                                         scores_stride * backward_state_scores_t];
        backward_arc_scores_data[arc_idx012] = arc_begin_prob;
      }
    };
    Eval(c_, step.arc_scores.NumElements(), lambda_set_arc_scores);
    MaxPerSublist(step.arc_scores, -std::numeric_limits<float>::infinity(),
                  &step.state_scores);
  }

  /*
    Called after DoStep() is done for all time steps, returns the total scores
    minus output_beam_.  (This is what it does in the absence of roundoff error
    making the forward and backward tot_scores differ; when they do, it tries to
    pick a looser beam if there is significant roundoff).
   */
<<<<<<< HEAD
  Array1<float> GetScoreCutoffs() {
    std::vector<float*> forward_state_scores_vec(T_);
    int32_t tot_states = a_fsas_.TotSize(1);
    for (int32_t t = 0; t <= T_; t++) {
      int32_t ft = steps_[t].forward_t;  // actually == t, but it's clearer.
      forward_state_scores_vec[ft] = steps_[t].forward_state_scores.Data();
    }
    forward_state_scores_ = Array1<float*>(c_, forward_state_scores_vec);
    float **forward_state_scores_data = forward_state_scores_.Data();

    float *backward_state_scores_t0 = steps_[0].backward_state_scores.Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    Array1<float> score_cutoffs(c_, num_fsas_);
    float *score_cutoffs_data = score_cutoffs.Data();
=======
  void PropagateBackward(int32_t t, FrameInfo *cur_frame,
                         FrameInfo *next_frame) {
    NVTX_RANGE(K2_FUNC);
    int32_t num_states = cur_frame->states.NumElements(),
            num_arcs = cur_frame->arcs.NumElements();
    Ragged<StateInfo> &cur_states = cur_frame->states;  // 2 axes: fsa,state
    StateInfo *cur_states_data = cur_states.values.Data();

    int32_t *a_fsas_row_ids1 = a_fsas_.shape.RowIds(1).Data(),
            *a_fsas_row_splits1 = a_fsas_.shape.RowSplits(1).Data();

    float minus_inf = -std::numeric_limits<float>::infinity();

    /* arc_backward_probs represents the backward-prob at the beginning of the
       arc.  Indexing is [state_idx01][arc_idx2], (where state_idx01 and
       arc_idx2 are named w.r.t. frames_[t]->arcs. */
    RaggedShape sub_curr_frame_shape = RemoveAxis(cur_frame->arcs.shape, 0);
    Array1<float> sub_curr_frame_values(c_, num_arcs);
    Ragged<float> arc_backward_prob(sub_curr_frame_shape,
                                    sub_curr_frame_values);
    float *arc_backward_prob_data = arc_backward_prob.values.Data();

    ArcInfo *ai_data = cur_frame->arcs.values.Data();
    int32_t *arcs_rowids1 = cur_frame->arcs.shape.RowIds(1).Data(),
            *arcs_rowids2 = cur_frame->arcs.shape.RowIds(2).Data(),
            *arcs_row_splits1 = cur_frame->arcs.shape.RowSplits(1).Data(),
            *arcs_row_splits2 = cur_frame->arcs.shape.RowSplits(2).Data();
>>>>>>> upstream/master
    float output_beam = output_beam_;
    auto lambda_set_cutoffs = [=] __host__ __device__ (int32_t fsa_idx0) -> void {
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float tot_score_start = backward_state_scores_t0[0],
        tot_score_end = forward_state_scores_data[fsa_info.T]
                 [fsa_info.state_offset + fsa_info.num_states - 1],
        tot_score_avg = 0.5 * (tot_score_start + tot_score_end),
        tot_score_diff = fabs(tot_score_end - tot_score_start);
      // TODO(dan): remove the following after the code is tested.
      K2_CHECK(tot_score_diff < 0.1);
      // subtracting the difference in scores is to help make sure we don't completely prune
      // away all states.
      score_cutoffs_data[fsa_idx0] = tot_score_avg - tot_score_diff - output_beam;
    };
    Eval(c_, num_fsas_, lambda_set_cutoffs);
    K2_LOG(INFO) << "Cutoffs = " << score_cutoffs;
    return score_cutoffs;
  }

<<<<<<< HEAD
=======
    int32_t *oshape_row_splits1 = oshape_unpruned_.RowSplits(1).Data(),
            *oshape_row_splits2 = oshape_unpruned_.RowSplits(2).Data(),
            *oshape_row_splits3 = oshape_unpruned_.RowSplits(3).Data();

    // these have the "output" formatting where we number things with
    // oshape_unpruned_, which is indexed [fsa][t][state][arc].
    char *keep_arcs_data = renumber_output_arcs_.Keep().Data(),
         *keep_states_data = renumber_output_states_.Keep().Data();

    if (next_frame != NULL) {
      // compute arc backward probs, and set elements of 'keep_arcs'

      StateInfo *cur_states_data = cur_frame->states.values.Data();

      // arc_row_ids maps from arc-idx to frame-state-idx, i.e. idx012 into
      // `arcs` to idx01 into `arcs`.

      // next_states_row_splits1 maps from fsa_idx0 to state_idx01
      int32_t *next_states_row_splits1 =
          next_frame->states.shape.RowSplits(1).Data();

      StateInfo *next_states_data = next_frame->states.values.Data();
      auto lambda_set_arc_backward_prob_and_keep =
          [=] __host__ __device__(int32_t arcs_idx012) -> void {
        ArcInfo *arc = ai_data + arcs_idx012;
        int32_t state_idx01 = arcs_rowids2[arcs_idx012],
                fsa_idx0 = arcs_rowids1[state_idx01],
                fsa_idx0x = arcs_row_splits1[fsa_idx0],
                fsa_idx0xx = arcs_row_splits2[fsa_idx0x],
                arcs_idx12 = arcs_idx012 - fsa_idx0xx;

        int32_t dest_state_idx01 = arc->u.dest_info_state_idx01;
        char keep_this_arc = 0;
        float backward_loglike = minus_inf;
        if (dest_state_idx01 >= 0) {  // Dest-state was not pruned out..
          int32_t next_state_idx0x = next_states_row_splits1[fsa_idx0],
                  dest_state_idx1 = dest_state_idx01 - next_state_idx0x;
          arc->u.dest_info_state_idx1 = dest_state_idx1;
          float arc_loglike = arc->arc_loglike;
          float dest_state_backward_loglike =
              next_states_data[dest_state_idx01].backward_loglike;
          // 'backward_loglike' is the loglike at the beginning of the arc
          backward_loglike = arc_loglike + dest_state_backward_loglike;
          float src_state_forward_loglike = OrderedIntToFloat(
              cur_states_data[arcs_rowids2[arcs_idx012]].forward_loglike);
          keep_this_arc =
              (backward_loglike + src_state_forward_loglike >= -output_beam);
        }
        int32_t oshape_arc_idx0x = oshape_row_splits1[fsa_idx0],
                oshape_arc_idx01 = oshape_arc_idx0x + t,
                oshape_arc_idx01x = oshape_row_splits2[oshape_arc_idx01],
                oshape_arc_idx01xx = oshape_row_splits3[oshape_arc_idx01x],
                oshape_arc_idx0123 = oshape_arc_idx01xx + arcs_idx12;
        // note, for the previous line: indexes 1 and 2 of FrameInfo::arcs
        // (==state,arc) become indexes 2 and 3 of oshape_unpruned_.
        keep_arcs_data[oshape_arc_idx0123] = keep_this_arc;
        arc_backward_prob_data[arcs_idx012] = backward_loglike;
      };
      Eval(c_, arc_backward_prob.NumElements(),
           lambda_set_arc_backward_prob_and_keep);
    } else {
      assert(arc_backward_prob.NumElements() == 0 &&
             "Caution: final frame has arcs; check that there were -infinities "
             "in the right place on the last frame of the 'scores' matrix.");
    }
>>>>>>> upstream/master

  /*
    Does pruning and returns a ragged array indexed [fsa][state][arc], containing
    the result of intersection.

         @param [out] arc_map_a_out  If non-NULL, the map from (arc-index of returned
                                  FsaVec) to (arc-index in a_fsas_) will be written
                                  to here.
         @param [out] arc_map_b_out  If non-NULL, the map from (arc-index of returned
                                     FsaVec) to (offset into b_fsas_.scores.Data())
                                     will be written to here.
         @return  Returns a FsaVec that is the composed result.  Note: due to roundoff,
                      it may possibly contain states and/or arcs that are not
                      accessible or not co-accessible.  It will be top-sorted,
                      and deterministic and arc-sorted if the input a_fsas_
                      had those properties.
   */
  FsaVec FormatOutput(Array1<int32_t> *arc_map_a_out,
                      Array1<int32_t> *arc_map_b_out) {
    NVTX_RANGE(__func__);
    Array1<float> score_cutoffs = GetScoreCutoffs();
    float *score_cutoffs_data = score_cutoffs.Data();
    int32_t num_states = a_fsas_.TotSize(1);
    int32_t product = ((size_t)(T_+1) * (size_t)num_states);

    // We'll do exclusive-sum on the following array, after setting its elements
    // to 1 if the corresponding state was not pruned away.  The order of
    // 'counts' is: (T+1) copies of all the states of fsa index 0, (T+1) copies
    // of all the states of FSA index 1, and so on.  In fact not all FSAs have
    // this many frames, most of them have fewer copies, but using this regular
    // structure avoids having to compute any extra row_ids vectors and the
    // like.  The out-of-range elements will be seto to zero.

    Renumbering renumber_states(c_, product);
    char *keep_state_data = renumber_states.Keep().Data();

    int32_t T = T_;
    const int32_t *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();

    float **forward_state_scores_data = forward_state_scores_.Data(),
         **backward_state_scores_data = backward_state_scores_.Data();

    // the following lambda will set elements within `keep_state_data` to 0 or 1.
    auto lambda_set_keep = [=] __host__ __device__ (int32_t i) -> void {
      // i is actually an idx012

      // the following works because each FSA has (its num-states * T_+1) states
      // allocated to it.  However (i / (T_+1)) does not directly map to a state
      // index.
      int32_t fsa_idx0 = a_fsas_row_splits1_data[(i / (T+1))];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float cutoff = score_cutoffs_data[fsa_idx0];

      int32_t idx_within_fsa = i - (T+1) * fsa_info.state_offset,
                    t = idx_within_fsa / fsa_info.num_states,
           state_idx1 = idx_within_fsa % fsa_info.num_states,
           state_idx01 = fsa_info.state_offset + state_idx1;

      char keep = (char)0;
      if (t <= fsa_info.T) {
        // This time is within the bounds for this FSA
        const float *forward_state_scores_t = forward_state_scores_data[t],
                   *backward_state_scores_t = backward_state_scores_data[t];
        if (forward_state_scores_t[state_idx01] +
            backward_state_scores_t[state_idx01] > cutoff)
          keep = (char)1;
      }
<<<<<<< HEAD
      keep_state_data[i] = keep;
    };
    Eval(c_, product, lambda_set_keep);

    Array1<int32_t> &new2old = renumber_states.New2Old();
    const int32_t *new2old_data = new2old.Data();
    int32_t ans_tot_num_states = new2old.Dim();

    // t_per_fsa will be set below to the number of time-steps that each FSA has
    // states active on; if each FSA i has scores for 0 <= t < T_i, then
    // t_per_fsa[i] will be T_i + 1, because there is also a copy of the state
    // at time T_i.
    Array1<int32_t> t_per_fsa(c_, num_fsas_ + 1);
    int32_t *t_per_fsa_data = t_per_fsa.Data();
    auto lambda_set_t_per_fsa_etc = [=] __host__ __device__ (int32_t i) -> void {
      t_per_fsa_data[i] = fsa_info_data[i].T + 1;
    };
    Eval(c_, num_fsas_, lambda_set_t_per_fsa_etc);
    ExclusiveSum(t_per_fsa, &t_per_fsa);

    // now t_per_fsa is the row_splits1 of the shape we'll be returning.  It allocates
    // fsa_info_data[i].T + 1 time-indexes to the i'th fsa.
    Array1<int32_t> &ans_row_splits1 = t_per_fsa;
    const int32_t *ans_row_splits1_data = ans_row_splits1.Data();
    Array1<int32_t> ans_row_ids1(c_, t_per_fsa.Back());
    RowSplitsToRowIds(ans_row_splits1, &ans_row_ids1);

    // ans_row_ids2 maps to an ans_idx01 that combines FSA-index and time-index.
    Array1<int32_t> ans_row_ids2(c_, ans_tot_num_states);
    int32_t *ans_row_ids2_data = ans_row_ids2.Data();
    // ans_num_arcs is the number of arcs potentially active for a state; we'll
    // prune out the invalid ones later on.
    Array1<int32_t> ans_num_arcs(c_, ans_tot_num_states + 1);
    int32_t *ans_num_arcs_data = ans_num_arcs.Data();

    // ans_state_idx01 contains the state_idx01 in a_fsas_ for each state in
    // the answer.
    Array1<int32_t> ans_state_idx01(c_, ans_tot_num_states);
    int32_t *ans_state_idx01_data = ans_state_idx01.Data();
    const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data();

    // set ans_row_ids2_data, which contains an ans_idx01 that combines
    // FSA-index and time-index.
    auto lambda_set_row_ids2 = [=] __host__ __device__ (int32_t ans_idx012) -> void {
      // old_i is the same as the index `i` into lambda_set_keep.  It is also an idx012.
      // The logic is the same as for lambda_set_keep, we keep the code but not the
      // comments.
      int32_t old_i = new2old_data[ans_idx012];
      int32_t fsa_idx0 = a_fsas_row_splits1_data[(old_i / (T+1))];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      int32_t idx_within_fsa = old_i - (T+1) * fsa_info.state_offset,
                           t = idx_within_fsa / fsa_info.num_states,
           a_fsas_state_idx1 = idx_within_fsa % fsa_info.num_states;
      int32_t a_fsas_state_idx01 = fsa_info.state_offset + a_fsas_state_idx1;
      int32_t ans_fsa_idx0x = ans_row_splits1_data[fsa_idx0],
                  ans_idx01 = ans_fsa_idx0x + t;
      ans_row_ids2_data[ans_idx012] = ans_idx01;
      ans_state_idx01_data[ans_idx012] = a_fsas_state_idx1;
      // note: fsa_info.state_offset == a_fsas_row_splits2_data[a_fsas_state_idx01];
      int32_t num_arcs = a_fsas_row_splits2_data[a_fsas_state_idx01 + 1] -
              a_fsas_row_splits2_data[a_fsas_state_idx01];
      if (t == fsa_info.T)  // No arcs leave copies of states on the last frame
                            // for each FSA.
        num_arcs = 0;
      K2_CHECK_EQ(0, 0); // temp
      ans_num_arcs_data[ans_idx012] = num_arcs;
    };
    Eval(c_, ans_tot_num_states, lambda_set_row_ids2);

    Array1<int32_t> &ans_row_splits3(ans_num_arcs);
    ExclusiveSum(ans_num_arcs, &ans_row_splits3);
    int32_t tot_arcs = ans_row_splits3.Back();
    Array1<int32_t> ans_row_ids3(c_, tot_arcs);
    RowSplitsToRowIds(ans_row_splits3, &ans_row_ids3);

    // Actually we'll do one more pass of pruning on 'ans' before we return it.
    Ragged<Arc> ans(
        RaggedShape4(&ans_row_splits1, &ans_row_ids1, -1,
                     nullptr, &ans_row_ids2, ans_tot_num_states,
                     &ans_row_splits3, &ans_row_ids3, -1),
        Array1<Arc>(c_, tot_arcs));
    Arc *ans_arcs_data = ans.values.Data();

    Array1<int32_t> arc_map_a(c_, tot_arcs),
        arc_map_b(c_, tot_arcs);
    int32_t *arc_map_a_data = arc_map_a.Data(),
            *arc_map_b_data = arc_map_b.Data();

    Renumbering renumber_arcs(c_, tot_arcs);
    char *keep_arc_data = renumber_arcs.Keep().Data();

    const int32_t *ans_row_ids1_data = ans_row_ids1.Data(),
                  *ans_row_ids3_data = ans_row_ids3.Data(),
               *ans_row_splits2_data = ans.shape.RowSplits(2).Data(),
               *ans_row_splits3_data = ans_row_splits3.Data(),
                *states_old2new_data = renumber_states.Old2New().Data();
    CompressedArc *carcs_data = carcs_.Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();
    const float *scores_data = b_fsas_.scores.Data();

    auto lambda_set_arcs_and_keep = [=] __host__ __device__ (int32_t arc_idx0123) -> void {
      int32_t ans_state_idx012 = ans_row_ids3_data[arc_idx0123],
                   ans_idx012x = ans_row_splits3_data[ans_state_idx012],
                     ans_idx01 = ans_row_ids2_data[ans_state_idx012],
                      fsa_idx0 = ans_row_ids1_data[ans_idx01],
                     ans_idx0x = ans_row_splits1_data[fsa_idx0],
                    ans_idx0xx = ans_row_splits2_data[ans_idx0x],
                        t_idx1 = ans_idx01 - ans_idx0x,
                      arc_idx3 = arc_idx0123 - ans_idx012x;
      int32_t a_fsas_state_idx01 = ans_state_idx01_data[ans_state_idx012];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float cutoff = score_cutoffs_data[fsa_idx0];
      int32_t a_fsas_state_idx0x = fsa_info.state_offset,
              a_fsas_state_idx1 = a_fsas_state_idx01 - a_fsas_state_idx0x;
      int32_t a_fsas_arc_idx012 = a_fsas_row_splits2_data[a_fsas_state_idx01]
                   + arc_idx3; //  arc_idx3 is an idx2 w.r.t. a_fsas.
      CompressedArc carc = carcs_data[a_fsas_arc_idx012];
      K2_CHECK_EQ(a_fsas_state_idx1, (int32_t)carc.src_state);
      int32_t a_fsas_dest_state_idx1 = carc.dest_state,
            a_fsas_dest_state_idx01 = fsa_info.state_offset + a_fsas_dest_state_idx1;
      arc_map_a_data[arc_idx0123] = a_fsas_arc_idx012;
      int32_t scores_index = fsa_info.scores_offset + (scores_stride * t_idx1) +
                       carc.label_plus_one;
      arc_map_b_data[arc_idx0123] = scores_index;

      float arc_score = carc.score + scores_data[scores_index];

      // unpruned_src_state_idx and unpruned_dest_state_idx are into
      // `renumber_states.Keep()` or `renumber_states.Old2New()`
      int32_t unpruned_src_state_idx = fsa_info.state_offset * (T+1) +
         ((t_idx1 + 1) * fsa_info.num_states) + a_fsas_state_idx1,
         unpruned_dest_state_idx = fsa_info.state_offset * (T+1) +
         ((t_idx1 + 1) * fsa_info.num_states) + a_fsas_dest_state_idx1;
      K2_CHECK_EQ(states_old2new_data[unpruned_src_state_idx], ans_state_idx012);
      K2_CHECK_LT(t_idx1, (int32_t)fsa_info.T);

      int32_t ans_dest_state_idx012 = states_old2new_data[unpruned_dest_state_idx],
         ans_dest_state_idx012_next = states_old2new_data[unpruned_dest_state_idx + 1];
      char keep_this_arc = (char)0;

      const float *forward_state_scores_t = forward_state_scores_data[t_idx1],
                  *backward_state_scores_t1 = backward_state_scores_data[t_idx1 + 1];

      if (ans_dest_state_idx012 > ans_dest_state_idx012_next) {
        // The dest-state of this arc has a number (was not pruned away)
        float arc_forward_backward_score = forward_state_scores_t[a_fsas_state_idx01] +
                                           arc_score +
                                           backward_state_scores_t1[a_fsas_dest_state_idx01];
        if (arc_forward_backward_score > cutoff) {
          keep_this_arc = (char)1;
          Arc arc;
          arc.label = static_cast<int32_t>(carc.label_plus_one) - 1;
          // the idx12 into `ans`, which includes the 't' and 'state' indexes,
          // corresponds to the state-index in the FSA we will return (the 't' index
          // will be removed).
          int32_t src_state_idx12 = ans_state_idx012 - ans_idx0xx,
                 dest_state_idx12 = ans_dest_state_idx012 - ans_idx0xx;
          arc.src_state = src_state_idx12;
          arc.dest_state = dest_state_idx12;
          arc.score = arc_score;
          ans_arcs_data[arc_idx0123] = arc;
        }
        keep_arc_data[arc_idx0123] = keep_this_arc;
=======
      char keep_this_state =
          (backward_loglike + forward_loglike >= -output_beam);

      // we can use the arcs row-splits because the structure of
      // FrameInfo::states is the same as the top level structure of
      // FrameInfo::arcs.
      int32_t states_idx0x = arcs_row_splits1[states_idx0],
              states_idx1 = state_idx01 - states_idx0x;

      int32_t oshape_idx0x = oshape_row_splits1[states_idx0],
              oshape_idx01 = oshape_idx0x + t,
              oshape_idx01x = oshape_row_splits2[oshape_idx01],
              oshape_idx012 = oshape_idx01x + states_idx1;
      // note: axis 1 of 'states' corresponds to axis 2 of 'oshape'; it's the
      // state index.  Also,

      keep_states_data[oshape_idx012] = keep_this_state;
      if (!keep_this_state) {
        // The reason we set the backward_loglike to -infinity here if it's
        // outside the beam, is to prevent disconnected states from appearing
        // after pruning due to numerical roundoff effects near the boundary
        // at `-beam`.  It would otherwise be correct and harmless to omit
        // this if-block.
        backward_loglike = minus_inf;
>>>>>>> upstream/master
      }
    };
    Eval(c_, tot_arcs, lambda_set_arcs_and_keep);

    if (arc_map_a_out)
      *arc_map_a_out = arc_map_a[renumber_arcs.New2Old()];
    if (arc_map_b_out)
      *arc_map_b_out = arc_map_b[renumber_arcs.New2Old()];
    // subsample the output shape, removing arcs that weren't kept
    RaggedShape ans_shape_subsampled = SubsampleRaggedShape(ans.shape,
                                                            renumber_arcs);
    // .. and remove the 't' axis
    return Ragged<Arc>(RemoveAxis(ans_shape_subsampled, 1),
                       ans.values[renumber_arcs.New2Old()]);
  }




  ContextPtr c_;
  FsaVec &a_fsas_;          // a_fsas_: decoding graphs, with same Dim0() as
                            // b_fsas_. Note: a_fsas_ has 3 axes.

  DenseFsaVec &b_fsas_;

  // num_fsas_ equals b_fsas_.shape.Dim0() == a_fsas_.Dim0().
  int32_t num_fsas_;

  // This is just a copy of a_fsas_.arcs, with a couple extra pieces of information.
  struct CompressedArc {
    // src_state of Arc, as uint16 (an idx1)
    uint16_t src_state;
    // dest_state of Arc, as uint16 (an idx1)
    uint16_t dest_state;
    // label of Arc, plus one, as uint16
    uint16_t label_plus_one;
    // FSA index, as uint16.
    uint16_t fsa_idx;
    // The idx012 of the position of this arc in the 'incoming arcs' array as
    // returned by GetIncomingArcs().  This is where we'll write the end-loglike
    // of this arc in the forward propagation, to make the reduction easier.
    int32_t incoming_arc_idx012;
    float score;
  };
  // The arcs in a_fsas_.arcs, converted to int16_t's and with a little more information.
  Array1<CompressedArc> carcs_;

  // incoming_arcs_.shape is a modified version of the shape a_fsas_.arcs.shape,
  // so arcs are arranged by dest_state rather than src_state, as returned by
  // GetIncomingArcs().  It's used to do reductions for the forward-pass
  // computation.
  Ragged<int32_t> incoming_arcs_;


  // forward_then_backward_shape_ is a shape with NumAxes()==3, equal to
  // (incoming_arcs_.shape, a_fsas_.shape) appended together.  The name is
  // because incoming_arcs_.shape is used for reduction of arc-scores to
  // state-scores in the forward pass, and a_fsas_.shape in the backward pass.
  // Prefixes of forward_then_backward_shape_ are used as the shape of the
  // "early" members of fb_state_scores_, those for which t_forward <= t_backward.
  // FSAs active in the forward pass is >= those active in the backward pass.
  RaggedShape forward_then_backward_shape_;

  // backward_then_forward_shape is (a_fsas_.shape, incoming_arcs_.shape)
  // appended together.  Read docs for forward_then_backward_shape_ to
  // understand.
  RaggedShape backward_then_forward_shape_;

  // Temporary array used in combined forward+backward computation, of dimension
  // a_fsas_.TotSize(2) * 2.  Used indirectly through fb_arc_scores_, which
  // share this data.
  Array1<float> arc_scores_;

  struct FsaInfo {
    // T is the number of frames in b_fsas_.scores that we have for this FSA, i.e.
    // `b_fsas_.scores.RowSplits(1)[i+1] -  b_fsas_.scores.RowSplits(1)[i].`
    // The number of copies of the states of a_fsas_ that we have in the total
    // state space equals T+1, i.e. we have copies of those states for times
    // 0 <= t <= T.
    uint16_t T;
    // num_states is the number of states this FSA has.
    uint16_t num_states;
    // scores_offset is the offset of first location in b_fsas_.scores.Data()
    // that is for this FSA, i.e. b_fsas_.scores.Data()[scores_offset] is the
    // score for t=0, symbol=-1 of this FSA.
    int32_t scores_offset;
    // state_offset is the idx0x corresponding to this FSA in a_fsas_.
    int32_t state_offset;
    // arc_offset is the idx0xx corresponding to this FSA in a_fsas_.
    int32_t arc_offset;
  };
  // fsa_info_ is of dimension num_fsas_ + 1 (the last one is not correct in all
  // respects, only certain fields make sense).
  Array1<FsaInfo> fsa_info_;

  struct Step {
    // 0 < forward_t <= T_ is the time whose states we are writing to in the
    // forward pass on this step of the algorithm (we read from those on time
    // forward_t - 1)
    int32_t forward_t;
    // backward_t = T_ - t_forward is the time whose states we are writing
    // to in the backward pass on this step of the algorithm (we read from those
    // on time backward_t + 1)
    int32_t backward_t;

    // true if forward_t <= backward_t.  Affects the order in which we append
    // the states we're processing for the forward and backward passes into a single
    // array (if true, forward goes first).
    bool forward_before_backward;

    // forward_num_fsas == num_fsas_for_t_[forward_t] is the number of FSAs that have states
    // active on t == forward_t.
    int32_t forward_num_fsas;

    // forward_num_fsas_full is num_fsas_ if forward_before_backward == true,
    // else forward_num_fsas.  (This is a padding we use so that we can share the
    // shape information for the arrays, they become prefixes of the same array).
    int32_t forward_num_fsas_full;

    // backward_num_fsas == num_fsas_for_t_[backward_t] is the number of FSAs that have states
    // active on t == backward_t.
    int32_t backward_num_fsas;

    // backward_num_fsas_full is num_fsas_ if backward_before_backward == true,
    // else backward_num_fsas.  (This is a padding we use so that we can share the
    // shape information for the arrays, they become prefixes of the same array).
    int32_t backward_num_fsas_full;

    // Ragged array where we will write the scores of arcs before reduction.
    // The shapes for all of these use shared memory, and so do the floats
    // (they are sub-arrays of arc_scores_).
    // If forward_before_backward it contains the forward scores first;
    // else the backward scores.
    Ragged<float> arc_scores;

    // forward_arc_scores is a sub-array of `arc_scores` containing
    // the arc-end probs we write for the forward pass prior to reduction.
    // Its Dim() is the total num-arcs corresponding to forward_num_fsas.
    Array1<float> forward_arc_scores;

    // backward_arc_scores is a sub-array of `arc_scores` containing
    // the arc-begin probs we write for the backward pass prior to reduction.
    // Its Dim() is the total num-arcs corresponding to backward_num_fsas.
    Array1<float> backward_arc_scores;

    // `state_scores` is where we reduce `arc_scores` to, its Dim() equals
    // arc_scores.TotSize(1).  [arc_scores has 3 axes]. This storage is ACTALLY
    // ALLOCATED HERE, unlike other arrays declared here.  state_scores.Dim() is
    // the num-states corresponding to backward_num_fsas_full plus the
    // num-states corresponding to forward_num_fsas_full (the order depends on
    // whether forward_before_backward == true).
    Array1<float> state_scores;

    // `forward_state_scores` is the sub-array of `state_scores` containing just
    // the forward probs written in this step; its Dim() is the number of states
    // corresponding to forward_num_fsas_full.  It's not needed here directly,
    // but in the next step.
    Array1<float> forward_state_scores;

    // `backward_state_scores` is the sub-array of `state_scores` containing just
    // the backward probs written in this step; its Dim() is the number of
    // states corresponding to backward_num_fsas_full.  It's not needed here
    // directly, but in the next step.
    Array1<float> backward_state_scores;
  };

  // steps_.size() ==  T_ + 1.
  // steps_[0] is "special", on that step we do initialization.
  std::vector<Step> steps_;

  // It happens to be convenient to cache these two things here; they point to
  // data owned in the elements of steps_.forward_state_scores and
  // backward_state_scores.
  Array1<float*> forward_state_scores_;
  Array1<float*> backward_state_scores_;

  float output_beam_;

  int32_t T_;  // == b_fsas_.MaxSize(1)

  // forward_probs_ contains the forward probabilities.
  // Its NumAxes() == 2, it's of dimension [num_fsas][tot_states_per_fsa]
  // where "tot_states_per_fsa" correponds to the elements of tot_states_per_fsa_.
  Ragged<float> forward_probs_;

  // backward_probs_ has the same shape as forward_probs_.  backward_probs_[i] +
  // forward_probs[i] equals the probability of paths including that arc divided
  // by the total-prob of the lattice.
  Ragged<float> backward_probs_;

  // forward_probs_temp_ is a temporary used in computing forward_probs_ on each
  // frames; its dimension is a_fsas_.TotSize(2) * (b_fsas_.Dim0() /
  // a_fsas_.Dim0()), and it's arranged the same way as a_fsas_incoming_.
  Array1<float> forward_probs_temp_;

  // This is as oshape_unpruned_, but after the backward-pass pruning.
  // It is indexed [fsa_id][t][state][arc].
  RaggedShape oshape_pruned_;
};


void IntersectDense(FsaVec &a_fsas, DenseFsaVec &b_fsas, float output_beam,
                    FsaVec *out,
                    Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b) {
  NVTX_RANGE("IntersectDense");
  FsaVec a_vec = FsaToFsaVec(a_fsas);
  MultiGraphDenseIntersect intersector(a_vec, b_fsas,
                                       output_beam);
  intersector.Intersect();
  FsaVec ret = intersector.FormatOutput(arc_map_a, arc_map_b);
  *out = ret;
}
}  // namespace k2
