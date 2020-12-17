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
#include "k2/csrc/hash.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

namespace intersect_pruned_internal {

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

struct ArcInfo {              // for an arc that wasn't pruned away...
  int32_t a_fsas_arc_idx012;  // the arc-index in a_fsas_.
  float arc_loglike;          // loglike on this arc: equals loglike from data
  // (nnet output, == b_fsas), plus loglike from
  // the arc in a_fsas.

  union {
    // these 3 different ways of storing the index of the destination state
    // are used at different stages of the algorithm; we give them different
    // names for clarity.
    int32_t dest_a_fsas_state_idx01;  // The destination-state as an index
    // into a_fsas_.
    int32_t dest_info_state_idx1;  // The destination-state as an idx1 into the
                                   // next FrameInfo's `arcs` or `states`,
                                   // omitting the FSA-index which can be worked
                                   // out from the structure of this frame's
                                   // ArcInfo.
  } u;
  float end_loglike;  // loglike at the end of the arc just before
  // (conceptually) it joins the destination state.
};

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


}  // namespace intersect_pruned_internal

using namespace intersect_pruned_internal;  // NOLINT

// Caution: this is really a .cu file.  It contains mixed host and device code.

/*
   Pruned intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  Can use either different decoding graphs (one
   per acoustic sequence) or a shared graph
*/
class MultiGraphDenseIntersectPruned {
 public:
  /**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Size0()==1 in which
                           case the graph is shared.
       @param [in] b_fsas  The neural-net output, with each frame containing the
                           log-likes of each phone.  A series of sequences of
                           (in general) different length.
       @param [in] search_beam    "Default" search/decoding beam.  The actual
                           beam is dynamic and also depends on max_active and
                           min_active.
       @param [in] output_beam    Beam for pruning the output FSA, will
                                  typically be smaller than search_beam.
       @param [in] min_active  Minimum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to have fewer than this
                           number active.
       @param [in] max_active  Maximum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to exceed that but may not
                           always succeed.  This determines the hash size.
   */
  MultiGraphDenseIntersectPruned(FsaVec &a_fsas, DenseFsaVec &b_fsas,
                                 float search_beam, float output_beam,
                                 int32_t min_active, int32_t max_active)
      : a_fsas_(a_fsas),
        b_fsas_(b_fsas),
        search_beam_(search_beam),
        output_beam_(output_beam),
        min_active_(min_active),
        max_active_(max_active),
        dynamic_beams_(a_fsas.Context(), b_fsas.shape.Dim0(), search_beam) {
    NVTX_RANGE(K2_FUNC);
    c_ = GetContext(a_fsas.shape, b_fsas.shape);
    K2_CHECK(b_fsas.scores.IsContiguous());
    K2_CHECK_GT(search_beam, 0);
    K2_CHECK_GT(output_beam, 0);
    K2_CHECK_GE(min_active, 0);
    K2_CHECK_GT(max_active, min_active);
    K2_CHECK(a_fsas.shape.Dim0() == b_fsas.shape.Dim0() ||
             a_fsas.shape.Dim0() == 1);
    K2_CHECK_GE(b_fsas.shape.Dim0(), 1);
    int32_t num_seqs = b_fsas.shape.Dim0();

    int32_t num_buckets = RoundUpToNearestPowerOfTwo(num_seqs * 4 *
                                                     max_active);
    if (num_buckets < 128)
      num_buckets = 128;
    state_map_ = Hash32(c_, num_buckets);
    int32_t num_a_copies;
    if (a_fsas.shape.Dim0() == 1) {
      a_fsas_stride_ = 0;
      state_map_fsa_stride_ = a_fsas.TotSize(1);
      num_a_copies = b_fsas.shape.Dim0();
    } else {
      K2_CHECK_EQ(a_fsas.shape.Dim0(), b_fsas.shape.Dim0());
      a_fsas_stride_ = 1;
      state_map_fsa_stride_ = 0;
      num_a_copies = 1;
    }
    int64_t num_keys = num_a_copies * (int64_t)a_fsas.TotSize(1);
    K2_CHECK(num_keys == (uint32_t)num_keys);
  }

  // The information we have for each frame of the pruned-intersection (really:
  // decoding) algorithm.  We keep an array of these, one for each frame, up to
  // the length of the longest sequence we're decoding plus one.
  struct FrameInfo {
    // States that are active at the beginning of this frame.  Indexed
    // [fsa_idx][state_idx], where fsa_idx indexes b_fsas_ (and a_fsas_, if
    // a_fsas_stride_ != 0); and state_idx just enumerates the active states
    // on this frame (as state_idx01's in a_fsas_).
    Ragged<StateInfo> states;  // 2 axes: fsa, state

    // Indexed [fsa_idx][state_idx][arc_idx].. the first 2 indexes are
    // the same as those into 'states' (the first 2 levels of the structure
    // are shared), and the last one enumerates the arcs leaving each of those
    // states.
    //
    // Note: there may be indexes [fsa_idx] that have no states (because that
    // FSA had fewer frames than the max), and indexes [fsa_idx][state_idx] that
    // have no arcs due to pruning.
    Ragged<ArcInfo> arcs;  // 3 axes: fsa, state, arc
  };

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
    T_ = b_fsas_.shape.MaxSize(1);
    int32_t num_fsas = b_fsas_.shape.Dim0(), T = T_;

    std::ostringstream os;
    os << "Intersect:T=" << T << ",num_fsas=" << num_fsas
       << ",TotSize(1)=" << b_fsas_.shape.TotSize(1);
    NVTX_RANGE(os.str().c_str());

    // we'll initially populate frames_[0.. T+1], but discard the one at T+1,
    // which has no arcs or states, the ones we use are from 0 to T.
    frames_.reserve(T + 2);

    frames_.push_back(InitialFrameInfo());

    for (int32_t t = 0; t <= T; t++) {
      frames_.push_back(PropagateForward(t, frames_.back().get()));
    }
    // The FrameInfo for time T+1 will have no states.  We did that
    // last PropagateForward so that the 'arcs' member of frames_[T]
    // is set up (it has no arcs but we need the shape).
    frames_.pop_back();

    SetBackwardProbsFinal(frames_[T].get());
    for (int32_t t = T - 1; t >= 0; t--) {
      PropagateBackwardAndPrune(t, frames_[t].get(),
                                frames_[t + 1].get());
    }
  }

  // Return FrameInfo for 1st frame, with `states` set but `arcs` not set.
  std::unique_ptr<FrameInfo> InitialFrameInfo() {
    NVTX_RANGE("InitialFrameInfo");
    int32_t num_fsas = b_fsas_.shape.Dim0();
    std::unique_ptr<FrameInfo> ans = std::make_unique<FrameInfo>();

    if (a_fsas_.Dim0() == 1) {
      int32_t start_states_per_seq = (a_fsas_.shape.TotSize(1) > 0),  // 0 or 1
          num_start_states = num_fsas * start_states_per_seq;
      ans->states = Ragged<StateInfo>(
          RegularRaggedShape(c_, num_fsas, start_states_per_seq),
          Array1<StateInfo>(c_, num_start_states));
      StateInfo *states_data = ans->states.values.Data();
      K2_EVAL(
          c_, num_start_states, lambda_set_states, (int32_t i)->void {
            StateInfo info;
            info.a_fsas_state_idx01 = 0;  // start state of a_fsas_
            info.forward_loglike = FloatToOrderedInt(0.0);
            states_data[i] = info;
          });
    } else {
      Ragged<int32_t> start_states = GetStartStates(a_fsas_);
      ans->states =
          Ragged<StateInfo>(start_states.shape,
                            Array1<StateInfo>(c_, start_states.NumElements()));
      StateInfo *ans_states_values_data = ans->states.values.Data();
      const int32_t *start_states_values_data = start_states.values.Data(),
                    *start_states_row_ids1_data =
                        start_states.shape.RowIds(1).Data();
      K2_EVAL(
          c_, start_states.NumElements(), lambda_set_state_info,
          (int32_t states_idx01)->void {
            StateInfo info;
            info.a_fsas_state_idx01 = start_states_values_data[states_idx01];
            info.forward_loglike = FloatToOrderedInt(0.0);
            ans_states_values_data[states_idx01] = info;
          });
    }
    return ans;
  }

  void FormatOutput(FsaVec *ofsa, Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b) {
    NVTX_RANGE("FormatOutput");

    int32_t T = T_;


    ContextPtr c_cpu = GetCpuContext();
    Array1<ArcInfo *> arcs_data_ptrs(c_cpu, T + 1);
    Array1<int32_t *> arcs_row_splits1_ptrs(c_cpu, T + 1);
    for (int32_t t = 0; t <= T; t++) {
      arcs_data_ptrs.Data()[t] = frames_[t]->arcs.values.Data();
      arcs_row_splits1_ptrs.Data()[t] = frames_[t]->arcs.RowSplits(1).Data();
    }
    // transfer to GPU if we're using a GPU
    arcs_data_ptrs = arcs_data_ptrs.To(c_);
    ArcInfo **arcs_data_ptrs_data = arcs_data_ptrs.Data();
    arcs_row_splits1_ptrs = arcs_row_splits1_ptrs.To(c_);
    int32_t **arcs_row_splits1_ptrs_data = arcs_row_splits1_ptrs.Data();
    const int32_t *b_fsas_row_splits1 = b_fsas_.shape.RowSplits(1).Data();
    const int32_t *a_fsas_row_splits1 = a_fsas_.RowSplits(1).Data();
    int32_t a_fsas_stride = a_fsas_stride_;  // 0 or 1 depending if the decoding
                                             // graph is shared.
    int32_t num_fsas = b_fsas_.shape.Dim0();

    RaggedShape final_arcs_shape;
    { /*  This block populates `final_arcs_shape`.  It is the shape of a ragged
          tensor of arcs that conceptually would live at frames_[T+1]->arcs.  It
          contains no actual arcs, but may contain some states, that represent
          "missing" final-states.  The problem we are trying to solve is that
          there was a start-state for an FSA but no final-state because it did
          not survive pruning, and this could lead to an output FSA that is
          invalid or is misinterpreted (because we are interpreting a non-final
          state as a final state).
       */
      Array1<int32_t> num_extra_states(c_, num_fsas + 1);
      int32_t *num_extra_states_data = num_extra_states.Data();
      K2_EVAL(c_, num_fsas, lambda_set_num_extra_states, (int32_t i) -> void {
          int32_t final_t = b_fsas_row_splits1[i+1] - b_fsas_row_splits1[i];
          int32_t *arcs_row_splits1_data = arcs_row_splits1_ptrs_data[final_t];
          int32_t num_states_final_t = arcs_row_splits1_data[i + 1] -
                                       arcs_row_splits1_data[i];
          K2_CHECK_LE(num_states_final_t, 1);

          // has_start_state is 1 if there is a start-state; note, we don't prune
          // the start-states, so they'll be present if they were present in a_fsas_.
          int32_t has_start_state = (a_fsas_row_splits1[i * a_fsas_stride] <
                                     a_fsas_row_splits1[i * a_fsas_stride + 1]);

          // num_extra_states_data[i] will be 1 if there was a start state but no final-state;
          // else, 0.
          num_extra_states_data[i] = has_start_state * (1 - num_states_final_t);
        });
      K2_LOG(INFO) << "num_extra_states = " << num_extra_states;
      ExclusiveSum(num_extra_states, &num_extra_states);

      RaggedShape top_shape = RaggedShape2(&num_extra_states, nullptr, -1),
               bottom_shape = RegularRaggedShape(c_, top_shape.NumElements(), 0);
      final_arcs_shape = ComposeRaggedShapes(top_shape, bottom_shape);
    }


    RaggedShape oshape;
    // see documentation of Stack() in ragged_ops.h for explanation.
    Array1<uint32_t> oshape_merge_map;

    {
      NVTX_RANGE("InitOshape");
      // each of these have 3 axes.
      std::vector<RaggedShape *> arcs_shapes(T + 2);
      for (int32_t t = 0; t <= T; t++)
        arcs_shapes[t] = &(frames_[t]->arcs.shape);
      arcs_shapes[T + 1] = &final_arcs_shape;

      // oshape is a 4-axis ragged tensor which is indexed:
      //   oshape[fsa_index][t][state_idx][arc_idx]
      int32_t axis = 1;
      oshape = Stack(axis, T + 2, arcs_shapes.data(), &oshape_merge_map);
    }


    int32_t *oshape_row_ids3 = oshape.RowIds(3).Data(),
            *oshape_row_ids2 = oshape.RowIds(2).Data(),
            *oshape_row_ids1 = oshape.RowIds(1).Data(),
            *oshape_row_splits3 = oshape.RowSplits(3).Data(),
            *oshape_row_splits2 = oshape.RowSplits(2).Data(),
            *oshape_row_splits1 = oshape.RowSplits(1).Data();


    int32_t num_arcs = oshape.NumElements();
    *arc_map_a = Array1<int32_t>(c_, num_arcs);
    *arc_map_b = Array1<int32_t>(c_, num_arcs);
    int32_t *arc_map_a_data = arc_map_a->Data(),
            *arc_map_b_data = arc_map_b->Data();
    Array1<Arc> arcs_out(c_, num_arcs);
    Arc *arcs_out_data = arcs_out.Data();
    const Arc *a_fsas_arcs = a_fsas_.values.Data();
    int32_t b_fsas_num_cols = b_fsas_.scores.Dim1();
    const int32_t *b_fsas_row_ids1 = b_fsas_.shape.RowIds(1).Data();

    const uint32_t *oshape_merge_map_data = oshape_merge_map.Data();

    K2_EVAL(
        c_, num_arcs, lambda_format_arc_data,
        (int32_t oarc_idx0123)->void {  // by 'oarc' we mean arc with shape `oshape`.
          int32_t oarc_idx012 = oshape_row_ids3[oarc_idx0123],
                   oarc_idx01 = oshape_row_ids2[oarc_idx012],
                    oarc_idx0 = oshape_row_ids1[oarc_idx01],
                   oarc_idx0x = oshape_row_splits1[oarc_idx0],
                  oarc_idx0xx = oshape_row_splits2[oarc_idx0x],
                    oarc_idx1 = oarc_idx01 - oarc_idx0x,
             oarc_idx01x_next = oshape_row_splits2[oarc_idx01 + 1];

          int32_t m = oshape_merge_map_data[oarc_idx0123],
                  t = m % (T + 2),  // actually we won't get t == T or t == T + 1
                                    // here since those frames have no arcs.
        arcs_idx012 = m / (T + 2);  // arc_idx012 into FrameInfo::arcs on time t,
                                    // index of the arc on that frame.

          K2_CHECK_EQ(t, oarc_idx1);

          const ArcInfo *arcs_data = arcs_data_ptrs_data[t];

          ArcInfo arc_info = arcs_data[arcs_idx012];
          Arc arc;
          arc.src_state = oarc_idx012 - oarc_idx0xx;
          // Note: the idx1 w.r.t. the frame's `arcs` is an idx2 w.r.t. `oshape`.
          int32_t dest_state_idx012 = oarc_idx01x_next +
                                      arc_info.u.dest_info_state_idx1;
          arc.dest_state = dest_state_idx012 - oarc_idx0xx;
          arc.label = a_fsas_arcs[arc_info.a_fsas_arc_idx012].label;

          int32_t fsa_id = oarc_idx0,
            b_fsas_idx0x = b_fsas_row_splits1[fsa_id],
            b_fsas_idx01 = b_fsas_idx0x + t,
             b_fsas_idx2 = (arc.label + 1),
       b_fsas_arc_idx012 = b_fsas_idx01 * b_fsas_num_cols + b_fsas_idx2;

          arc.score = arc_info.arc_loglike;
          arc_map_a_data[oarc_idx0123] = arc_info.a_fsas_arc_idx012;
          arc_map_b_data[oarc_idx0123] = b_fsas_arc_idx012;
          arcs_out_data[oarc_idx0123] = arc;
        });

    // Remove axis 1, which corresponds to time.
    *ofsa = FsaVec(RemoveAxis(oshape, 1), arcs_out);
  }

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
                    'search_beam_' as long as the number of active states in
                    each FSA is between min_active and max_active.
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

    K2_EVAL(
        c_, num_fsas, lambda_set_beam_and_cutoffs, (int32_t i)->void {
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
            dynamic_beam *= 0.8;
          }
          dynamic_beams_data[i] = dynamic_beam;
          cutoffs_data[i] = best_loglike - dynamic_beam;
        });

    return cutoffs;
  }

  /*
    Returns list of arcs on this frame, consisting of all arcs leaving
    the states active on 'cur_frame'.

       @param [in] t       The time-index (on which to look up log-likes),
                           t >= 0
       @param [in] cur_frame   The FrameInfo for the current frame; only its
                       'states' member is expected to be set up on entry.
   */
  Ragged<ArcInfo> GetArcs(int32_t t, FrameInfo *cur_frame) {
    NVTX_RANGE(K2_FUNC);
    Ragged<StateInfo> &states = cur_frame->states;
    const StateInfo *state_values = states.values.Data();

    // in a_fsas_ (the decoding graphs), maps from state_idx01 to arc_idx01x.
    const int32_t *fsa_arc_splits = a_fsas_.shape.RowSplits(2).Data();

    int32_t num_states = states.values.Dim();
    Array1<int32_t> num_arcs(c_, num_states + 1);
    int32_t *num_arcs_data = num_arcs.Data();
    // `num_arcs` gives the num-arcs for each state in `states`.
    K2_EVAL(
        c_, num_states, num_arcs_lambda, (int32_t state_idx01)->void {
          int32_t a_fsas_state_idx01 =
                      state_values[state_idx01].a_fsas_state_idx01,
                  a_fsas_arc_idx01x = fsa_arc_splits[a_fsas_state_idx01],
                  a_fsas_arc_idx01x_next =
                      fsa_arc_splits[a_fsas_state_idx01 + 1],
                  a_fsas_num_arcs = a_fsas_arc_idx01x_next - a_fsas_arc_idx01x;
          num_arcs_data[state_idx01] = a_fsas_num_arcs;
        });
    ExclusiveSum(num_arcs, &num_arcs);

    // initialize shape of array that will hold arcs leaving the active states.
    // Its shape is [fsa_index][state][arc]; the top two levels are shared with
    // `states`.  'ai' means ArcInfo.
    RaggedShape ai_shape =
        ComposeRaggedShapes(states.shape, RaggedShape2(&num_arcs, nullptr, -1));

    // from state_idx01 (into `states` or `ai_shape`) -> fsa_idx0
    const int32_t *ai_row_ids1 = ai_shape.RowIds(1).Data();
    // from arc_idx012 (into `ai_shape`) to state_idx01
    const int32_t *ai_row_ids2 = ai_shape.RowIds(2).Data();
    // from state_idx01 to arc_idx01x
    const int32_t *ai_row_splits2 = ai_shape.RowSplits(2).Data();
    // from state_idx01 (into a_fsas_) to arc_idx01x (into a_fsas_)
    const int32_t *a_fsas_row_splits2 = a_fsas_.shape.RowSplits(2).Data();

    const Arc *arcs = a_fsas_.values.Data();
    // fsa_idx0 to ind0x (into b_fsas_), which gives the 1st row for this
    // sequence.
    const int32_t *b_fsas_row_ids1 = b_fsas_.shape.RowIds(1).Data();
    const int32_t *b_fsas_row_splits1 = b_fsas_.shape.RowSplits(1).Data();
    const float *score_data = b_fsas_.scores.Data();
    int32_t scores_num_cols = b_fsas_.scores.Dim1();
    auto scores_acc = b_fsas_.scores.Accessor();

    Ragged<ArcInfo> ai(ai_shape);
    ArcInfo *ai_data = ai.values.Data();  // uninitialized

    K2_EVAL(
        c_, ai.values.Dim(), ai_lambda, (int32_t ai_arc_idx012)->void {
          int32_t ai_state_idx01 = ai_row_ids2[ai_arc_idx012],
                  ai_fsa_idx0 = ai_row_ids1[ai_state_idx01],
                  ai_arc_idx01x = ai_row_splits2[ai_state_idx01],
                  ai_arc_idx2 = ai_arc_idx012 - ai_arc_idx01x;
          StateInfo sinfo = state_values[ai_state_idx01];
          int32_t a_fsas_arc_idx01x =
                      a_fsas_row_splits2[sinfo.a_fsas_state_idx01],
                  a_fsas_arc_idx012 = a_fsas_arc_idx01x + ai_arc_idx2;
          Arc arc = arcs[a_fsas_arc_idx012];

          int32_t scores_idx0x = b_fsas_row_splits1[ai_fsa_idx0],
                  scores_idx01 = scores_idx0x + t,  // t == ind1 into 'scores'
              scores_idx2 =
                  arc.label + 1;  // the +1 is so that -1 can be handled
          K2_DCHECK_LT(static_cast<uint32_t>(scores_idx2),
                       static_cast<uint32_t>(scores_num_cols));
          float acoustic_score = scores_acc(scores_idx01, scores_idx2);
          ArcInfo ai;
          ai.a_fsas_arc_idx012 = a_fsas_arc_idx012;
          ai.arc_loglike = acoustic_score + arc.score;
          ai.end_loglike =
              OrderedIntToFloat(sinfo.forward_loglike) + ai.arc_loglike;
          // at least currently, the ArcInfo object's src_state and dest_state
          // are idx1's not idx01's, i.e. they don't contain the FSA-index,
          // where as the ai element is an idx01, so we need to do this to
          // convert to an idx01; this relies on the fact that
          // sinfo.abs_state_id == arc.src_state
          // + a_fsas_fsa_idx0x.
          ai.u.dest_a_fsas_state_idx01 =
              sinfo.a_fsas_state_idx01 + arc.dest_state - arc.src_state;
          ai_data[ai_arc_idx012] = ai;
        });
    return ai;
  }

  // Later we may choose to support b_fsas_.Dim0() == 1 and a_fsas_.Dim0() > 1,
  // and we'll have to change various bits of code for that to work.
  inline int32_t NumFsas() { return b_fsas_.shape.Dim0(); }

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
    NVTX_RANGE("PropagateForward");
    int32_t num_fsas = NumFsas();
    // Ragged<StateInfo> &states = cur_frame->states;
    // arc_info has 3 axes: fsa_id, state, arc.
    cur_frame->arcs = GetArcs(t, cur_frame);
    Ragged<ArcInfo> &arc_info = cur_frame->arcs;

    ArcInfo *ai_data = arc_info.values.Data();
    Array1<float> ai_data_array1(c_, cur_frame->arcs.values.Dim());
    float *ai_data_array1_data = ai_data_array1.Data();
    K2_EVAL(
        c_, ai_data_array1.Dim(), lambda_set_ai_data,
        (int32_t i)->void { ai_data_array1_data[i] = ai_data[i].end_loglike; });
    Ragged<float> ai_loglikes(arc_info.shape, ai_data_array1);

    // `cutoffs` is of dimension num_fsas.
    Array1<float> cutoffs = GetPruningCutoffs(ai_loglikes);
    if (t % 100 == 0)
      K2_LOG(INFO) << "Dynamic_beams = " << dynamic_beams_;
    float *cutoffs_data = cutoffs.Data();

    // write certain indexes ( into ai.values) to state_map_.Data().  Keeps
    // track of the active states and will allow us to assign a numbering to
    // them.
    int32_t *ai_row_ids1 = arc_info.shape.RowIds(1).Data(),
            *ai_row_ids2 = arc_info.shape.RowIds(2).Data();
    auto state_map_acc = state_map_.GetAccessor();
    int32_t state_map_fsa_stride = state_map_fsa_stride_;

    // renumber_states will be a renumbering that dictates which of the arcs in
    // 'ai' correspond to unique states.  Only one arc for each dest-state is
    // kept (it doesn't matter which one).
    Renumbering renumber_states(c_, arc_info.NumElements());
    char *keep_this_state_data = renumber_states.Keep().Data();


    {
      NVTX_RANGE("LambdaSetStateMap");
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_state_map,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
            int32_t dest_state_idx01 =
                ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            float end_loglike = ai_data[arc_idx012].end_loglike,
                  cutoff = cutoffs_data[fsa_id];
            char keep_this_state = 0;  // only one arc entering any state will
                                       // have its 'keep_this_state_data' entry
                                       // set to 1.
            if (end_loglike > cutoff) {
              int32_t state_map_idx = dest_state_idx01 +
                                      fsa_id * state_map_fsa_stride;
              if (state_map_acc.Insert(state_map_idx, arc_idx012))
                keep_this_state = 1;
            }
            keep_this_state_data[arc_idx012] = keep_this_state;
          });
    }


    int32_t num_states = renumber_states.NumNewElems();
    // state_reorder_data maps from (state_idx01 on next frame) to (the
    // arc_idx012 on this frame which is the source arc which we arbitrarily
    // choose as being "responsible" for the creation of that state).
    int32_t *state_reorder_data = renumber_states.Old2New().Data();

    // state_to_fsa_id maps from an index into the next frame's
    // FrameInfo::states.values() vector to the sequence-id (fsa_id) associated
    // with it.  It should be non-decreasing.
    Array1<int32_t> state_to_fsa_id(c_, num_states);
    {  // This block sets 'state_to_fsa_id'.
      NVTX_RANGE("LambdaSetStateToFsaId");
      int32_t *state_to_fsa_id_data = state_to_fsa_id.Data();
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_state_to_fsa_id,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                    this_state_j = state_reorder_data[arc_idx012],
                    next_state_j = state_reorder_data[arc_idx012 + 1];
            if (next_state_j > this_state_j) {
              state_to_fsa_id_data[this_state_j] = fsa_id;
            }
          });

      K2_DCHECK(IsMonotonic(state_to_fsa_id));
    }

    std::unique_ptr<FrameInfo> ans = std::make_unique<FrameInfo>();
    Array1<int32_t> states_row_splits1(c_, num_fsas + 1);
    RowIdsToRowSplits(state_to_fsa_id, &states_row_splits1);
    ans->states = Ragged<StateInfo>(
        RaggedShape2(&states_row_splits1, &state_to_fsa_id, num_states),
        Array1<StateInfo>(c_, num_states));
    StateInfo *ans_states_data = ans->states.values.Data();
    const int32_t minus_inf_int =
        FloatToOrderedInt(-std::numeric_limits<float>::infinity());
    K2_EVAL(
        c_, num_states, lambda_init_loglike, (int32_t i)->void {
          ans_states_data[i].forward_loglike = minus_inf_int;
        });

    {
      NVTX_RANGE("LambdaModifyStateMap");
      // Modify the elements of `state_map` to refer to the indexes into
      // `ans->states` / `kept_states_data`, rather than the indexes into
      // ai_data. This will decrease some of the values in `state_map`, in
      // general.
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_modify_state_map,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
            int32_t dest_state_idx01 =
                ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            int32_t this_j = state_reorder_data[arc_idx012],
                    next_j = state_reorder_data[arc_idx012 + 1];
            if (next_j > this_j) {
              int32_t state_map_idx = dest_state_idx01 +
                                      fsa_id * state_map_fsa_stride;
              int32_t value, *value_addr;
              bool ans = state_map_acc.Find(state_map_idx,
                                            &value, &value_addr);
              K2_CHECK(ans);
              K2_CHECK_EQ(value, arc_idx012);
              // Note: this_j is an idx01 into ans->states.  previously it
              // contained an arc_idx012 (of the entering arc that won the
              // race).
              *value_addr = this_j;
            }
          });
    }

    // We'll set up the data of the kept states below...
    StateInfo *kept_states_data = ans->states.values.Data();

    {
      int32_t *ans_states_row_splits1_data = ans->states.RowSplits(1).Data();

      NVTX_RANGE("LambdaSetStates");
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_arcs_and_states,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];

            ArcInfo &info = ai_data[arc_idx012];

            int32_t dest_a_fsas_state_idx01 = info.u.dest_a_fsas_state_idx01;


            int32_t state_map_idx = dest_a_fsas_state_idx01 +
                                    fsa_id * state_map_fsa_stride;
            int32_t state_idx01;
            if (!state_map_acc.Find(state_map_idx, &state_idx01))
              state_idx01 = -1;   // The destination state did not survive
                                  // pruning.

            int32_t state_idx1;
            if (state_idx01 >= 0) {
              int32_t state_idx0x = ans_states_row_splits1_data[fsa_id];
              state_idx1 = state_idx01 - state_idx0x;
            } else {
              state_idx1 = -1;  // Meaning: invalid.
            }
            // state_idx1 is the idx1 into ans->states, of the destination
            // state.
            info.u.dest_info_state_idx1 = state_idx1;
            if (state_idx1 < 0)
              return;

            // multiple threads may write the same value to the address written
            // to in the next line.
            kept_states_data[state_idx01].a_fsas_state_idx01 =
                dest_a_fsas_state_idx01;
            int32_t end_loglike_int = FloatToOrderedInt(info.end_loglike);
            // Set the forward log-like of the dest state to the largest of any
            // of those of the incoming arcs.  Note: we initialized this in
            // lambda_init_loglike above.
            atomicMax(&(kept_states_data[state_idx01].forward_loglike),
                      end_loglike_int);
          });
    }
    {
      NVTX_RANGE("LambdaResetStateMap");
      const int32_t *next_states_row_ids1 = ans->states.shape.RowIds(1).Data();
      K2_EVAL(
          c_, ans->states.NumElements(), lambda_reset_state_map,
          (int32_t state_idx01)->void {
            int32_t a_fsas_state_idx01 =
                        kept_states_data[state_idx01].a_fsas_state_idx01,
                fsa_idx0 = next_states_row_ids1[state_idx01];
            int32_t state_map_idx = a_fsas_state_idx01 +
                                    fsa_idx0 * state_map_fsa_stride;
            state_map_acc.Delete(state_map_idx);
          });
    }
    return ans;
  }


  /*
    Sets backward_loglike fields of StateInfo to the negative of the forward
    prob if (this is the final-state or !only_final_probs), else -infinity.

    This is used in computing the backward loglikes/scores for purposes of
    pruning.  This may be done after we're finished decoding/intersecting,
    or while we are still decoding.

    Note: something similar to this (setting backward-prob == forward-prob) is
    also done in PropagateBackward() when we detect final-states.  That's needed
    because not all sequences have the same length, so some may have reached
    their final state earlier.  (Note: we only get to the final-state of a_fsas_
    if we've reached the final frame of the input, because for non-final frames
    we always have -infinity as the log-prob corresponding to the symbol -1.)

    While we are still decoding, a background process will do pruning
    concurrently with the forward computation, for purposes of reducing memory
    usage (and so that most of the pruning can be made concurrent with the
    forward computation).  In this case we want to avoid pruning away anything
    that wouldn't have been pruned away if we were to have waited to the end;
    and it turns out that setting the backward probs to the negative of the
    forward probs (i.e.  for all states, not just final states) accomplishes
    this.  The issue was mentioned in the "Exact Lattice Generation.." paper and
    also in the code for Kaldi's lattice-faster-decoder; search for "As in [3],
    to save memory..."

      @param [in] cur_frame    Frame on which to set the backward probs
  */
  void SetBackwardProbsFinal(FrameInfo *cur_frame) {
    NVTX_RANGE("SetBackwardProbsFinal");
    Ragged<StateInfo> &cur_states = cur_frame->states;  // 2 axes: fsa,state
    int32_t num_states = cur_states.values.Dim();
    if (num_states == 0)
      return;
    StateInfo *cur_states_data = cur_states.values.Data();
    const int32_t *a_fsas_row_ids1_data = a_fsas_.shape.RowIds(1).Data(),
               *a_fsas_row_splits1_data = a_fsas_.shape.RowSplits(1).Data(),
              *cur_states_row_ids1_data = cur_states.RowIds(1).Data();
    double minus_inf = -std::numeric_limits<double>::infinity();

    K2_EVAL(c_, num_states, lambda_set_backward_prob, (int32_t state_idx01) -> void {
        StateInfo *info = cur_states_data + state_idx01;
        double backward_loglike,
            forward_loglike = OrderedIntToFloat(info->forward_loglike);
        if (forward_loglike - forward_loglike == 0) { // not -infinity...
          // canonically we'd set this to zero, but setting it to the forward
          // loglike when this is the final-state (in a_fsas_) has the effect of
          // making the (forward+backward) probs equivalent to the logprob minus
          // the best-path log-prob, which is convenient for pruning.  If this
          // is not actually the last frame of this sequence, which can happen
          // if this was called before the forward decoding process was
          // finished, what we are doing is a form of pruning that is guaranteed
          // not to prune anything out that would not have been pruned out if we
          // had waited until the real end of the file to do the pruning.
          backward_loglike = -forward_loglike;
        } else {
          backward_loglike = minus_inf;
        }
        info->backward_loglike = backward_loglike;
      });
    K2_LOG(INFO) << "states = " << cur_frame->states;
  }

  /*
    Does backward propagation of log-likes, which means setting the
    backward_loglike field of the StateInfo variable (for cur_frame); and
    prune/renumber the states of next_frame and the arcs of cur_frame.

    These backward log-likes are normalized in such a way that you can add them
    with the forward log-likes to produce the log-likelihood ratio vs the best
    path (this will be non-positive).  (To do this, for the final state we have
    to set the backward log-like to the negative of the forward log-like; see
    SetBackwardProbsFinal()).

    This function also prunes arc-indexes on `cur_frame` and state-indexes
    on `next_frame`.

       @param [in] t       The time-index (on which to look up log-likes),
                           t >= 0
       @param [in]  cur_frame The FrameInfo for the frame on which we want to
                              set the forward log-like, and prune the arcs.

       @param [in]  next_frame The next frame's FrameInfo; we will prune the
                             states on this frame (which also affects the
                             shape of the 'arcs').
                             Arcs on `cur_frame` have destination-states
                             on `next_frame`. The `backward_loglike` values
                             of states on `next_frame` are assumed to
                             already be set.
  */
  void PropagateBackwardAndPrune(int32_t t,
                                 FrameInfo *cur_frame,
                                 FrameInfo *next_frame) {
    NVTX_RANGE("PropagateBackwardAndPrune");
    int32_t num_states = cur_frame->states.NumElements(),
            num_arcs = cur_frame->arcs.NumElements();

    int32_t *a_fsas_row_ids1_data = a_fsas_.shape.RowIds(1).Data(),
            *a_fsas_row_splits1_data = a_fsas_.shape.RowSplits(1).Data();

    float minus_inf = -std::numeric_limits<float>::infinity();

    Ragged<float> arc_backward_prob(cur_frame->arcs.shape,
                                    Array1<float>(c_, cur_frame->arcs.NumElements()));
    float *arc_backward_prob_data = arc_backward_prob.values.Data();

    ArcInfo *ai_data = cur_frame->arcs.values.Data();
    int32_t *arcs_rowids1 = cur_frame->arcs.shape.RowIds(1).Data(),
            *arcs_rowids2 = cur_frame->arcs.shape.RowIds(2).Data(),
            *arcs_row_splits1 = cur_frame->arcs.shape.RowSplits(1).Data(),
            *arcs_row_splits2 = cur_frame->arcs.shape.RowSplits(2).Data();
    float output_beam = output_beam_;

    // compute arc backward probs, and set elements of 'keep_arcs'
    int32_t next_num_states = next_frame->states.TotSize(1);

    Renumbering renumber_cur_arcs(c_, num_arcs),
        renumber_next_states(c_, next_num_states);
    char *keep_cur_arcs_data = renumber_cur_arcs.Keep().Data(),
      *keep_next_states_data = renumber_next_states.Keep().Data();

    StateInfo *next_states_data = next_frame->states.values.Data();
    int32_t *next_arcs_row_splits2_data = next_frame->arcs.RowSplits(2).Data();
    K2_EVAL(c_, next_num_states, lambda_set_keep_next_states, (int32_t next_state_idx01) -> void {
        // Keep a state if there are arcs leaving it on the next frame (this is
        // required because we don't want to renumber the next frame's arcs); or
        // if its backward_loglike is nevertheless not minus_inf because it is a
        // final-state.
        keep_next_states_data[next_state_idx01] =
            ((next_arcs_row_splits2_data[next_state_idx01+1] !=
              next_arcs_row_splits2_data[next_state_idx01]) ||
             (next_states_data[next_state_idx01].backward_loglike != minus_inf));
      });

    K2_LOG(INFO) << "For t = " << t << ", renumber_next_states.Keep() = "
                 << renumber_next_states.Keep()
                 << ", next_states = " << next_frame->states;

    Array1<int32_t> next_states_row_splits1 = next_frame->states.RowSplits(1);
    const int32_t *next_states_row_splits1_data_old =
        next_states_row_splits1.Data();

    // Renumber the states on the next frame (note: the shape of the `states` is
    // the same as the 1st layer of the shape of the `arcs`).
    next_frame->arcs.shape = RemoveSomeEmptyLists(next_frame->arcs.shape, 1,
                                                  renumber_next_states);
    next_frame->states.shape = GetLayer(next_frame->arcs.shape, 0);
    next_frame->states.values = next_frame->states.values[renumber_next_states.New2Old()];

    StateInfo *next_states_data_new = next_frame->states.values.Data();

    const int32_t *next_states_row_splits1_data_new
        = next_frame->states.RowSplits(1).Data();

    int32_t *next_states_old2new_data = renumber_next_states.Old2New().Data();

    StateInfo *cur_states_data = cur_frame->states.values.Data();

    // next_states_row_splits1 maps from fsa_idx0 to state_idx01.
    int32_t *next_states_row_splits1_new =
        next_frame->states.shape.RowSplits(1).Data();

    //K2_EVAL(
    //    c_, num_arcs,
    //        lambda_set_arc_backward_prob_and_keep, (int32_t arcs_idx012)->void {

    auto lambda_set_arc_backward_prob_and_keep = [=]  __host__ __device__ (int32_t arcs_idx012) -> void {
          ArcInfo *arc = ai_data + arcs_idx012;
          int32_t state_idx01 = arcs_rowids2[arcs_idx012],
                     fsa_idx0 = arcs_rowids1[state_idx01],
          next_states_idx0x_old = next_states_row_splits1_data_old[fsa_idx0],
          next_states_idx0x_new = next_states_row_splits1_data_new[fsa_idx0];

          // `old` and `new` here are before and after pruning the state-ids of
          // the next frame, see `renumber_next_states`.
          // Note: if dest_state_idx1_old == -1, dest_state_idx01_old has a meaningless
          // value below, but it's never referenced.
          int32_t dest_state_idx1_old = arc->u.dest_info_state_idx1,
                 dest_state_idx01_old = next_states_idx0x_old + dest_state_idx1_old,
                 dest_state_idx01_new = -1,
                  dest_state_idx1_new = -1;
          float backward_loglike = minus_inf;
          char keep_this_arc = 0;
          if (dest_state_idx1_old == -1 ||
              next_states_old2new_data[dest_state_idx01_old + 1] ==
              next_states_old2new_data[dest_state_idx01_old]) {
            // dest_state_idx1_old == -1 means this arc was already pruned in
            // the forward pass.. do nothing.
            // If the second half of the || is true, it means the dest-state of
            // this arc was pruned away.
            // Do nothing.
          } else {
            dest_state_idx01_new = next_states_old2new_data[dest_state_idx01_old];
            dest_state_idx1_new = dest_state_idx01_new - next_states_idx0x_new;
            float arc_loglike = arc->arc_loglike;
            float dest_state_backward_loglike =
                next_states_data_new[dest_state_idx01_new].backward_loglike;
            // 'backward_loglike' is the loglike at the beginning of the arc
            backward_loglike = arc_loglike + dest_state_backward_loglike;
            float src_state_forward_loglike = OrderedIntToFloat(
                cur_states_data[arcs_rowids2[arcs_idx012]].forward_loglike);
            if (backward_loglike + src_state_forward_loglike >= -output_beam) {
              keep_this_arc = 1;
            } else {
              backward_loglike = minus_inf;  // Don't let arcs outside beam
                                             // contribute to their
                                             // start-states's backward prob
                                             // (we'll use that to prune them
                                             // away.)
            }
          }
          keep_cur_arcs_data[arcs_idx012] = keep_this_arc;
          // Correct the dest-state in the arc for the pruning of the next
          // frame's states.
          // We store the idx1 rather than the idx01 because it's more convenient
          // later on while formatting the output, although it's a little
          // inconvenient for this operation.
          arc->u.dest_info_state_idx1 = dest_state_idx1_new;
          arc_backward_prob_data[arcs_idx012] = backward_loglike;
    };
    // TODO: after debugging, maybe use K2_EVAL.
    Eval(c_, num_arcs, lambda_set_arc_backward_prob_and_keep);

    /* note, the elements of state_backward_prob that don't have arcs leaving
       them will be set to the supplied default.  */
    Array1<float> state_backward_prob(c_, num_states);
    MaxPerSublist(arc_backward_prob, minus_inf, &state_backward_prob);

    const float *state_backward_prob_data = state_backward_prob.Data();
    const int32_t *cur_states_row_ids1 =
        cur_frame->states.shape.RowIds(1).Data();

    int32_t num_fsas = NumFsas();
    K2_DCHECK_EQ(cur_frame->states.shape.Dim0(), num_fsas);
    K2_EVAL(
        c_, cur_frame->states.NumElements(), lambda_set_state_backward_prob,
        (int32_t state_idx01)->void {
          StateInfo *info = cur_states_data + state_idx01;
          int32_t fsas_state_idx01 = info->a_fsas_state_idx01,
                  a_fsas_idx0 = a_fsas_row_ids1_data[fsas_state_idx01],
                  fsas_state_idx0x_next = a_fsas_row_splits1_data[a_fsas_idx0 + 1];
          float forward_loglike = OrderedIntToFloat(info->forward_loglike),
                backward_loglike;
          // `is_final_state` means this is the final-state in a_fsas.  this
          // implies it's final in b_fsas too, since they both would have seen
          // symbols -1.
          int32_t is_final_state =
              (fsas_state_idx01 + 1 >= fsas_state_idx0x_next);
          if (is_final_state) {
            // Note: there is only one final-state.
            backward_loglike = -forward_loglike;
          } else {
            backward_loglike = state_backward_prob_data[state_idx01];
          }
          info->backward_loglike = backward_loglike;
        });

    K2_LOG(INFO) << "For t = " << t << ", renumber_cur_arcs.Keep() = "
                 << renumber_cur_arcs.Keep() << ", cur_arcs = "
                 << cur_frame->arcs;

    cur_frame->arcs = SubsampleRagged(cur_frame->arcs,
                                      renumber_cur_arcs);
  }

  ContextPtr c_;
  FsaVec &a_fsas_;         // Note: a_fsas_ has 3 axes.
  int32_t a_fsas_stride_;  // 1 if we use a different FSA per sequence
                           // (a_fsas_.Dim0() > 1), 0 if the decoding graph is
                           // shared (a_fsas_.Dim0() == 1).
  DenseFsaVec &b_fsas_;
  int32_t T_;  // == b_fsas_.MaxSize(1).
  float search_beam_;
  float output_beam_;
  int32_t min_active_;
  int32_t max_active_;
  Array1<float> dynamic_beams_;  // dynamic beams (initially just search_beam_
                                 // but change due to max_active/min_active
                                 // constraints).

  int32_t state_map_fsa_stride_;  // state_map_fsa_stride_ is a_fsas_.TotSize(1)
                                  // if a_fsas_.Dim0() == 1, else 0.

  Hash32 state_map_;  // state_map_ maps from:
                      // key == (state_map_fsa_stride_*n) + a_fsas_state_idx01,
                      //    where n is the fsa_idx, i.e. the index into b_fsas_
                      // to
                      // value, where at different stages of PropagateForward(),
                      // value is an arc_idx012 (into cur_frame->arcs), and
                      // then later a state_idx01 into the next frame's `state`
                      // member.

  // The 1st dim is needed because If all the
  // streams share the same FSA in a_fsas_, we need
  // separate maps for each).  This map is used on
  // each frame to compute and store the mapping
  // from active states to the position in the
  // `states` array.  Between frames, all values
  // have -1 in them.

  std::vector<std::unique_ptr<FrameInfo>> frames_;

};

void IntersectDensePruned(FsaVec &a_fsas, DenseFsaVec &b_fsas,
                          float search_beam, float output_beam,
                          int32_t min_active_states, int32_t max_active_states,
                          FsaVec *out, Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b) {
  NVTX_RANGE("IntersectDensePruned");
  FsaVec a_vec = FsaToFsaVec(a_fsas);
  MultiGraphDenseIntersectPruned intersector(a_vec, b_fsas, search_beam,
                                             output_beam, min_active_states,
                                             max_active_states);

  intersector.Intersect();
  intersector.FormatOutput(out, arc_map_a, arc_map_b);
}
}  // namespace k2
