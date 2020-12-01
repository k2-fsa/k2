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
    int32_t dest_info_state_idx01;  // The destination-state as an index into
    // the next FrameInfo's `arcs` or `states`
    int32_t dest_info_state_idx1;  // The destination-state as an index the
    // next FrameInfo's `arcs` or `states`,
    // this time omitting the FSA-index.
  } u;
  float end_loglike;  // loglike at the end of the arc just before
  // (conceptually) it joins the destination state.
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
                           always succeed.
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
    NVTX_RANGE(__func__);
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
    if (a_fsas.shape.Dim0() == 1) {
      a_fsas_stride_ = 0;
      state_map_ = Array2<int32_t>(c_, num_seqs, a_fsas.shape.TotSize(1), -1);
    } else {
      K2_CHECK_EQ(a_fsas.shape.Dim0(), b_fsas.shape.Dim0());
      a_fsas_stride_ = 1;
      state_map_ = Array2<int32_t>(c_, 1, a_fsas.shape.TotSize(1), -1);
    }
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
    NVTX_RANGE(__func__);
    int32_t T = b_fsas_.shape.MaxSize(1), num_fsas = b_fsas_.shape.Dim0();

    std::ostringstream os;
    os << "Intersect:T=" << T << ",num_fsas=" << num_fsas
       << ",TotSize(1)=" << b_fsas_.shape.TotSize(1);
    NVTX_RANGE(os.str().c_str());

    frames_.reserve(T + 1);

    frames_.push_back(InitialFrameInfo());

    for (int32_t t = 0; t <= T; t++) {
      frames_.push_back(PropagateForward(t, frames_.back().get()));
    }
    // The FrameInfo for time T+1 will have no states.  We did that
    // last PropagateForward so that the 'arcs' member of frames_[T]
    // is set up (it has no arcs but we need the shape).
    frames_.pop_back();

    {
      NVTX_RANGE("InitOshapeUnpruned..");
      // each of these have 3 axes.
      std::vector<RaggedShape *> arcs_shapes(T + 1);
      for (int32_t t = 0; t <= T; t++)
        arcs_shapes[t] = &(frames_[t]->arcs.shape);

      // oshape_unpruned_ is a 4-axis ragged tensor which is indexed:
      //   oshape_unpruned_[fsa_index][t][state_idx][arc_idx]
      // This is BEFORE BACKWARD PRUNING... oshape_pruned_ will
      // be after backward pruning
      int32_t axis = 1;
      oshape_unpruned_ = Stack(axis, T + 1, &(arcs_shapes[0]));
    }
    renumber_output_states_.Init(c_, oshape_unpruned_.TotSize(2));
    renumber_output_arcs_.Init(c_, oshape_unpruned_.TotSize(3));

    for (int32_t t = T; t >= 0; t--) {
      // this writes to elements of renumber_output_states_.Keep() and
      // renumber_output_arcs_.Keep().
      PropagateBackward(t, frames_[t].get(),
                        (t == T ? NULL : frames_[t + 1].get()));
    }
    oshape_pruned_ = SubsampleRaggedShape(
        oshape_unpruned_, renumber_output_states_, renumber_output_arcs_);
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

    K2_EVAL(
        c_, tot_arcs_pruned, lambda_format_arc_data,
        (int32_t pruned_idx0123)->void {
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
          int32_t pruned_src_state_idx012 =
                      old2new_state_map012[unpruned_idx012],
                  pruned_src_state_idx12 =
                      pruned_src_state_idx012 - pruned_dest_state_idx0xx;

          Arc arc;
          // The numbering for the dest-state in the output Arc is the numbering
          // *within the FSA*, and we ignore the time index (1) because that
          // index will be removed as the FSA format has no notion of time;
          // that's why we use the indx12.

          arc_map_a_data[pruned_idx0123] = arc_info.a_fsas_arc_idx012;

          arc.src_state = pruned_src_state_idx12;
          arc.dest_state = pruned_dest_state_idx12;
          arc.label = a_fsas_arcs[arc_info.a_fsas_arc_idx012].label;
          K2_CHECK_LE(static_cast<uint32_t>(arc.label + 1),
                      static_cast<uint32_t>(b_fsas_num_cols))
              << "label out of range";
          int32_t fsa_id = unpruned_idx0,
                  b_fsas_idx0x = b_fsas_row_splits1[fsa_id],
                  b_fsas_idx01 = b_fsas_idx0x + t,
                  b_fsas_idx2 = (arc.label + 1),
                  b_fsas_arc_idx012 =
                      b_fsas_idx01 * b_fsas_num_cols + b_fsas_idx2;
          arc.score = arc_info.arc_loglike;
          arc_map_b_data[pruned_idx0123] = b_fsas_arc_idx012;
          arcs_out_data[pruned_idx0123] = arc;
        });

    // Remove axis 1, which corresponds to time.
    RaggedShape output_fsas_shape = RemoveAxis(oshape_pruned_, 1);
    *ofsa = FsaVec(output_fsas_shape, arcs_out);
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
    'search_beam_' as long as the number of active states in each FSA is between
    min_active and max_active.
  */
  Array1<float> GetPruningCutoffs(Ragged<float> &arc_end_scores) {
    NVTX_RANGE(__func__);
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
            dynamic_beam *= 0.9;
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
    NVTX_RANGE(__func__);
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
    Ragged<ArcInfo> arc_info = GetArcs(t, cur_frame);

    ArcInfo *ai_data = arc_info.values.Data();
    Array1<float> ai_data_array1(c_, arc_info.values.Dim());
    float *ai_data_array1_data = ai_data_array1.Data();
    K2_EVAL(
        c_, ai_data_array1.Dim(), lambda_set_ai_data,
        (int32_t i)->void { ai_data_array1_data[i] = ai_data[i].end_loglike; });
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
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_state_map,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
            int32_t dest_state_idx01 =
                ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            float end_loglike = ai_data[arc_idx012].end_loglike,
                  cutoff = cutoffs_data[fsa_id];
            if (end_loglike > cutoff) {
              // The following is a race condition as multiple threads may write
              // to the same location, but it doesn't matter, the point is to
              // assign one of the indexes.
              state_map_acc(fsa_id, dest_state_idx01) = arc_idx012;
            }
          });
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
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_keep,
          (int32_t arc_idx012)->void {
            int32_t dest_state = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            int32_t j = state_map_data[dest_state];
            if (j != -1) {  // the dest-state was kept..
              // caution: keep_this_state_data is indexed by *arc*
              if (j == arc_idx012)  // this arc 'won' the data race..
                keep_this_state_data[arc_idx012] = 1;
            }
          });
    } else {
      NVTX_RANGE("LambdaSetKeepB");
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_keep,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                    dest_state = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            int j = state_map_acc(fsa_id, dest_state);
            if (j != -1) {  // the dest-state was kept..
              // caution: keep_this_state_data is indexed by *arc*
              if (j == arc_idx012)  // this arc 'won' the data race..
                keep_this_state_data[arc_idx012] = 1;
            }
          });
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
    cur_frame->arcs = arc_info;

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
            int32_t dest_state_idx =
                ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
            int32_t this_j = state_reorder_data[arc_idx012],
                    next_j = state_reorder_data[arc_idx012 + 1];
            if (next_j > this_j) {
              K2_CHECK_EQ(state_map_acc(fsa_id, dest_state_idx), arc_idx012);
              // Note: this_j is an idx01 into ans->states.  previously
              // state_map_data cotained an arc_idx012 (of the entering arc that
              // won the race).
              state_map_acc(fsa_id, dest_state_idx) = this_j;
            }
          });
    }

    // We'll set up the data of the kept states below...
    StateInfo *kept_states_data = ans->states.values.Data();

    {
      NVTX_RANGE("LambdaSetArcsAndStates");
      K2_EVAL(
          c_, arc_info.NumElements(), lambda_set_arcs_and_states,
          (int32_t arc_idx012)->void {
            int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];

            ArcInfo &info = ai_data[arc_idx012];

            int32_t dest_a_fsas_state_idx01 = info.u.dest_a_fsas_state_idx01;
            // state_idx01 is the index into ans->states, of the destination
            // state. Note: multiple arcs may enter this state, which is why we
            // had to set that in a separate kernel (lambda_modify_state_map).
            int32_t state_idx01 =
                state_map_acc(fsa_id, dest_a_fsas_state_idx01);
            info.u.dest_info_state_idx01 = state_idx01;

            // state_idx01 will be -1 for states that were pruned out because
            // they were outside the beam.
            if (state_idx01 < 0) return;
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
            K2_CHECK_EQ(state_map_acc(fsa_idx0, a_fsas_state_idx01),
                        state_idx01);
            // We're resetting state_map to its original clean condition.
            state_map_acc(fsa_idx0, a_fsas_state_idx01) = -1;
          });
    }

    return ans;
  }

  /*
    Does backward propagation of log-likes, which means setting the
    backward_loglike field of the StateInfo variable.  These backward log-likes
    are normalized in such a way that you can add them with the forward
    log-likes to produce the log-likelihood ratio vs the best path (this will be
    non-positive).  To do this, for the final state we have to set the backward
    log-like to the negative of the forward log-like.

       @param [in] t       The time-index (on which to look up log-likes),
                           t >= 0
       @param [in]  cur_frame    The FrameInfo for the frame on which we want to
                                 set the forward log-like
       @param [in]  next_frame  NULL if this is is the last frame of the
                                sequence; otherwise the next frame's FrameInfo;
                                arcs on `cur_frame` have transitions to states
                                on `next_frame`. The `backward_loglike` values
                                in `next_frame` are assumed to already be set.
   */
  void PropagateBackward(int32_t t, FrameInfo *cur_frame,
                         FrameInfo *next_frame) {
    NVTX_RANGE("PropagateBackward");
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
    float output_beam = output_beam_;

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
      K2_EVAL(
          c_, arc_backward_prob.NumElements(),
          lambda_set_arc_backward_prob_and_keep, (int32_t arcs_idx012)->void {
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
              keep_this_arc = (backward_loglike + src_state_forward_loglike >=
                               -output_beam);
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
          });
    } else {
      K2_DCHECK_EQ(arc_backward_prob.NumElements(), 0)
          << "Caution: final frame has arcs; check that there were -infinities "
             "in the right place on the last frame of the 'scores' matrix.";
    }

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
                  a_fsas_idx0 = a_fsas_row_ids1[fsas_state_idx01],
                  states_idx0 = cur_states_row_ids1[state_idx01],
                  fsas_state_idx0x_next = a_fsas_row_splits1[a_fsas_idx0 + 1];
          // Note: a_fsas_idx0 and states_idx0 will be the same if
          // a_fsas_.Dim0() >= b_fsas_.Dim0().
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
          // note: axis 1 of 'states' corresponds to axis 2 of 'oshape'; it's
          // the state index.  Also,

          keep_states_data[oshape_idx012] = keep_this_state;
          if (!keep_this_state) {
            // The reason we set the backward_loglike to -infinity here if it's
            // outside the beam, is to prevent disconnected states from
            // appearing after pruning due to numerical roundoff effects near
            // the boundary at `-beam`.  It would otherwise be correct and
            // harmless to omit this if-block.
            backward_loglike = minus_inf;
          }
          info->backward_loglike = backward_loglike;
        });
  }

  ContextPtr c_;
  FsaVec &a_fsas_;         // Note: a_fsas_ has 3 axes.
  int32_t a_fsas_stride_;  // 1 if we use a different FSA per sequence
                           // (a_fsas_.Dim0() > 1), 0 if the decoding graph is
                           // shared (a_fsas_.Dim0() == 1).
  DenseFsaVec &b_fsas_;
  float search_beam_;
  float output_beam_;
  int32_t min_active_;
  int32_t max_active_;
  Array1<float> dynamic_beams_;  // dynamic beams (initially just search_beam_
                                 // but change due to max_active/min_active
                                 // constraints).
  Array2<int32_t> state_map_;    // state_map_ is of shape
                                 // (n, a_fsas_.TotSize(1)) where n is
  // 1 if a_fsas_.Dim0() > 1, else b_fsas_.Dim0().

  // The 1st dim is needed because If all the
  // streams share the same FSA in a_fsas_, we need
  // separate maps for each).  This map is used on
  // each frame to compute and store the mapping
  // from active states to the position in the
  // `states` array.  Between frames, all values
  // have -1 in them.

  std::vector<std::unique_ptr<FrameInfo>> frames_;

  // This is a rearranged version of the info in 'frames', computed at the end
  // of the forward pass before pruning.  It is indexed [fsa_id][t][state][arc].
  RaggedShape oshape_unpruned_;

  // these two Renumbering objects dictate how we renumber oshape_unpruned_,
  // i.e. which states and arcs we delete.  The data in their Keep() members,
  // which are vectors of chars, are written to in PropagateBackward().
  Renumbering renumber_output_states_;
  Renumbering renumber_output_arcs_;

  // This is as oshape_unpruned_, but after the backward-pass pruning.
  // It is indexed [fsa_id][t][state][arc].
  RaggedShape oshape_pruned_;
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
