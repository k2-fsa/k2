// k2/csrc/cuda/compose.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/compose.h"

namespace k2 {

// Caution: this is really a .cu file.  It contains mixed host and device code.



// Note: b is FsaVec<Arc>.
void Intersect(const DenseFsa &a, const FsaVec &b, Fsa *c,
               Array<int32_t> *arc_map_a = nullptr,
               Array<int32_t> *arc_map_b = nullptr) {

}



/*
   Pruned intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  Can use either different decoding graphs (one per
   acoustic sequence) or a shared graph
*/
class MultiGraphDenseIntersect {
 public:
  /**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might just be a linear
                      sequence of phones, or might be something more complicated.  Must
                      have either the same Size0() as b_fsas, or Size0()==1 in which
                      case the graph is shared.
       @param [in] b_fsas  The neural-net output, with each frame containing the log-likes of
                      each phone.  A series of sequences of (in general) different length.
       @param [in] beam    "Default" decoding beam.  The actual beam is dynamic and also depends
                      on max_active and min_active.
       @param [in] max_active  Maximum number of FSA states that are allowed to be active on any given
                     frame for any given intersection/composition task.  This is advisory,
                     in that it will try not to exceed that but may not always succeed.
       @param [in] min_active  Minimum number of FSA states that are allowed to be active on any given
                     frame for any given intersection/composition task.  This is advisory,
                     in that it will try not to have fewer than this number active.
   */
  MultiGraphDenseIntersect(FsaVec &a_fsas,
                           DenseFsaVec &b_fsas,
                           float beam,
                           int32_t max_active,
                           int32_t min_active):
      a_fsas_(a_fsas), b_fsas_(b_fsas), beam_(beam),
      max_active_(max_active), min_active_(min_active),
      dynamic_beams_(a_fsas.Dim0(), beam),
      state_map_(a_fsas.TotSize1(), -1) {
    c_ = GetContext(a_fsas, b_fsas);
    CHECK_GT(beam, 0);
    CHECK_GT(min_active, 0);
    CHECK_GT(max_active, min_active);
    assert(a_fsas.Size0() == b_fsas.Size0() || a_fsas.Size0() == 1);
    CHECK_GT(b_fsas.Size0(), 1);
    a_fsas_stride_ == (a_fsas.Size0() == b_fsas.Size0() ? 1 : 0);
    Intersect();
  }


  /* Does the main work of intersection/composition, but doesn't produce any output;
     the output is provided when you call FormatOutput(). */
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
    int32_t T = b_fsas_.MaxSize1(),
        num_fsas = b_fsas_.Size0();

    frames_.reserve(T+1);

    frames_.push_back(InitialFrameInfo());

    for (int32_t t = 0; t < T; t++) {
      frames.push_back(PropagateForward(t, frames.back()));
    }


    {
      std::vector<RaggedShape3*> arcs_shapes(T+1);
      for (int32_t t = 0; t <= T; t++)
        arcs_shapes[t] = &(frames_[t].arcs.shape);

      // 'output_arc_sizes' is a ragged tensor which is indexed:
      //   output_arc_idx[fsa_index][t][state_idx][arc_idx]
      // where the 'state_idx' and 'arc_idx'
      // This is BEFORE PRUNING...
      oshape_unpruned_ = MergeToAxis1(arcs_shapes);
    }
    renumber_output_states_.Init(oshape_unpruned_.TotSize2());
    renumber_output_arcs_.Init(oshape_unpruned_.TotSize3());

    for (int32_t t = T; t >= 0; t--) {
      // this writes to elements of renumber_output_states_.Keep() and
      // renumber_output_arcs_.Keep().
      PropagateBackward(frames_[t], (t == T ? NULL : frames_[t+1] ));
    }
    oshape_pruned_ = RaggedShape4Subsampled(
        oshape_unpruned_, NULL, NULL,
        &renumber_output_states_, &renumber_output_arcs_);
  }

  FrameInfo *InitialFrameInfo() {
    // TODO
  }

  void FormatOutput(FsaVec *ofsa,
                    Array<int32_t> *arc_map_a,
                    Array<int32_t> *arc_map_b) {

    Context c_cpu = c_->CpuContext();
    int32_t T = a_fsas.MaxSize1();

    int32_t *oshapeu_row_ids3 = oshape_unpruned_.RowIds3(),
        *oshapeu_row_ids2 = oshape_unpruned_.RowIds2(),
        *oshapeu_row_ids1 = oshape_unpruned_.RowIds1(),
        *oshapeu_row_splits3 = oshape_unpruned_.RowSplits3(),
        *oshapeu_row_splits2 = oshape_unpruned_.RowSplits2(),
        *oshapeu_row_splits1 = oshape_unpruned_.RowSplits1();

    int32_t *oshapep_row_ids3 = oshape_pruned_.RowIds3(),
        *oshapep_row_ids2 = oshape_pruned_.RowIds2(),
        *oshapep_row_ids1 = oshape_pruned_.RowIds1(),
        *oshapep_row_splits3 = oshape_pruned_.RowSplits3(),
        *oshapep_row_splits2 = oshape_pruned_.RowSplits2(),
        *oshapep_row_splits1 = oshape_pruned_.RowSplits1();

    // the 0123 and 012 express what type of indexes they are, see comment at
    // top of utils.h
    int32_t *reverse_arc_map0123 = renumbered_output_states_.New2Old().Data(),
        *reverse_state_map012 = renumbered_output_states_.New2Old().Data();

    Array1<Arc*> arcs_data_ptrs(c_cpu, T+1);
    Array1<int32_t*> arcs_row_splits1_ptrs(c_cpu, T+1);
    Array1<int32_t*> arcs_row_splits2_ptrs(c_cpu, T+1);

    int32_t **arcs_row_splits1_ptrs_data = arcs_row_splits1_ptrs.Data();
    for (int32_t t = 0; t <= T; t++) {
      arcs_data_ptrs.Data()[t] = frames_[t]->arcs.values.Data();
      arcs_row_splits1_ptrs.Data()[t] = frames_[t]->arcs.shape.RowSplits1();
      arcs_row_splits2_ptrs.Data()[t] = frames_[t]->arcs.shape.RowSplits2();
    }
    // transfer to GPU if we're using a GPU
    arcs_data_ptrs = arcs_data_ptrs.To(c_);
    arcs_row_splits1_ptrs = arcs_row_splits1_ptrs.To(c_);
    arcs_row_splits2_ptrs = arcs_row_splits2_ptrs.To(c_);
    const ArcInfo **arcs_data_ptrs_data = arcs_data_ptrs.Data();
    const int32_t **arcs_row_splits1_ptrs_data = arcs_row_splits1_ptrs.Data(),
        **arcs_row_splits2_ptrs_data = arcs_row_splits2_ptrs.Data();


    int32_t tot_arcs_pruned = oshape_pruned_.TotSize3();
    arc_map_a = Array<int32_t>(c_, tot_arcs_pruned);
    arc_map_b = Array<int32_t>(c_, tot_arcs_pruned);
    int32_t *arc_map_a_data = arc_map_a.Data(),
        *arc_map_b_data = arc_map_b.Data();
    Array<Arc> arcs_out(c_, tot_arcs_pruned);
    Arc *arcs_out_data = arcs.Data();
    const Arc *a_fsas_arcs = a_fsas_.values.Data();
    int32_t b_fsas_num_cols = b_fsas_.scores.Dim1();
    const int32_t *b_fsas_row_ids1 = b_fsas.shape.RowIds1().Data();

    auto lambda_format_arc_data = __host__ __device__ [=] (int32_t pruned_ind0123) -> void {
         int32_t unpruned_ind0123 = reverse_state_map0123[pruned_ind0123];
         int32_t unpruned_ind012 = oshapeu_row_ids3[unpruned_ind0123],
             unpruned_ind01 = oshapeu_row_ids2[unpruned_ind012],
             unpruned_ind01x = oshapeu_row_splits2[unpruned_ind01],
             unpruned_ind01xx = oshapeu_row_splits3[unpruned_ind01x],
             unpruned_indxx23 = unpruned_ind0123 - unpruned_ind01xx,
             unpruned_ind0 = oshapeu_row_ids1[unpruned_ind01], // fsa-id
             unpruned_ind0x = oshapeu_row_splits1[unpruned_ind0],
             unpruned_ind0xx = oshapeu_row_splits2[unpruned_ind0x],
             unpruned_ind1 = unpruned_ind01 - unpruned_ind01, // t
             unpruned_ind01_next_t = unpruned_ind01 + 1,
             unpruned_ind01x_next_t = oshapeu_row_splits2[unpruned_ind01_next_t];

         int32_t *arcs_row_splits1_data = arcs_row_splits1_ptrs_data[t],
             *arcs_row_splits2_data = arcs_row_splits2_ptrs_data[t],
             arcs_ind0x = arcs_row_splits1_data[unpruned_ind0],
             arcs_ind0xx = arcs_row_splits2_data[arcs_ind0x];
         // below: axes 2,3 of the unpruned layout coincide with axes 1,2 of
         // 'arcs'; these are state and arc indexes (within this frame
         // of this FSA).
         int32_t arcs_ind012 = arcs_ind0xx + unpruned_indxx23;
         ArcInfo *arcs_data = arcs_data_ptrs_data[t];
         ArcInfo arc_info = arcs_data[arcs_ind012];

         // we call it ind2 because the state-index is axis 2 of oshape.
         int32_t unpruned_dest_state_ind2 = arc_info.dest_state_ind1,
             unpruned_dest_state_ind012 = unpruned_ind01x_next_t + unpruned_dest_state_ind2,
             pruned_dest_state_ind012 = reverse_state_map012[unpruned_dest_state_ind012],
             pruned_dest_state_ind01 = oshapep_row_ids2[pruned_dest_state_ind012],
             pruned_dest_state_ind0 = oshapep_row_ids1[pruned_dest_state_ind01],
             pruned_dest_state_ind0x = oshapep_row_splits1[pruned_dest_state_ind0],
             pruned_dest_state_ind0xx = oshapep_row_splits2[pruned_dest_state_ind0x],
             pruned_dest_state_indx12 = pruned_dest_state_ind012 - pruned_dest_state_ind0xx;

         // note: the src-state and dest-state have the same ind0 which is the FSA-id.
         int32_t pruned_src_state_ind012 = reverse_state_map012[unpruned_ind012],
             pruned_src_state_indx12 = pruned_src_state_ind012 - pruned_dest_state_ind0xx;

         Arc arc;
         // The numbering for the dest-state in the output Arc is the numbering *within the FSA*,
         // and we ignore the time index (1) because that index will be removed as the FSA format
         // has no notion of time; that's why we use the indx12.

         arc_map_a_data[pruned_ind0123] = arc_info.arc_idx;

         arc.src_state = pruned_src_state_indx12;
         arc.dest_state = pruned_dest_state_indx12;
         arc.symbol = a_fsas_arc[arc_info.arc_idx].symbol;
         int32_t fsa_id = unpruned_ind0, t = unpruned_ind1,
             b_fsas_ind0x = b_fsas_row_ids1[fsa_id],
             b_fsas_ind01 = b_fsas_ind0x + t,
             b_fsas_indxx2 = (arc.symbol + 1),
             b_fsas_arc_ind012 =  b_fsas_ind01 * b_fsas_num_cols + b_fsas_indxx2;
         arc.score = arc_info.arc_loglike;

         arc_map_b_data[pruned_ind0123] = b_fsas_arc_ind012;
         arcs_out_data[pruned_ind0123] = arc;
    };
    Eval(c_, tot_arcs_pruned, lambda_format_arc_data);


    // The output shape will get rid of axis one of oshape_pruned_ (which is the 't' index).
    Array1<int32_t> arcs_row_splits1_out = oshape_pruned_.RowSplits2()[oshape_pruned_.RowSplits1()],
        arcs_row_ids2_out = oshape_pruned_.RowIds1()[oshape_pruned_.RowIds2()];


    Eval(c_, oshape_pruned_.RowSplits1().Dim(), lambda_set_row_splits1_out);
    RaggedShape3 output_fsas_shape(arcs_row_splits1_out,
                                   oshape_pruned_.RowSplits3(),
                                   oshape_pruned_.TotSize3(),
                                   arcs_row_ids2_out,
                                   oshape_pruned_.RowIds3());

    *ofsa = FsaVec(output_fsas_shape, arcs_out);
  }


  /*
    Compute pruning cutoffs for this frame: these are the cutoffs for the arc
    "forward score", one per FSA.  This is a dynamic process involving
    dynamic_beams_ which are updated on each frame (they start off at beam_).

       @param [in] arc_end_probs  The "forward log-probs" (scores) at the
                   end of each arc, i.e. its contribution to the following
                   state.
       @param [out] cutoffs   Outputs the cutoffs, one per FSA (will be -infinity
                    for FSAs that don't have any active states).  These will
                    be of the form: the best score for any arc, minus the
                    dynamic beam.  Note: `cutoffs` does not have to be sized
                    correctly at entry, it will be overwritten.
   */
  void ComputePruningCutoffs(const Ragged3<float> &arc_end_scores,
                             Array1<float> *cutoffs) {
    int32 num_fsas = arc_end_scores.Dim0();

    // get the maximum score from each sub-list (i.e. each FSA, on this frame).
    // Note: can probably do this with a cub Reduce operation using an operator
    // that has side effects (that notices when it's operating across a
    // boundary).
    // the max will be -infinity for any FSA-id that doesn't have any active
    // states (e.g. because that stream has finished).
    // Casting to ragged2 just considers the top 2 indexes, ignoring the 3rd.
    // i.e. it's indexed by [fsa_id][state].
    Array1<float> max_per_fsa = MaxPerSubSublist((Ragged2<float>&)end_probs);

    Array1<int32_t> active_states_per_fsa =  end_probs.Sizes1();
    int32_t *active_per_fsa_data = active_states_per_fsa.values.data();
    float *max_per_fsa_data = max_per_fsa.values.data(),
        *dynamic_beams_data = dynamic_beams_.values.data();
    float beam = beam_,
        max_active = max_active_,
        min_active = min_active_;

    auto lambda = __host__ __device__ [=] (int32_t i) -> float {
              float best_loglike = max_per_fsa_data[i],
                  dynamic_beam = dynamic_beams_data[i];
              int32_t num_active = active_per_fsa_data[i];
              float ans;
              if (num_active <= max_active) {
                if (num_active > min_active) {
                  // Neither the max_active nor min_active constraints
                  // apply.  Gradually approach 'beam'
                  ans = 0.8 * dynamic_beam + 0.2 * beam;
                } else {
                  // We violated the min_active constraint -> increase beam
                  if (ans < beam) ans = beam;
                  // gradually make the beam larger as long
                  // as we are below min_active
                  ans *= 1.25;
                }
              } else {
                // We violated the max_active constraint -> decrease beam
                if (ans > beam) ans = beam;
                // Decrease the beam as long as we have more than
                // max_active active states.
                ans *= 0.9;
              }
              return ans;
    };
    Array1<float> new_beam(c_, num_fsas, lambda);
    dynamic_beam_.swap(&new_beam);

    float *dynamic_beam_data = dynamic_beam_.data;
    auto lambda2 = __device__ [dynamic_beam_data,max_per_fsa_data] (int32_t i)->float {
                                return max_per_fsa_data[i] - dynamic_beam_data[i]; };
    *cutoffs = Array1<float>(c_, num_fsas, lambda2);
  }


  /*
    Returns un-pruned list of arcs on this frame, consisting of all arcs leaving
    the states active on 'cur_frame'.  (Note: this is the 1st pass of pruning,
    the forward pass).

       @param [in] t       The time-index (on which to look up log-likes),
                           t >= 0
       @param [in] cur_frame   The FrameInfo for the current frame; only its
                     'states' member is expected to be set up on entry.
   */
  Array1<Arc> GetUnprunedArcs(int32_t t, FrameInfo *cur_frame) {
    Ragged2<StateInfo> &states = cur_frame->states;
    const StateInfo *state_values = states.values.data;
    // in a_fsas_ (the decoding graphs), maps from state_ind01 to arc_ind01x.
    const int32_t *fsa_arc_splits = a_fsas_.RowSplits2().data();
    int32_t a_fsas_stride = a_fsas_stride_;

    // frame_state_ind01 combines the FSA-index and state-index (into 'cur_frame->states')
    __host__ __device__ auto num_arcs_lambda = [=] (int32_t state_ind01) -> int32_t {
          int32_t a_fsas_state_ind01 = state_values[i].a_fsas_state_ind01,
                      a_fsas_arc_ind01x = fsa_arc_splits[a_fsas_state_ind01],
                      a_fsas_arc_ind01x_next = fsa_arc_splits[a_fsas_state_ind01+1],
                      a_fsas_num_arcs_x1x = fsa_arc_ind01x - fsa_arc_ind01x_next;
                   return num_arcs_x1x;
              };
    // `num_arcs` gives the num-arcs for each state in `states`.
    Array<int32_t> num_arcs(c_, states.values.size(), num_arcs_lambda);

    // initialize shape of array that will hold arcs leaving the active states.
    // Its shape is [fsa_index][state][arc]; the top two levels are shared with
    // `states`.  'ai' means ArcInfo.
    RaggedShape3 ai_shape = Ragged3FromSizes(states.shape, num_arcs);

    // 'ai' means ArcInfo
    const int32_t *ai_row_ids1 = ai_shape.RowIds1(),   // from state_ind01 (into `states` or `ai_shape`) -> fsa_ind0
        *ai_row_ids2 = ai_shape.RowIds2(),   // from arc_ind012 (into `ai_shape`) to state_ind01
        *ai_row_splits2 = ai_shape.RowSplits2(),  // from state_ind01 to arc_ind01x
        *a_fsas_row_splits2 = a_fsas_.RowSplits2();  // from state_ind01 (into
                                                     // a_fsas_) to arc_ind01x
                                                     // (into a_fsas_)

    const Arc *arcs = a_fsas_.values.Data();
    // fsa_ind0 to ind0x (into b_fsas_), which gives the 1st row for this
    // sequence.
    const int32_t *b_fsas_row_ids1 = b_fsas_.shape.RowIds1();
    const float *score_data = b_fsas_.scores.Data();
    int score_num_cols = b_fsas_.scores.Dim1();

    Ragged3<ArcInfo> ai(ai_shape);
    ArcInfo *ai_data = ai.values.Data();  // uninitialized

    __host__ __device__ auto ai_lambda = [=] (int32_t ai_arc_ind012) -> void {
          int32_t ai_state_ind01 = ai_row_ids2[ai_arc_ind012],
              ai_fsa_ind0 = ai_row_ids1[ai_state_ind01],
              ai_arc_ind01x = ai_row_splits2[ai_state_ind01],
              ai_arc_indxx2 = ai_arc_ind012 - ai_arc_ind01x;
          StateInfo sinfo = state_values[ai_row_id];
          int32_t a_fsas_arc_ind01x = a_fsas_row_splits2[sinfo.a_fsas_state_ind01],
              a_fsas_arc_ind012 = a_fsas_arc_ind01x + ai_arc_indxx2;
          Arc arc = arcs[a_fsas_arc_ind012];

          int32_t scores_ind0x = b_fsas_row_ids1[ai_fsa_ind0],
              scores_ind01 = scores_ind0x + t, // t == ind1 into 'scores'
              scores_ind2 = arc.symbol + 1,  // the +1 is so that -1 can be handled
              scores_ind012 = (scores_ind01 * scores_num_cols) + scores_ind2;
          assert(static_cast<uint32_t>(scores_ind2) < static_cast<uint32_t>(scores_num_cols));
          float acoustic_score = score_data[scores_ind012];
          ArcInfo ai;
          ai.a_fsas_arc_ind012 = a_fsas_arc_ind012;
          ai.arc_loglike = acoustic_score + arc.score;
          ai.end_loglike = sinfo.forward_loglike + ai.arc_loglike;
          // at least currently, the ArcInfo object's src_state and dest_state are
          // ind1's not ind01's, i.e. they don't contain the FSA-index, where as
          // the ai element is an ind01, so we need to do this to conver to an ind01;
          // this relies on the fact that sinfo.abs_state_id == arc.src_state + a_fsas_fsa_ind0x.
          ai.u.dest_a_fsas_state_ind01 = sinfo.a_fsas_state_ind01 + arc.dest_state - arc.src_state;
          ai_data[i] = ai;
        };
    Eval(c_, ai.values.size(), ai_lambda);
    return ai;
  }


  /*
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* object for the next frame.
   */
  FrameInfo* PropagateForward(int t, FrameInfo *cur_frame) {

    Ragged2<StateInfo> &states = cur_frame->states;
    Array3<ArcInfo> ai = GetUnprunedArcs(t, cur_frame);

    ArcInfo *ai_data = arc_info.elems.data();

    Ragged3<float> ai_loglikes(arc_info.shape,
                               Array1<float>(arc_info.elems.size(),
                                             [ai_data](int32_t i) -> float { return ai_data[i].end_loglike; }));


    Array1<float> cutoffs; // per fsa.
    ComputePruningCutoffs(&ai_loglikes, &cutoffs);

    float *cutoffs_data = cutoffs.data();


    // write certain indexes i (indexes into arc_info.elems) to state_map_.data().
    // Keeps track of the active states and will allow us to assign a numbering to them.
    int *ai_row_ids1 = arc_info.RowIds1().data(),
        *ai_row_ids2 = arc_info.RowIds2().data(),
        *state_map_data = state_map_.data();
    auto lambda3 == __host__ __device__ [ai_data,ai_row_ids1,ai_row_ids2,cutoffs_data](int32_t i) -> void {
                                 int32_t fsa_id = ai_row_ids1[ai_row_ids2[i]];
                                 int32_t abs_dest_state = ai_data[i].abs_dest_state;
                                 float end_loglike = ai_data[i].end_loglike,
                                     cutoff = cutoffs_data[fsa_id];
                                 if (end_loglikes > cutoff) {
                                   // The following is a race condition as multiple threads may write to
                                   // the same location, but it doesn't matter, the point32_t is to assign
                                   // one index i
                                   state_map_data[abs_dest_state] = i;
                                 }
                               };
    Eval(D, arc_info.elems.size(), lambda3);


    // 1 if we keep this arc, else 0.  The unusual way we initialize this is
    // because we need the memory region to cover one extra element, due to how
    // ExclusiveSum() works.
    Array1<char> keep_this_arc = Array1<char>(c_, arc_info.elems.size()+1,
                                              0).Range(0, arc_info.elems.size());

    // This is like `keep_this_arc`, but but only *one* of the arcs leading to
    // any given state has this nonzero (that one becomes the representative
    // that determines the order in which we'll list the destination states).
    Array1<char> keep_this_state = Array1<char>(c_, arc_info.elems.size()+1,
                                                0).Range(0, arc_info.elems.size());

    // Note: we don't just keep arcs that were above the pruning threshold, we
    // keep all arcs whose destination-states survived pruning.  Later we'll
    // prune with the lattice beam, using both forward and backward scores.
    char *keep_this_arc_data = keep_this_arc.data(),
        *keep_this_state_data = keep_this_state.data();
    auto lambda_keep = __host__ __device__ [=](int32_t i) -> void {
                                         int32_t abs_dest_state = ai_data[i].abs_dest_state;
                                         int j = state_map_data[abs_dest_state];
                                         if (j != -1) {
                                           keep_this_arc_data[j] = 1;
                                           if (j == i)
                                             keep_this_state_data[i] = 1;
                                         }
                                       };
    Eval(D, arc_info.elems.size(), lambda_keep);

    // The '+1' is so we can easily get the total num-arcs and num-states.
    Array1<int32_t> arc_reorder(c_, arc_info.elems.size() + 1),
        state_reorder(c_, arc_info.elems.size() + 1);
    ExclusiveSum(keep_this_arc, &arc_reorder);
    ExclusiveSum(keep_this_state, &state_reorder);

    // OK, 'arc_reorder' and 'state_reorder' contain indexes for where we put
    // each kept arc and each kept state.  They map from old index to new index.
    int num_arcs = arc_reorder[arc_reorder.size()-1],
        num_states = state_reorder[state_reorder.size()-1];

    int *arc_reorder_data = arc_reorder.data(),
        *state_reorder_data = state_reorder.data();

    // state_to_fsa_id maps from an index into the next frame's
    // FrameInfo::states.values() vector (i.e. a frame-state-index on the *next*
    // frame) to the FSA-id associated with it.  It should be non-decreasing.
    Array1<int32_t> state_to_fsa_id(c_, num_states);
    { // This block sets 'state_to_fsa_id'.
      int *states_row_ids = states.RowIds1();  // maps from index into `states`,
      // == current frame's
      // frame-state-index to FSA-id.

      int *abs_state_to_fsa_index = a_fsas_.RowIds1().data();
      int *state_to_fsa_id_data = state_to_fsa_id.data();
      auto lambda_stateid = __host__ __device__ [=]->void {
                                                  int this_state_j = state_reorder_data[i],
                                                      next_state_j = state_reorder_data[i+1];
                                                  if (next_state_j > this_state_j) {
                                                    int abs_dest_state = ai_data[i].u.abs_dest_state,
                                                        fsa_id = abs_state_to_fsa_index[abs_dest_state];
                                                    state_to_fsa_id_data[this_state_j] = fsa_id;
                                                  }
                                                };
      Eval(c_, arc_info.elems.size(), lambda_stateid);
      assert(IsMonotonic(state_to_fsa_id));
    }
    // The following creates a structure that contains a subset of the elements
    // of `arc_info`, determined by `keep_this_arc`.  We already computed the
    // exclusive sum so we use that rather than using `keep_this_arc` directly.
    cur_frame->arcs = Ragged3<Arc>(RaggedShape3SubsampledFromNumbering(arc_info.shape,
                                                                       arc_reorder),
                                   Array1<Arc>(c_, num_arcs));

    FrameInfo *ans = new FrameInfo();
    ans->states = Ragged2FromRowIds(num_fsas,
                                    state_to_fsa_id,
                                    Array1<ArcInfo>(D, num_arcs));
    auto lambda_init_loglike = __host__ __device__ [] (int32_t i) -> void {
                           ans->states[i].forward_loglike = FloatToOrderedInt(-std::numeric_limits<BaseFloat>::infinity());
                                                   };
    Eval(c_, num_states, lambda_init_loglike);


    // we'll set up the data of the kept arcs below..
    ArcInfo *kept_ai_data = cur_frame->arcs.values.data();
    StateInfo *kept_states_data = ans->states.data();

    // Modify the elements of `state_map` to refer to the indexes into
    // `ans->states` / `kept_states_data`, rather than the indexes into ai_data.
    // Note: this will decrease some of the values in `state_map`, in general.
    auto lambda_modify_state_map = __host__ __device__ [=](int32_t i) ->void {
               int this_j = state_reorder_data[i],
                   next_j = state_reorder_data[i+1];
               if (next_j > this_j)
                 state_map[ai_data[i].abs_dest_state] = this_j;
            };
    Eval(c_, arc_info.elems.size(), lambda_modify_state_map);

    auto lambda_set_arcs_and_states = __host__ __device__ [=](int32_t i) ->void {
            int this_j = arc_reorder_data[i],
                next_j = arc_reorder_data[i+1];
            // Note: I have a idea to reduce main-memory bandwidth by
            // caching writes in a fixed-size array in shared memory
            // (store: best-prob, index).  We'd go round robin, try e.g.
            // twice.
            // Would have to do this with a functor rather than a lambda,
            // as __shared__ won't work on CPU.

            //
            if (next_j > this_j) {
              // implies this was one of the arcs to keep.
              ArcInfo info = ai_data[i];
              int abs_dest_state = info.u.abs_dest_state;
              // set the dest_frame_state_idx in
              int dest_frame_state_idx = state_map[abs_dest_state];
              info.u.dest_frame_state_idx = dest_frame_state_idx;
              kept_ai_data[this_j] = info;
              // Note: multiple threads may write the same thing, on the next line.
              kept_states_data[dest_frame_state_idx].abs_state_id = abs_dest_state;
              int32_t end_loglike_int = FloatToOrderedInt(info.end_loglike);
              // Set the forward log-like of the dest state to the largest of any of the
              // incoming arcs.
              atomicMax(&(kept_states_data[dest_frame_state_idx].forward_loglike),
                        end_loglike_int);
            }
        };
    Eval(c_, arc_info.elems.size(), lambda_set_arcs_and_states);
    return ans;
  }
  /*
    Does backward propagation of log-likes, which means setting the backward_loglike
    field of the StateInfo variable.  These backward log-likes are normalized in such
    a way that you can add them with the forward log-likes and get the log-like ratio
    to the best path.  So for the final state we set the backward log-like to
    the negative of the forward loglike.
       @param [in]  cur_frame    The FrameInfo for the frame on which we want to
                                 set the forward log-like
       @param [in]  next_frame  NULL if this is is the last frame of the sequence;
                                otherwise the next frame's FrameInfo; arcs on
                                `cur_frame` have transitions to states on `next_frame`.
                                The `backward_loglike` values in `next_frame` are
                                assumed to already be set.
   */
  void PropagateBackward(FrameInfo *cur_frame,
                         FrameInfo *next_frame) {

    int32_t num_states = cur_frame->states.values.size(),
        num_arcs = cur_frame->arcs.values.size();
    Ragged2<StateInfo> &cur_states = cur_frame->states;
    StateInfo *cur_states_data = cur_states.values.data();

    int32_t tot_states = a_fsas_.TotSize1();
    int32_t *a_fsas_row_ids1 = a_fsas_.RowIds1(),
        *a_fsas_row_splits1 = a_fsas_.RowSplits1();

    const int32_t minus_inf = FloatToOrderedInt(-std::numeric_limits<BaseFloat>::infinity());

    /* arc_backward_probs represents the backward-prob at the beginning of the
       arc.  Indexing is [frame_state_index][arc_index], where frame_state_index
       and arc_index are respectively ind01 and ind2 w.r.t. frames_[t]->arcs. */

    Ragged2<int32_t> arc_backward_prob(AppendAxis(cur_frame->arcs.shape, 0),
                                       Array<int32_t>(c_, num_arcs));
    int32_t *arc_backward_prob_data = arc_backward_prob.values.data();

    ArcInfo *arcs_data = cur_frame->arcs.values.data();
    int32_t *arcs_rowids1 = cur_frame->arcs.RowIds2().data(),
        *arcs_rowids2 = cur_frame->arcs.RowIds2().data(),
        *arcs_row_splits1 = cur_frame->arcs.RowSplits1().data(),
        *arcs_row_splits2 = cur_frame->arcs.RowSplits2().data(),

    float beam = beam_;

    int32_t *oshape_row_splits1 = oshape_unpruned_.RowSplits1(),
        *oshape_row_splits2 = oshape_unpruned_.RowSplits2(),
        *oshape_row_splits3 = oshape_unpruned_.RowSplits3();

    // these have the "output" formatting where we number things with
    // oshape_unpruned_, which is indexed [fsa][t][state][arc].
    char *keep_arcs_data = renumber_output_arcs_.Keep().Data(),
        *keep_states_data = renumber_output_states_.Keep().Data();

    if (next_frame != NULL) {
      // compute arc backward probs, and set elements of 'keep_arcs'a
      StateInfo *cur_states_data = cur_frame->states.values.data();

      // arc_row_ids maps from arc-idx to frame-state-idx, i.e. ind012 into
      // `arcs` to ind01 into `arcs`.

      // next_states_row_splits1 maps from fsa_ind0 to state_ind01
      int32_t *next_states_row_splits1 = next_frame->states.RowSplits1().Data();

      StateInfo *next_states_data = next_frame->states.values.data();
      auto lambda_set_arc_backward_prob_and_keep = __host__ __device__ [=] (int32_t arcs_ind012) -> void {
          ArcInfo *arc = arcs_data + arcs_ind012;
          int32_t state_ind01 = arcs_rowids2[arc_ind012],
              fsa_ind0 = arc_rowids1[state_ind01],
              fsa_ind0x = arc_row_splits1[fsa_ind0],
              fsa_ind0xx = arc_row_splits2[fsa_ind0x],
              arcs_indx12 = arcs_ind012 - fsa_ind0xx;

          int32_t dest_state_ind01 =  arc->u.dest_state_ind01,
              next_state_ind0x = next_states_row_splits1[fsa_ind0],
              dest_state_ind1 = dest_state_ind01 - next_state_ind0x;
          arc->u.dest_state_ind1 = dest_state_ind1;

          float arc_loglike = arc->arc_loglike;
          int32_t dest_state_backward_loglike =
              next_states_data[dest_state_ind01].backward_loglike;
          // 'backward_loglike' is the loglike at the beginning of the arc
          float backward_loglike = arc_loglike + OrderedIntToFloat(
              dest_state_backward_loglike);
          float src_state_forward_loglike =
              OrderedIntToFloat(cur_states_data[arc_row_ids[arcs_ind012]].forward_loglike);
          char keep_this_arc = (backward_loglike + src_state_forward_loglike >= -beam);
          int32_t oshape_arc_ind0x = oshape_row_splits1[fsa_ind0],
              oshape_arc_ind01 = oshape_arc_ind0x + t,
              oshape_arc_ind01x = oshape_row_splits2[oshape_arc_ind01],
              oshape_arc_ind01xx = oshape_row_splits3[oshape_arc_ind01x],
              oshape_arc_ind0123 = oshape_arc_ind01xx + arc_indx12;
          // note, for the previous line: indexes 1 and 2 of FrameInfo::arcs (==state,arc)
          // become indexes 2 and 3 of oshape_unpruned_.

          keep_arcs_data[oshape_arc_ind0123] = keep_this_arc;
          arc_backward_prob_data[i] = FloatToOrderedInt(backward_loglike);
      };
      Eval(c_, arc_backward_prob.values.Dim(), lambda_set_arc_backward_prob);
    } else {
      assert(arc_backward_prob.size() == 0);
    }
    Eval(c_, cur_frame->arcs.values.size(), lambda_set_arc_backward_prob);

    /* note, the elements of state_backward_prob that don't have arcs leaving them will
       be set to the supplied default.  */
    Array1<int32_t> state_backward_prob = MaxPerSublist(
        arc_backward_prob,
        FloatToOrderedInt(-std::numeric_limits<float>::infinity()));


    int32_t num_fsas = a_fsas_.Dim0();
    assert(cur_frame->states.Dim0() == num_fsas);

    auto lambda_set_state_backward_prob = __host__ __device__ [=] (int32_t state_ind01) -> void {
        StateInfo *info = cur_states_data + state_ind01;
        int32_t fsas_state_ind01 = info->fsas_state_ind01,
            fsa_ind0 = a_fsas_row_ids1[fsas_state_ind01],
            fsas_state_ind0x_next = a_fsas_row_splits1[fsa_ind0+1];
        float forward_loglike = OrderedIntToFloat(info->forward_loglike),
            backward_loglike;
        int32_t is_final_state = (fsas_state_ind01+1 > fsas_state_ind0x_next);
        if (is_final_state) {
          backward_loglike = forward_loglike;
        } else {
          backward_loglike = FloatToOrderedInt(state_backward_prob[i]);
        }
        char keep_this_state = (backward_loglike + forward_loglike >= -beam);

        // we can use the arcs row-splits because the structure of
        // FrameInfo::states is the same as the top level structure of
        // FrameInfo::arcs.
        int32_t states_ind0x = next_arcs_row_splits1[fsa_ind0],
            states_indx1 = state_ind01 - state_ind0x;

        int32_t oshape_ind0x = oshape_row_splits1[fsa_ind0],
            oshape_ind01 = oshape_ind0x + t,
            oshape_ind01x = oshape_row_splits2[oshape_ind01],
            oshape_ind012 = oshape_ind01x + states_indx1;
        // note: axis 1 of 'states' corresponds to axis 2 of 'oshape'; it's the
        // state index.  Also,

        keep_states_data[oshape_ind012] = keep_this_state;
        if (!keep_this_state) {
          // The reason we set the backward_loglike to -infinity here if it's
          // outside the beam, is to prevent disconnected states from appearing
          // after pruning due to numerical roundoff effects near the boundary
          // at `-beam`.  It would otherwise be correct and harmless to omit
          // this if-block.
          backward_loglike = -std::numeric_limits<float>::infinity();
        }
        info->backward_loglike = FloatToOrderedInt(backward_loglike);
    };
    Eval(c_, cur_frame->states.values.size(), lambda_set_state_backward_prob);
  }


  /*
    renumber cur_frame->arcs and cur_frame->states, based on cur_frame->states_renumber,
    cur_frame->arcs_renumber and next_frame->states_renumber.  This gets rid of arcs that
    were pruned on the backward pass.  Initially I was going to combine this with formatting
    the output, but it became too complicated.
        @param [in]  cur_frame   The current frame's FrameInfo, that we are renumbering
        @param [in] next_frame   The next frame's FrameInfo, or nullptr if this is the last
                                 frame; used to modify the dest_frame_state_idx element of
                                 the ArcInfo.
        @param [in] num_arcs_renumbered  Will equal cur_frame->arcs_renumber[-1], supplied
                                 for performance reasons.
        @param [in] num_states_renumbered  Will equal cur_frame->states_renumber[-1], supplied
                                 for performance reasons.
  */
  void RenumberArcsAndStates(FrameInfo *cur_frame,
                             FrameInfo *next_frame,
                             int32_t num_arcs_renumbered,
                             int32_t num_states_renumbered) {
    Array<int32_t> mem(c_, num_states_renumbered + num_arcs_renumbered),
        row_ids1 = mem.Range(0, num_states_renumbered),
        row_ids2 = mem.Range(num_states_renumbered, num_arcs_renumbered);


    int32_t *new_row_ids1_data = row_ids1.Data(),
        *new_row_ids2_data = row_ids2.Data(),
        *old_row_ids1_data = cur_frame->arcs.shape.RowIds1().Data(),
        *old_row_ids2_data = cur_frame->arcs.shape.RowIds2().Data();

    Array<ArcInfo> new_ai(c_, num_arcs_renumbered);
    ArcInfo *new_ai_data = new_ai.Data();
    int32_t *arcs_renumber_data = cur_frame->arcs_renumber.Data(),
        *states_renumber_data = cur_frame->arcs_renumber.Data(),
        *next_states_renumber_data = (next_frame != nullptr ?
                                      next_frame->states_renumber.Data() : nullptr);
    // RE nullptr above: we won't ever dereference that because the last frame
    // will have zero arcs, so the kernel will never run.

    auto lambda_renumber_arcs = __host__ __device__ [=] (int i) {
        int32_t new_i = arcs_renumber_data[i],
            new_i1 = arcs_renumber_data[i+1];
        if (new_i1 == new_i)  // This arc was pruned away
          return;

                                                    };


    // TODO: we may actually not need the StateInfo, but renumbering it for now in case it's
    // useful for debugging.
    Array<StateInfo> new_si(c_, num_states_renumbered);
    ArcInfo *new_si_data = new_ai.Data();





  }


  /*
     Get the info about the number of states and arcs on this frame for each FSA
     in 0..num_fsas-1, after renumbering (i.e. taking into account the backward
     pass pruning which created `states_renumber` and `arcs_renumber`).
           @param [in] frame   FrameInfo that we need sizes for
           @param [out] this_states_sizes   Vector of size num_fsas; we write
                               to here the number of un-pruned states that are
                               active for each FSA.
           @param [out] this_arcs_sizes   Vector of size num_fsas; we write
                               to here the number of un-pruned arcs that are
                               active for each FSA.
  */
  void GetSizeInfo(ContextPtr ctx,
                   FrameInfo *frame,
                   Array1<int32_t> this_states_sizes,
                   Array1<int32_t> this_arcs_sizes) {
    int32_t num_fsas = a_fsas_.Dim0();
    CHECK_EQ(num_fsas, this_states_sizes.Size());
    CHECK_EQ(num_fsas, this_arcs_sizes.Size());

    // Note: frame->arcs.RowSplits1() == frame->states.RowSplits1().
    const int32_t *arcs_row_splits1 = frame->arcs.RowSplits1().Data(),
        *arcs_row_splits2 = frame->arcs.RowSplits2().Data(),
        *states_renumber = frame->states_renumber.Data(),
        *arcs_renumber = frame->arcs_renumber.Data();
    int32_t *states_sizes = this_states_sizes.Data(),
        *arcs_sizes = this_arcs_sizes.Data();

    auto lambda_get_size = __host__ __device__ [=] (int i) {
     int state_begin = arcs_row_splits1[i],
         state_end = arc_row_splits1[i+1],
         arc_begin = arcs_row_splits2[state_begin],
         arc_end = arcs_row_splits2[state_end],
         mapped_state_begin = states_renumber[state_begin],
         mapped_state_end = states_renumber[state_end],
         mapped_arc_begin = arcs_renumber[arc_begin],
         mapped_arc_end = arcs_renumber[arc_end];
     states_sizes[i] = mapped_state_end - mapped_state_begin;
     arcs_sizes[i] = mapped_arc_end = mapped_arc_begin;
                                               };
    Eval(c_, num_fsas, lambda_get_size);
  }

  /* Information associated with a state active on a particular frame..  */
  struct StateInfo {
    /* abs_state_id is the state-index in a_fsas_.  Note: the ind0 in here
       won't necessarily match the ind0 within FrameInfo::state if
       a_fsas_stride_ == 0. */
    int32_t a_fsas_state_ind01;

    /* Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
       FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
       represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
       use max not log-sum).  You can take the pruned lattice and rescore it if
       you want log-sum.  */
    int32_t forward_loglike;

    /* Note: this `backward_loglike` is the best score of any path from here to the
       end, minus the best path in the overall FSA, i.e. it's the backward score
       you get if, at the final-state, you set backward_loglike == forward_loglike.
       So backward_loglike + forward_loglike <= 0, and you can treat it somewhat
       like a posterior (except they don't sum to one as we're using max, not
       log-add).

       Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
       FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
       represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
       use max not log-sum).  You can take the pruned lattice and rescore it if
       you want log-sum.  */
    int32_t backward_loglike;
  };

  struct ArcInfo {   // for an arc that wasn't pruned away...
    int32_t a_fsas_arc_ind012;  // the arc-index in a_fsas_.
    float arc_loglike;  // loglike on this arc: equals loglike from data (nnet
                        // output, == b_fsas), plus loglike from the arc in a_fsas.

    union {
      // these 3 different ways of storing the index of the destination state are
      // used at different stages of the algorithm; we give them different names
      // for clarity.
      int32_t dest_a_fsas_state_ind01;  // The destination-state as an index into a_fsas_.
      int32_t dest_info_state_ind01;  // The destination-state as an index into the next
                                      // FrameInfo's `arcs` or `states`
      int32_t dest_info_state_ind1;   // The destination-state as an index the next
                                      // FrameInfo's `arcs` or `states`, this time
                                      // omitting the FSA-index.


    } u;

    float end_loglike;  // loglike at the end of the arc just before
                        // (conceptually) it joins the destination state.

  };


  // The information we have for each frame of the pruned-intersection (really: decoding)
  // algorithm.  We keep an array of these, one for each frame, up to the length of the
  // longest sequence we're decoding plus one.
  struct FrameInfo {
    // States that are active at the beginning of this frame.  Indexed
    // [fsa_ind][state_ind], where fsa_ind indexes b_fsas_ (and a_fsas_, if
    // a_fsas_stride_ != 0); and state_ind just enumerates the active states
    // on this frame.
    Ragged2<StateInfo> states;

    // Indexed [fsa_ind][state_ind][arc_ind].. the first 2 indexes are
    // the same as those into 'states' (the first 2 levels of the structure
    // are shared), and the last one enumerates the arcs leaving each of those
    // states.
    //
    // Note: there may be indexes [fsa_ind] that have no states (because that
    // FSA had fewer frames than the max), and indexes [fsa_ind][state_ind] that
    // have no arcs due to pruning.
    Ragged3<ArcInfo> arcs;

  };

  ContextPtr c_;
  FsaVec &a_fsas_;
  int32_t a_fsas_stride_;  // 1 if we use a different FSA per sequence, 0 if the
                           // decoding graph is shared.
  DenseFsaVec &b_fsas_;
  float beam_;
  int32_t max_active_;
  int32_t max_active_;
  Array1<float> dynamic_beams_;  // dynamic beams (initially just beam_ but
                                 // change due to max_active/min_active
                                 // constraints).
  Array1<int32_t> state_map_;  // state_map_ is of size (total number of states
                               // in all the FSAS in a_fsas_).  It is used on
                               // each frame to compute and store the mapping
                               // from active states to the position in the
                               // `states` array.  Between frames, all values
                               // have -1 in them.

  std::vector<FrameInfo*> frames_;

  // This is a rearranged version of the info in 'frames', computed at the end of the forward
  // pass before pruning.  It is indexed [fsa_id][t][state][arc].
  RaggedShape4 oshape_unpruned_;

  // these two Renumbering objects dictate how we renumber oshape_unpruned_,
  // i.e. which states and arcs we delete.  The data in their Keep() members,
  // which are vectors of chars, are written to in PropagateBackward().
  Renumbering renumber_output_states_;
  Renumbering renumber_output_arcs_;

  // This is as oshape_unpruned_, but after the backward-pass pruning.
  RaggedShape4 oshape_pruned_;

};

// compose/intersect array of FSAs (multiple streams decoding or training in
// parallel, in a batch)...
void IntersectDensePruned(FsaVec &a_fsas,
                          DenseFsaVec &b_fsas,
                          float beam,
                          int32_t max_active,
                          FsaVec *ofsa,
                          Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b) {

  // We should maybe give these objects some kind of Device() function so we can
  // use a single template to do this kind of thing.
  // We don't need to check device of arc_map_a / arc_map_b since we'll
  // output to them by assignment, overwriting any old data.
  CheckSameDevice(a_fsas, b_fsas);

  int32_t tot_num_active = a_fsas.tot_size1();

  // `state_id_map` is used on each frame to establish a mapping between
  // absolute-state-ids in a_fsas (i.e. indexes into a_fsas.{row_splits2(),row_ids3()})
  // and a linear vector of active states used on each frame.
  Array1<int32_t> state_id_map(tot_num_active, -1);


  /*
     T is the largest number of (frames+1) in any of the FSAs.  Note: each of the FSAs
     has a last-frame that has a NULL pointer to the data, and on this 'fake frame' we
     process final-transitions.
   */
  int32_t T = a_fsas.MaxSize1();




  Array1<int32_t> a_index;
  Array1<int32_t> b_index;

  Array1<int32_t> fsa_id;
  Array1<int32_t> out_state_id;

  Array1<HashKeyType> state_repr_hash;  // hash-value of corresponding elements of a_fsas and b_fsas

  Hash<HashKeyType, int32_t, Hasher> repr_hash_to_id;  // Maps from (fsa_index, hash of state_repr) to
}

}  // namespace k2
