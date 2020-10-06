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

namespace k2 {

// Caution: this is really a .cu file.  It contains mixed host and device code.

/*
   Pruned intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  Can use either different decoding graphs (one
   per acoustic sequence) or a shared graph
*/
class TopSorter {
 public:
  /**
     Topological sorter

       @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  TopSorter(FsaVec &fsas)
      : fsas_(fsas) {
    c_ = GetContext(fsas_);

    K2_CHECK_EQ(fsas_.NumAxes(), 3);
    int32_t num_fsas = fsas_.shape.TotSize(0),
        num_states = fsas_.shape.TotSize(1);
    state_map_ = Array1<int32_t>(c_, num_states, -1);
    // need 1 extra element allocated to avoid invalid read in ExclusiveSum()
    state_offsets_ = Array1<int32_t>(c_, num_fsas + 1, 0).Range(0, num_fsas);
  }


  // The information we have for each iteration of the algorithm.

  struct IterInfo {
    // This is a ragged array with 2 axes, indexed: [fsa_id][s] where s
    // is just an index into this list.  It contains idx01's into
    // fsas_ (i.e. state indexes).
    Ragged<int32_t> states;

    // This is of dimension [fsa_id]; it contains the value of state_offsets_
    // at the start of this iteration.  It will later be convenient to
    // use this to create the row_splits of the output.
    Array<int32_t> offsets;

    // This is a ragged array with 3 axes, where the first 2 axes are the same
    // as those of `states`.  It is: [fsa_index][list_of_states][list_of_arcs].
    // It contains the indexes of the destination states of those arcs.  These
    // are idx1's into the FSA that we will create at the output (as a special
    // case we use -1 when the destination state is the final-state).  We don't
    // need to store the corresponding arc-indexes (in the input FSA) because
    // these can be worked out from the information in `states`.
    //
    // Inside ComputeDestStates() we also temporarily store here numbers of
    // the form  - 2 - fsa_dest_state_idx01, for states that haven't been
    // numbered yet.
    Ragged<int32_t> dest_states;
  };

  int32_t NumFsas() { return fsas_.Dim0(); }


  /*
    Create the IterInfo object for the 1st iteration of the algorithm (not including
    its `dest_states` member).
   */
  std::unique_ptr<IterInfo> GetFirstIterInfo() {
    std::unique_ptr<IterInfo> ans = std::make_unique<IterInfo>();
    ans->offsets = Clone(state_offsets_);  // currently contains all 0.
    int32_t num_fsas = NumFsas();

    const int32_t *fsa_row_splits1 = fsas_.RowSplits(1).Data();
    int32_t *states_offsets_data = state_offsets_.Data();

    auto lambda_set_num_initial_states =
      [=] __host__ __device__ (int32_t i) {
      // num_initial_states for this FSA is 1 if the input FSA had at least
      // one state.
      state_offsets_data[i] = (fsa_row_splits1[i+1] > fsa_row_splits1[i]);
    };
    Eval(c_, num_fsas, lambda_set_num_initial_states);


    Array1<int32_t> states_row_splits(c_, num_fsas + 1);
    ExclusiveSum(state_offsets_, &states_row_splits);
    int32_t num_elems = state_row_splits[num_fsas];
    Array1<int32_t> row_ids(num_elems),
        state_idx01(num_elems);
    int32_t *states_row_splits_data = states_row_splits.Data(),
        *row_ids_data = row_ids.Data(),
        *state_idx01_data = state_idx01.Data();
    auto lambda_compute_states = [=] __host__ __device__ (int32_t i) {
      // i == fsa-index.
      int32_t this_row_split = states_row_splits[i],
          next_row_split = states_row_splits[i+1],
          num_states = next_row_split - this_row_split;

      if (num_states > 0) { // [then it's 1]..
        row_ids_data[this_row_split] = i;
        // '+ 0' is because we want state 0 of that FSA (the start state).
        int32_t this_state_idx01 = fsa_row_splits1[i] + 0;
        state_idx01_data[this_row_split] = this_state_idx01;
      }
    };

    ans->states = Ragged<int32>(RaggedShape2(states_row_splits, row_ids, num_elems),
                                state_idx01);

  }

  /*
    Assuming the `states` and `offsets` fields of cur_iter have already been set up,
    sets up `cur_iter.dest_states` and returns a ragged array with 2 axes consisting
    of any new state-ids that were not previously encountered, that need to have
    arcs leaving them processed.  (These will become the `states` element of
    the next IterInfo).
   */
  Ragged<int32_t> ComputeDestStates(IterInfo &cur_iter) {
    int32_t num_states = cur_iter.states.values.Dim();
    CHECK_GT(num_states, 0);
    // `num_arcs` will is the number of of arcs leaing each of these states;
    // the extra element is so that we can turn it into row_splits.
    Array1<int32_t> num_arcs = Array1<int32_t>(c_, num_states + 1);
    int32_t *num_arcs_data = num_arcs.Data();

    const int32_t *states_data = cur_iter.states.values.Data(),
        *states_row_ids = cur_iter.states.shape.RowIds(1).Data(),
        *fsas_row_splits2 = fsas_.shape.RowSplits(2).Data();
    auto lambda_get_num_arcs = [=] __host__ __device__ (int32_t i) {
       int32_t state_idx01 = states_data[i],
           num_arcs = fsas_row_splits2[state_idx01+1] -
              fsas_row_splits2[state_idx01];
       num_arcs_data[i] = num_arcs;
    };
    auto &states_row_splits = num_arcs;  // just an alias, for clarity.
    ExclusiveSum(num_arcs, &states_row_splits);
    int32_t num_arcs = states_row_splits[num_states];

    cur_iter.dest_states = Ragged<int32_t>(
        ComposeRaggedShapes(cur_iter.states.shape,
                            Ragged2(states_row_splits, nullptr, num_arcs)),
        Array<int32_t>(c_, num_arcs));

    // the elements of `is_new` from 0 to num_arcs - 1 will be set to 1 if this
    // arc "generates a new state" and 0 otherwise.  We'll use this to number
    // any newly created states.
    Array<int32_t> is_new(c_, num_arcs + 1, -1);
    const int32_t *dest_states_row_ids2 = cur_iter.dest_stats.shape.RowIds(2).Data(),
        *dest_states_row_splits2 = cur_iter.dest_stats.shape.RowSplits(2).Data(),
        *dest_states_row_ids1 = cur_iter.dest_stats.shape.RowIds(1).Data(),
        *fsas_row_splits2 = fsas_.shape.RowSplits(2);
    const Arc *arcs = fsas_.values.Data();
    int32_t *dest_states_data = cur_iter.dest_states.Data(),
        *state_map_data = state_map_.Data(),
        *is_new_data = is_new.Data();

    auto lambda_set_dest_states1 = [=] __host__ __device__ (int32_t i) {
       int32_t arc_idx012 = i;  // into cur_iter.dest_states
       int32_t state_idx01 = dest_states_row_ids2[arc_idx012],
           fsa_idx0 = dest_states_row_ids1[state_idx01],
           arc_idx0x = dest_states_row_splits2[state_idx01],
           arc_idx1 = arc_idx01 - arc_idx0x;
       // `fsa_state_idx01` is an idx01 into `fsas_`.
       int32_t fsa_state_idx01 = states_data[state_idx01],
           fsa_arc_idx01x = fsas_row_splits2[fsa_state_idx01],
           fsa_arc_idx012 = fsa_arc_idx01x + arc_idx1,
           fsa_dest_state_idx01 = arcs[fsa_arc_idx012].dest_state,
           symbol = arcs[fsa_arc_idx012].symbol;
       if (symbol == -1) {
         // dest-state is final-state, treat this specially as it has to be
         // numbered last so we can't figure out its numbering yet.
         dest_states_data[arc_idx012] = -1;
       } else {
         int32_t cur_state_idx1 = state_map_data[fsa_dest_state_idx01];
         if (cur_state_idx1 >= 0) {  // This is an already-processed state,
                                     // i.e. we know its numbering within this
                                     // output FSA
           dest_states_data[arc_idx012] = cur_state_idx1;
         } else {
           // temporarily set it to -2 - arc_idx012; if there are multiple arcs
           // going to this state, one will (arbitrarily) "win" and we'll use
           // this to decide the numbering of the new state.
           state_map_data[fsa_dest_state_idx01] = -2 - arc_idx012;
           // temporarily set `dest_states_data` to a number < -1 from which
           // we can work out the corresponding index into `state_map`.
           dest_states_data[arc_idx012] = -2 - fsa_dest_state_idx01;
         }
       }
    };
    Eval(c_, num_arcs, lambda_set_dest_states1);



  }

  /* Does the main work of top-sorting and returns the resulting FSAs. */
  FsaVec TopSort() {

  }

    iters_.push_back(GetFirstIterInfo());


    /*
      T is the largest number of (frames+1) of neural net output, or the largest
      number of frames of log-likelihoods we count the final frame with (0,
      -inf, -inf..) that is used for the final-arcc.  The largest number of
      states in the fsas represented by b_fsas equals T+1 (e.g. 1 frame would
      require 2 states, because that 1 frame is the arc from state 0 to state
      1).  So the #states is 2 greater than the actual number of frames in the
      neural-net output.
    */
    int32_t T = b_fsas_.shape.MaxSize(1), num_fsas = b_fsas_.shape.Dim0();

    frames_.reserve(T + 1);

    frames_.push_back(InitialFrameInfo());  // TODO: implement
                                            // InitialFrameInfo().
    for (int32_t t = 0; t < T; t++) {
      frames_.push_back(PropagateForward(t, frames_.back()));
    }

    {
      // each of these have 3 axes.
      std::vector<const RaggedShape *> arcs_shapes(T + 1);
      for (int32_t t = 0; t <= T; t++)
        arcs_shapes[t] = &(frames_[t]->arcs.shape);

      // oshape_unpruned_ is a 4-axis ragged tensor which is indexed:
      //   oshape_unpruned_[fsa_index][t][state_idx][arc_idx]
      // This is BEFORE BACKWARD PRUNING... oshape_pruned_ will
      // be after backward pruning
      int32_t axis = 1;
      oshape_unpruned_ = Stack(axis, T + 1, &(arcs_shapes[0]));
    }
    renumber_output_states_.Init(oshape_unpruned_.TotSize(2));
    renumber_output_arcs_.Init(oshape_unpruned_.TotSize(3));

    for (int32_t t = T; t >= 0; t--) {
      // this writes to elements of renumber_output_states_.Keep() and
      // renumber_output_arcs_.Keep().
      PropagateBackward(t, frames_[t], (t == T ? NULL : frames_[t + 1]));
    }
    // TODO(haowen): replace with SubsampleRaggedShape?
    // oshape_pruned_ = RaggedShape4Subsampled(oshape_unpruned_, NULL, NULL,
    //                                        &renumber_output_states_,
    //                                        &renumber_output_arcs_);
  }

  FrameInfo *InitialFrameInfo() {
    // TODO
    K2_LOG(FATAL) << "Not Implemeted";
    return nullptr;
  }

  void FormatOutput(FsaVec *ofsa, Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b) {
    ContextPtr c_cpu = c_->GetCpuContext();
    int32_t T = a_fsas_.shape.MaxSize(1);

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
    int32_t *reverse_arc_map0123 = renumber_output_states_.New2Old().Data(),
            *reverse_state_map012 = renumber_output_states_.New2Old().Data();

    Array1<ArcInfo *> arcs_data_ptrs(c_cpu, T + 1);
    Array1<int32_t *> arcs_row_splits1_ptrs(c_cpu, T + 1);
    Array1<int32_t *> arcs_row_splits2_ptrs(c_cpu, T + 1);

    for (int32_t t = 0; t <= T; t++) {
      arcs_data_ptrs.Data()[t] = frames_[t]->arcs.values.Data();
      arcs_row_splits1_ptrs.Data()[t] =
          frames_[t]->arcs.shape.RowSplits(1).Data();
      arcs_row_splits2_ptrs.Data()[t] =
          frames_[t]->arcs.shape.RowSplits(2).Data();
    }
    // transfer to GPU if we're using a GPU
    arcs_data_ptrs = arcs_data_ptrs.To(c_);
    arcs_row_splits1_ptrs = arcs_row_splits1_ptrs.To(c_);
    arcs_row_splits2_ptrs = arcs_row_splits2_ptrs.To(c_);
    ArcInfo **arcs_data_ptrs_data = arcs_data_ptrs.Data();
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

    auto lambda_format_arc_data =
        [=] __host__ __device__(int32_t pruned_idx0123) -> void {
      int32_t unpruned_idx0123 = reverse_state_map012[pruned_idx0123];
      int32_t unpruned_idx012 = oshapeu_row_ids3[unpruned_idx0123],
              unpruned_idx01 = oshapeu_row_ids2[unpruned_idx012],
              unpruned_idx01x = oshapeu_row_splits2[unpruned_idx01],
              unpruned_idx01xx = oshapeu_row_splits3[unpruned_idx01x],
              unpruned_idxxx23 = unpruned_idx0123 - unpruned_idx01xx,
              unpruned_idx0 = oshapeu_row_ids1[unpruned_idx01],  // fsa-id
          // unpruned_idx0x = oshapeu_row_splits1[unpruned_idx0],
          // unpruned_idx0xx = oshapeu_row_splits2[unpruned_idx0x],
          unpruned_idx1 = unpruned_idx01 - unpruned_idx01,  // t
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
      int32_t arcs_idx012 = arcs_idx0xx + unpruned_idxxx23;
      ArcInfo *arcs_data = arcs_data_ptrs_data[t];
      ArcInfo arc_info = arcs_data[arcs_idx012];

      // we call it ind2 because the state-index is axis 2 of oshape.
      int32_t unpruned_dest_state_idx2 = arc_info.u.dest_info_state_idx1,
              unpruned_dest_state_idx012 =
                  unpruned_idx01x_next_t + unpruned_dest_state_idx2,
              pruned_dest_state_idx012 =
                  reverse_state_map012[unpruned_dest_state_idx012],
              pruned_dest_state_idx01 =
                  oshapep_row_ids2[pruned_dest_state_idx012],
              pruned_dest_state_idx0 =
                  oshapep_row_ids1[pruned_dest_state_idx01],
              pruned_dest_state_idx0x =
                  oshapep_row_splits1[pruned_dest_state_idx0],
              pruned_dest_state_idx0xx =
                  oshapep_row_splits2[pruned_dest_state_idx0x],
              pruned_dest_state_idxx12 =
                  pruned_dest_state_idx012 - pruned_dest_state_idx0xx;

      // note: the src-state and dest-state have the same ind0 which is the
      // FSA-id.
      int32_t pruned_src_state_idx012 = reverse_state_map012[unpruned_idx012],
              pruned_src_state_idxx12 =
                  pruned_src_state_idx012 - pruned_dest_state_idx0xx;

      Arc arc;
      // The numbering for the dest-state in the output Arc is the numbering
      // *within the FSA*, and we ignore the time index (1) because that index
      // will be removed as the FSA format has no notion of time; that's why we
      // use the indx12.

      arc_map_a_data[pruned_idx0123] = arc_info.a_fsas_arc_idx012;

      arc.src_state = pruned_src_state_idxx12;
      arc.dest_state = pruned_dest_state_idxx12;
      arc.symbol = a_fsas_arcs[arc_info.a_fsas_arc_idx012].symbol;
      int32_t fsa_id = unpruned_idx0, b_fsas_idx0x = b_fsas_row_ids1[fsa_id],
              b_fsas_idx01 = b_fsas_idx0x + t, b_fsas_idxxx2 = (arc.symbol + 1),
              b_fsas_arc_idx012 =
                  b_fsas_idx01 * b_fsas_num_cols + b_fsas_idxxx2;
      arc.score = arc_info.arc_loglike;

      arc_map_b_data[pruned_idx0123] = b_fsas_arc_idx012;
      arcs_out_data[pruned_idx0123] = arc;
    };
    Eval(c_, tot_arcs_pruned, lambda_format_arc_data);

    // The output shape will get rid of axis one of oshape_pruned_ (which is the
    // 't' index).
    Array1<int32_t> arcs_row_splits1_out = oshape_pruned_.RowSplits(
                        2)[oshape_pruned_.RowSplits(1)],
                    arcs_row_ids1_out =
                        oshape_pruned_.RowIds(1)[oshape_pruned_.RowIds(2)];

    auto lambda_set_row_splits1_out =
        [=] __host__ __device__(int32_t i) -> void {
      // TODO
      K2_LOG(FATAL) << "Not Implemented";
    };
    Eval(c_, oshape_pruned_.RowSplits(1).Dim(), lambda_set_row_splits1_out);

    RaggedShape output_fsas_shape = RaggedShape3(
        &arcs_row_splits1_out, &arcs_row_ids1_out, -1,
        &oshape_pruned_.RowSplits(3), &oshape_pruned_.RowIds(3), -1);

    *ofsa = FsaVec(output_fsas_shape, arcs_out);
  }

  /*
    Computes pruning cutoffs for this frame: these are the cutoffs for the arc
    "forward score", one per FSA.  This is a dynamic process involving
    dynamic_beams_ which are updated on each frame (they start off at beam_).

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
                    the dynamic beam is adjusted; it will approach 'beam_'
                    as long as the number of active states in each FSA is
                    between min_active and max_active.
  */
  Array1<float> GetPruningCutoffs(Ragged<float> &arc_end_scores) {
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
    Array1<float> max_per_fsa;
    MaxPerSublist(end_scores_per_fsa, -std::numeric_limits<float>::infinity(),
                  &max_per_fsa);
    const int32_t *per_fsa_row_splits1_data =
        end_scores_per_fsa.shape.RowSplits(1).Data();

    const float *max_per_fsa_data = max_per_fsa.Data();
    float *dynamic_beams_data = dynamic_beams_.Data();

    float default_beam = beam_, max_active = max_active_,
          min_active = min_active_;

    Array1<float> cutoffs(c_, num_fsas);
    float *cutoffs_data = cutoffs.Data();

    auto lambda_set_beam_and_cutoffs =
        [=] __host__ __device__(int32_t i) -> void {
      float best_loglike = max_per_fsa_data[i],
            dynamic_beam = dynamic_beams_data[i];
      int32_t num_active =
          per_fsa_row_splits1_data[i + 1] - per_fsa_row_splits1_data[i];
      float new_dynamic_beam;
      if (num_active <= max_active) {
        // Not constrained by max_active...
        if (num_active > min_active) {
          // Neither the max_active nor min_active constraints
          // apply.  Gradually approach 'beam'
          new_dynamic_beam = 0.8 * dynamic_beam + 0.2 * default_beam;
        } else {
          // We violated the min_active constraint -> increase beam
          if (new_dynamic_beam < default_beam) new_dynamic_beam = default_beam;
          // gradually make the beam larger as long
          // as we are below min_active
          new_dynamic_beam *= 1.25;
        }
      } else {
        // We violated the max_active constraint -> decrease beam
        if (new_dynamic_beam > default_beam) new_dynamic_beam = default_beam;
        // Decrease the beam as long as we have more than
        // max_active active states.
        new_dynamic_beam *= 0.85;
      }
      dynamic_beams_data[i] = new_dynamic_beam;
      cutoffs_data[i] = best_loglike - new_dynamic_beam;
    };
    Eval(c_, num_fsas, lambda_set_beam_and_cutoffs);
    return cutoffs;
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
  Ragged<ArcInfo> GetUnprunedArcs(int32_t t, FrameInfo *cur_frame) {
    Ragged<StateInfo> &states = cur_frame->states;
    const StateInfo *state_values = states.values.Data();

    // in a_fsas_ (the decoding graphs), maps from state_idx01 to arc_idx01x.
    const int32_t *fsa_arc_splits = a_fsas_.shape.RowSplits(2).Data();

    // int32_t a_fsas_stride = a_fsas_stride_;

    // frame_state_idx01 combines the FSA-index and state-index (into
    // 'cur_frame->states')
    Array1<int32_t> num_arcs(c_, states.values.Dim());
    int32_t *num_arcs_data = num_arcs.Data();
    auto num_arcs_lambda =
        [=] __host__ __device__(int32_t state_idx01) -> void {
      int32_t a_fsas_state_idx01 = state_values[state_idx01].a_fsas_state_idx01,
              a_fsas_arc_idx01x = fsa_arc_splits[a_fsas_state_idx01],
              a_fsas_arc_idx01x_next = fsa_arc_splits[a_fsas_state_idx01 + 1],
              a_fsas_num_arcs_x1x = a_fsas_arc_idx01x - a_fsas_arc_idx01x_next;
      num_arcs_data[state_idx01] = a_fsas_num_arcs_x1x;
    };
    // `num_arcs` gives the num-arcs for each state in `states`.
    Eval(c_, num_arcs.Dim(), num_arcs_lambda);

    // initialize shape of array that will hold arcs leaving the active states.
    // Its shape is [fsa_index][state][arc]; the top two levels are shared with
    // `states`.  'ai' means ArcInfo.
    // TODO(haowen): implement RaggedShapeFromSizes?
    // RaggedShape ai_shape = RaggedShapeFromSizes(states.shape, num_arcs);
    RaggedShape ai_shape;

    // 'ai' means ArcInfo

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
    const float *score_data = b_fsas_.scores.Data();
    int32_t scores_num_cols = b_fsas_.scores.Dim1();

    Ragged<ArcInfo> ai(ai_shape);
    ArcInfo *ai_data = ai.values.Data();  // uninitialized

    auto ai_lambda = [=] __host__ __device__(int32_t ai_arc_idx012) -> void {
      int32_t ai_state_idx01 = ai_row_ids2[ai_arc_idx012],
              ai_fsa_idx0 = ai_row_ids1[ai_state_idx01],
              ai_arc_idx01x = ai_row_splits2[ai_state_idx01],
              ai_arc_idxxx2 = ai_arc_idx012 - ai_arc_idx01x;
      StateInfo sinfo = state_values[ai_state_idx01];
      int32_t a_fsas_arc_idx01x = a_fsas_row_splits2[sinfo.a_fsas_state_idx01],
              a_fsas_arc_idx012 = a_fsas_arc_idx01x + ai_arc_idxxx2;
      Arc arc = arcs[a_fsas_arc_idx012];

      int32_t scores_idx0x = b_fsas_row_ids1[ai_fsa_idx0],
              scores_idx01 = scores_idx0x + t,  // t == ind1 into 'scores'
          scores_idx2 = arc.symbol + 1,  // the +1 is so that -1 can be handled
          scores_idx012 = (scores_idx01 * scores_num_cols) + scores_idx2;
      assert(static_cast<uint32_t>(scores_idx2) <
             static_cast<uint32_t>(scores_num_cols));
      float acoustic_score = score_data[scores_idx012];
      ArcInfo ai;
      ai.a_fsas_arc_idx012 = a_fsas_arc_idx012;
      ai.arc_loglike = acoustic_score + arc.score;
      ai.end_loglike = sinfo.forward_loglike + ai.arc_loglike;
      // at least currently, the ArcInfo object's src_state and dest_state are
      // ind1's not idx01's, i.e. they don't contain the FSA-index, where as
      // the ai element is an idx01, so we need to do this to conver to an
      // idx01; this relies on the fact that sinfo.abs_state_id == arc.src_state
      // + a_fsas_fsa_idx0x.
      ai.u.dest_a_fsas_state_idx01 =
          sinfo.a_fsas_state_idx01 + arc.dest_state - arc.src_state;
      ai_data[ai_arc_idx012] = ai;
    };
    Eval(c_, ai.values.Dim(), ai_lambda);
    return ai;
  }

  /*
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* object for the next frame.
   */
  FrameInfo *PropagateForward(int32_t t, FrameInfo *cur_frame) {
    // Ragged<StateInfo> &states = cur_frame->states;
    // ai has 3 axes: fsa_id, state, arc.
    Ragged<ArcInfo> arc_info = GetUnprunedArcs(t, cur_frame);
    const ArcInfo *ai_data = arc_info.values.Data();
    Array1<float> ai_data_array1(c_, arc_info.values.Dim());
    float *ai_data_array1_data = ai_data_array1.Data();
    auto lambda_set_ai_data = [=] __host__ __device__(int32_t i) -> void {
      ai_data_array1_data[i] = ai_data[i].end_loglike;
    };
    Eval(c_, ai_data_array1.Dim(), lambda_set_ai_data);
    Ragged<float> ai_loglikes(arc_info.shape, ai_data_array1);

    // `cutoffs` is of dimension num_fsas.
    Array1<float> cutoffs = GetPruningCutoffs(ai_loglikes);
    int32_t num_fsas = cutoffs.Dim();

    float *cutoffs_data = cutoffs.Data();

    // write certain indexes ( into ai.values) to state_map_.Data().  Keeps
    // track of the active states and will allow us to assign a numbering to
    // them.
    int32_t *ai_row_ids1 = arc_info.shape.RowIds(1).Data(),
            *ai_row_ids2 = arc_info.shape.RowIds(2).Data(),
            *state_map_data = state_map_.Data();
    // We use a separate state_map vector per FSA we're processing if a_fsas_
    // only has one FSA (in this case it means we're sharing the FSA among
    // potentially multiple streams).
    int32_t state_map_stride =
        (a_fsas_.shape.Dim0() > 1 ? 0 : state_map_.ElemStride0());
    auto lambda_set_state_map =
        [=] __host__ __device__(int32_t arc_idx012) -> void {
      int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
      int32_t dest_state_idx012 = ai_data[arc_idx012].a_fsas_arc_idx012;
      float end_loglike = ai_data[arc_idx012].end_loglike,
            cutoff = cutoffs_data[fsa_id];
      if (end_loglike > cutoff) {
        // The following is a race condition as multiple threads may write to
        // the same location, but it doesn't matter, the point is to assign
        // one of the indexes.
        state_map_data[fsa_id * state_map_stride + dest_state_idx012] =
            arc_idx012;
      }
    };
    Eval(c_, arc_info.values.Dim(), lambda_set_state_map);

    // renumber_arcs will be a renumbering that dictates which of the arcs
    // in 'ai' we keep
    Renumbering renumber_arcs(arc_info.values.Dim());

    // renumber_states will be a renumbering that dictates which of the arcs in
    // 'ai' correspond to unique states.  Only one arc for each dest-state is
    // kept (it doesn't matter which one).
    Renumbering renumber_states(arc_info.values.Dim());

    // Note: we don't just keep arcs that were above the pruning threshold, we
    // keep all arcs whose destination-states survived pruning.  Later we'll
    // prune with the lattice beam, using both forward and backward scores.
    char *keep_this_arc_data = renumber_arcs.Keep().Data(),
         *keep_this_state_data = renumber_states.Keep().Data();
    if (state_map_stride == 0) {
      auto lambda_set_keep = [=] __host__ __device__(int32_t i) -> void {
        int32_t dest_state = ai_data[i].u.dest_a_fsas_state_idx01;
        int32_t j = state_map_data[dest_state];
        if (j != -1) {  // the dest-state was kept..
          keep_this_arc_data[i] = 1;
          // caution: keep_this_state_data is indexed by *arc*
          if (j == i)  // this arc 'won' the data race..
            keep_this_state_data[i] = 1;
        }
      };
      Eval(c_, arc_info.values.Dim(), lambda_set_keep);
    } else {
      auto lambda_set_keep =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                dest_state = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
        int j = state_map_data[dest_state + fsa_id * state_map_stride];
        if (j != -1) {  // the dest-state was kept..
          keep_this_arc_data[arc_idx012] = 1;
          // caution: keep_this_state_data is indexed by *arc*
          if (j == arc_idx012)  // this arc 'won' the data race..
            keep_this_state_data[arc_idx012] = 1;
        }
      };
      Eval(c_, arc_info.values.Dim(), lambda_set_keep);
    }

    int32_t num_arcs = renumber_arcs.NumNewElems(),
            num_states = renumber_states.NumNewElems();

    int32_t *arc_reorder_data = renumber_arcs.Old2New().Data(),
            *state_reorder_data = renumber_states.Old2New().Data();

    // state_to_fsa_id maps from an index into the next frame's
    // FrameInfo::states.values() vector (i.e. a frame-state-index on the *next*
    // frame) to the FSA-id associated with it.  It should be non-decreasing.
    Array1<int32_t> state_to_fsa_id(c_, num_states);
    {  // This block sets 'state_to_fsa_id'.
      int32_t *state_to_fsa_id_data = state_to_fsa_id.Data();
      auto lambda_state_to_fsa_id =
          [=] __host__ __device__(int32_t arc_idx012) -> void {
        int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]],
                this_state_j = state_reorder_data[arc_idx012 + 1],
                next_state_j = state_reorder_data[arc_idx012 + 1];
        if (next_state_j > this_state_j) {
          state_to_fsa_id_data[this_state_j] = fsa_id;
        }
      };
      Eval(c_, arc_info.values.Dim(), lambda_state_to_fsa_id);

      // TODO(haowen): implement it?
      // assert(IsMonotonic(state_to_fsa_id));
    }
    // The following creates a structure that contains a subset of the elements
    // of `arc_info`, determined by the renumbering in `renumber_arcs`.
    Array1<ArcInfo> curr_frame_arc_array1(c_, num_arcs);
    RaggedShape curr_frame_arc_shape =
        SubsampleRaggedShape(arc_info.shape, renumber_arcs);
    cur_frame->arcs =
        Ragged<ArcInfo>(curr_frame_arc_shape, curr_frame_arc_array1);

    FrameInfo *ans = new FrameInfo();
    // TODO(haowen): implement Ragged2FromRowIds?
    // ans->states = Ragged2FromRowIds(num_fsas, state_to_fsa_id,
    //                               Array1<ArcInfo>(c_, num_arcs));
    StateInfo *ans_states_data = ans->states.values.Data();
    const int32_t minus_inf =
        FloatToOrderedInt(-std::numeric_limits<float>::infinity());
    auto lambda_init_loglike = [=] __host__ __device__(int32_t i) -> void {
      ans_states_data[i].forward_loglike = minus_inf;
    };
    Eval(c_, num_states, lambda_init_loglike);

    // Modify the elements of `state_map` to refer to the indexes into
    // `ans->states` / `kept_states_data`, rather than the indexes into ai_data.
    // This will decrease some of the values in `state_map`, in general.
    auto lambda_modify_state_map =
        [=] __host__ __device__(int32_t arc_idx012) -> void {
      int32_t fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];
      int32_t dest_state_idx = ai_data[arc_idx012].u.dest_a_fsas_state_idx01;
      int32_t this_j = state_reorder_data[arc_idx012],
              next_j = state_reorder_data[arc_idx012 + 1];
      if (next_j > this_j) {
        // this_j is the idx01 into ans->states
        int32_t state_map_idx = dest_state_idx + fsa_id * state_map_stride;
        K2_CHECK_EQ(state_map_data[state_map_idx], dest_state_idx);
        state_map_data[state_map_idx] = this_j;
        // Note: this_j is an idx01 into ans->states.
      }
    };
    Eval(c_, arc_info.values.Dim(), lambda_modify_state_map);

    // We'll set up the data of the kept arcs and states below...
    ArcInfo *kept_ai_data = cur_frame->arcs.values.Data();
    StateInfo *kept_states_data = ans->states.values.Data();

    auto lambda_set_arcs_and_states =
        [=] __host__ __device__(int32_t arc_idx012) -> void {
      // arc_idx012 is the inde into the unpruned arcs, 'ai'
      int32_t this_pruned_idx012 = arc_reorder_data[arc_idx012],
              next_pruned_idx012 = arc_reorder_data[arc_idx012 + 1],
              fsa_id = ai_row_ids1[ai_row_ids2[arc_idx012]];

      // Note: I have a idea to reduce main-memory bandwidth by
      // caching writes in a fixed-size array in shared memory
      // (store: best-prob, index).  We'd go round robin, try e.g.
      // twice.
      // Would have to do this with a functor rather than a lambda,
      // as __shared__ won't work on CPU.
      if (next_pruned_idx012 > this_pruned_idx012) {
        // ... this was one of the arcs to keep.  Load the ArcInfo from the
        // un-pruned array..
        ArcInfo info = ai_data[arc_idx012];
        int32_t state_map_idx =
            info.u.dest_a_fsas_state_idx01 + fsa_id * state_map_stride;
        // state_idx01 is the index into ans->states, of the destination state.
        // Note: multiple arcs may enter this state, which is why we had to set
        // that in a separate kernel (lambda_modify_state_map).
        int32_t state_idx01 = state_map_data[state_map_idx];
        // multiple threads may write the same value to the address written to
        // in the next line.
        kept_states_data[state_idx01].a_fsas_state_idx01 =
            info.u.dest_a_fsas_state_idx01;
        info.u.dest_info_state_idx01 = state_idx01;
        kept_ai_data[this_pruned_idx012] = info;
        int32_t end_loglike_int = FloatToOrderedInt(info.end_loglike);
        // Set the forward log-like of the dest state to the largest of any of
        // the incoming arcs.  Note: we initialized this in lambda_init_loglike
        // above.
        atomicMax(&(kept_states_data[state_idx01].forward_loglike),
                  end_loglike_int);
      }
    };
    Eval(c_, arc_info.values.Dim(), lambda_set_arcs_and_states);

    const int32_t *next_states_row_ids1 = ans->states.shape.RowIds(1).Data();
    auto lambda_reset_state_map =
        [=] __host__ __device__(int32_t state_idx01) -> void {
      int32_t a_fsas_state_idx01 =
                  kept_states_data[state_idx01].a_fsas_state_idx01,
              fsa_idx0 = next_states_row_ids1[state_idx01];
      int32_t state_map_idx = a_fsas_state_idx01 + fsa_idx0 * state_map_stride;
      K2_CHECK_EQ(state_map_data[state_map_idx], state_idx01);
      state_map_data[state_map_idx] = -1;
    };

    Eval(c_, ans->states.values.Dim(), lambda_reset_state_map);

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
    int32_t num_states = cur_frame->states.values.Dim(),
            num_arcs = cur_frame->arcs.values.Dim();
    Ragged<StateInfo> &cur_states = cur_frame->states;  // 2 axes: fsa,state
    StateInfo *cur_states_data = cur_states.values.Data();

    int32_t tot_states = a_fsas_.shape.TotSize(1);
    int32_t *a_fsas_row_ids1 = a_fsas_.shape.RowIds(1).Data(),
            *a_fsas_row_splits1 = a_fsas_.shape.RowSplits(1).Data();

    const int32_t minus_inf =
        FloatToOrderedInt(-std::numeric_limits<float>::infinity());

    /* arc_backward_probs represents the backward-prob at the beginning of the
       arc.  Indexing is [frame_state_index][arc_index], where frame_state_index
       and arc_index are respectively idx01 and ind2 w.r.t. frames_[t]->arcs. */
    RaggedShape sub_curr_frame_shape = RemoveAxis(cur_frame->arcs.shape, 0);
    Array1<int32_t> sub_curr_frame_values(c_, num_arcs);
    Ragged<int32_t> arc_backward_prob(sub_curr_frame_shape,
                                      sub_curr_frame_values);
    int32_t *arc_backward_prob_data = arc_backward_prob.values.Data();

    ArcInfo *arcs_data = cur_frame->arcs.values.Data();
    int32_t *arcs_rowids1 = cur_frame->arcs.shape.RowIds(2).Data(),
            *arcs_rowids2 = cur_frame->arcs.shape.RowIds(2).Data(),
            *arcs_row_splits1 = cur_frame->arcs.shape.RowSplits(1).Data(),
            *arcs_row_splits2 = cur_frame->arcs.shape.RowSplits(2).Data();
    float beam = beam_;

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
        ArcInfo *arc = arcs_data + arcs_idx012;
        int32_t state_idx01 = arcs_rowids2[arcs_idx012],
                fsa_idx0 = arcs_rowids1[state_idx01],
                fsa_idx0x = arcs_row_splits1[fsa_idx0],
                fsa_idx0xx = arcs_row_splits2[fsa_idx0x],
                arcs_idxx12 = arcs_idx012 - fsa_idx0xx;

        int32_t dest_state_idx01 = arc->u.dest_info_state_idx01,
                next_state_idx0x = next_states_row_splits1[fsa_idx0],
                dest_state_idx1 = dest_state_idx01 - next_state_idx0x;
        arc->u.dest_info_state_idx1 = dest_state_idx1;

        float arc_loglike = arc->arc_loglike;
        int32_t dest_state_backward_loglike =
            next_states_data[dest_state_idx01].backward_loglike;
        // 'backward_loglike' is the loglike at the beginning of the arc
        float backward_loglike =
            arc_loglike + OrderedIntToFloat(dest_state_backward_loglike);
        float src_state_forward_loglike = OrderedIntToFloat(
            cur_states_data[arcs_rowids2[arcs_idx012]].forward_loglike);
        char keep_this_arc =
            (backward_loglike + src_state_forward_loglike >= -beam);
        int32_t oshape_arc_idx0x = oshape_row_splits1[fsa_idx0],
                oshape_arc_idx01 = oshape_arc_idx0x + t,
                oshape_arc_idx01x = oshape_row_splits2[oshape_arc_idx01],
                oshape_arc_idx01xx = oshape_row_splits3[oshape_arc_idx01x],
                oshape_arc_idx0123 = oshape_arc_idx01xx + arcs_idxx12;
        // note, for the previous line: indexes 1 and 2 of FrameInfo::arcs
        // (==state,arc) become indexes 2 and 3 of oshape_unpruned_.

        keep_arcs_data[oshape_arc_idx0123] = keep_this_arc;
        arc_backward_prob_data[arcs_idx012] =
            FloatToOrderedInt(backward_loglike);
      };
      Eval(c_, arc_backward_prob.values.Dim(),
           lambda_set_arc_backward_prob_and_keep);
    } else {
      assert(arc_backward_prob.values.Dim() == 0);
    }
    // Eval(c_, cur_frame->arcs.values.Dim(), lambda_set_arc_backward_prob);

    /* note, the elements of state_backward_prob that don't have arcs leaving
       them will be set to the supplied default.  */
    Array1<int32_t> state_backward_prob;
    MaxPerSublist(arc_backward_prob,
                  FloatToOrderedInt(-std::numeric_limits<float>::infinity()),
                  &state_backward_prob);
    const int32_t *state_backward_prob_data = state_backward_prob.Data();

    int32_t num_fsas = a_fsas_.shape.Dim0();
    assert(cur_frame->states.shape.Dim0() == num_fsas);
    float float_minus_inf = -std::numeric_limits<float>::infinity();
    auto lambda_set_state_backward_prob =
        [=] __host__ __device__(int32_t state_idx01) -> void {
      StateInfo *info = cur_states_data + state_idx01;
      int32_t fsas_state_idx01 = info->a_fsas_state_idx01,
              fsa_idx0 = a_fsas_row_ids1[fsas_state_idx01],
              fsas_state_idx0x_next = a_fsas_row_splits1[fsa_idx0 + 1];
      float forward_loglike = OrderedIntToFloat(info->forward_loglike),
            backward_loglike;
      int32_t is_final_state = (fsas_state_idx01 + 1 > fsas_state_idx0x_next);
      if (is_final_state) {
        backward_loglike = forward_loglike;
      } else {
        backward_loglike =
            FloatToOrderedInt(state_backward_prob_data[state_idx01]);
      }
      char keep_this_state = (backward_loglike + forward_loglike >= -beam);

      // we can use the arcs row-splits because the structure of
      // FrameInfo::states is the same as the top level structure of
      // FrameInfo::arcs.
      int32_t states_idx0x = arcs_row_splits1[fsa_idx0],
              states_idxx1 = state_idx01 - states_idx0x;

      int32_t oshape_idx0x = oshape_row_splits1[fsa_idx0],
              oshape_idx01 = oshape_idx0x + t,
              oshape_idx01x = oshape_row_splits2[oshape_idx01],
              oshape_idx012 = oshape_idx01x + states_idxx1;
      // note: axis 1 of 'states' corresponds to axis 2 of 'oshape'; it's the
      // state index.  Also,

      keep_states_data[oshape_idx012] = keep_this_state;
      if (!keep_this_state) {
        // The reason we set the backward_loglike to -infinity here if it's
        // outside the beam, is to prevent disconnected states from appearing
        // after pruning due to numerical roundoff effects near the boundary
        // at `-beam`.  It would otherwise be correct and harmless to omit
        // this if-block.
        backward_loglike = float_minus_inf;
      }
      info->backward_loglike = FloatToOrderedInt(backward_loglike);
    };

    Eval(c_, cur_frame->states.values.Dim(), lambda_set_state_backward_prob);
  }

  ContextPtr c_;
  FsaVec &fsas_;
  std::vector<unique_ptr<IterInfo> > iters_;

  // state_map_ is of size (total number of states
  // in fsas_.  It maps to:
  //  - -1 if this state has not been reached yet
  //  -  If this state has been reached on a previous
  //     frame, then the state-index (idx1) in the
  //     output FSA.
  //  -  If this state is currently being processed,
  //     then ... it is temporarily set to -2 - an idx012 into
  // a ragged tensor (TODO: name it.)
  // (note: final-states always have it set to -1,
  // as do states that are never reached.)
  // have -1 in them.
  Array1<int32_t> state_map_;

  // state_offsets_ is of dimension (fsas_.Dim0()), i.e. the number of
  // of separate FSAs we are top-sorting.  It represents the "next free number"
  // for the state numbering within this FSA.  These numbers represent idx1's
  // into the output FSA, not idx01's.
  Array1<int32_t> state_offsets_;

};

void IntersectDensePruned(FsaVec &a_fsas, DenseFsaVec &b_fsas, float beam,
                          int32_t max_active_states, int32_t min_active_states,
                          FsaVec *out, Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b) {
  MultiGraphDenseIntersect intersector(a_fsas, b_fsas, beam, max_active_states,
                                       min_active_states);

  intersector.FormatOutput(out, arc_map_a, arc_map_b);
}
}
