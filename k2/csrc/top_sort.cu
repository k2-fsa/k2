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
     Topological sorter object.  You should call TopSort() after
     constructing it.

       @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  TopSorter(FsaVec &fsas)
    : c_(fsas.Context()), fsas_(fsas) {
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
    Array1<int32_t> offsets;

    // This is a ragged array with 3 axes, where the first 2 axes are the same
    // as those of `states`.  It is: [fsa_index][list_of_states][list_of_arcs].
    // It contains the indexes of the destination states of those arcs.  These
    // are idx1's into the FSA that we will create at the output (as a special
    // case we use -1 when the destination state is the final-state).  We don't
    // need to store the corresponding arc-indexes (in the input FSA) because
    // these can be worked out from the information in `states`.  Note: these
    // are the same kinds of quantities as those in state_map_.
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
    ans->offsets = state_offsets_.Clone();  // currently contains all 0.
    int32_t num_fsas = NumFsas();

    const int32_t *fsa_row_splits1 = fsas_.RowSplits(1).Data();
    int32_t *state_offsets_data = state_offsets_.Data();

    auto lambda_set_num_initial_states =
      [=] __host__ __device__ (int32_t i) {
      // num_initial_states for this FSA is 1 if the input FSA had at least
      // one state.
      state_offsets_data[i] = (fsa_row_splits1[i+1] > fsa_row_splits1[i]);
    };
    Eval(c_, num_fsas, lambda_set_num_initial_states);


    Array1<int32_t> ans_states_row_splits1(c_, num_fsas + 1);
    ExclusiveSum(state_offsets_, &ans_states_row_splits1);
    int32_t num_elems = ans_states_row_splits1[num_fsas];
    Array1<int32_t> ans_states_row_ids1(c_, num_elems),
      ans_states_values(c_, num_elems);
    int32_t *ans_states_row_splits1_data = ans_states_row_splits1.Data(),
      *ans_states_row_ids1_data = ans_states_row_ids1.Data(),
      *ans_states_values_data = ans_states_values.Data();
    auto lambda_compute_states = [=] __host__ __device__ (int32_t i) {
      // i == fsa_idx0.
      int32_t this_row_split = ans_states_row_splits1_data[i],
          next_row_split = ans_states_row_splits1_data[i+1],
          num_states = next_row_split - this_row_split;
      if (num_states != 0) { // [then it's 1]..
        K2_CHECK_EQ(num_states, 1);
        ans_states_row_ids1_data[this_row_split] = i;
        // '+ 0' is because we want state 0 of that FSA (the start state).
        int32_t this_state_idx01 = fsa_row_splits1[i] + 0;
        ans_states_values_data[this_row_split] = this_state_idx01;
      }
    };
    Eval(c_, num_fsas, lambda_compute_states);

    ans->states = Ragged<int32_t>(RaggedShape2(&ans_states_row_splits1,
                                               &ans_states_row_ids1,
                                               num_elems),
                                  ans_states_values);
    return ans;
  }

  /*
    Assuming the `states` and `offsets` fields of cur_iter have already been set up,
    set up `cur_iter.dest_states` and returns a ragged array with 2 axes consisting
    of any new state-ids that were not previously encountered, that need to have
    arcs leaving them processed.  (These will become the `states` element of
    the next IterInfo, containing idx01's into fsas_).


   */
  std::unique_ptr<IterInfo> ComputeDestStates(IterInfo &cur) {
    int32_t num_states = cur.states.values.Dim();
    K2_CHECK_GT(num_states, 0);
    // `num_arcs` will is the number of of arcs leaing each of these states;
    // the extra element is so that we can turn it into row_splits.
    Array1<int32_t> num_arcs(c_, num_states + 1);
    int32_t *num_arcs_data = num_arcs.Data();

    const int32_t *states_data = cur.states.values.Data(),
        *states_row_ids = cur.states.shape.RowIds(1).Data(),
        *fsas_row_splits2 = fsas_.shape.RowSplits(2).Data();
    auto lambda_get_num_arcs = [=] __host__ __device__ (int32_t i) {
       int32_t state_idx01 = states_data[i],
             this_num_arcs = fsas_row_splits2[state_idx01+1] -
              fsas_row_splits2[state_idx01];
       num_arcs_data[i] = this_num_arcs;
    };
    auto &dest_state_row_splits2 = num_arcs;  // just an alias, for clarity.
    ExclusiveSum(num_arcs, &dest_state_row_splits2);
    int32_t tot_arcs = dest_state_row_splits2[num_states];

    cur.dest_states = Ragged<int32_t>(
        ComposeRaggedShapes(cur.states.shape,
                            RaggedShape2(&dest_state_row_splits2,
                                         nullptr, tot_arcs)),
        Array1<int32_t>(c_, tot_arcs));

    // the elements of `is_new` from 0 to tot_arcs - 1 will be set to 1 if this
    // arc "generates a new state" and 0 otherwise.  We'll use this to number
    // any newly created states.
    Array1<int32_t> is_new(c_, tot_arcs + 1, -1);
    const int32_t
      *dest_states_row_ids2_data = cur.dest_states.shape.RowIds(2).Data(),
      *dest_states_row_splits2_data = cur.dest_states.shape.RowSplits(2).Data(),
      *dest_states_row_ids1_data = cur.dest_states.shape.RowIds(1).Data(),
      *fsas_row_splits2_data = fsas_.shape.RowSplits(2).Data();
    const Arc *fsas_arcs_data = fsas_.values.Data();
    int32_t *dest_states_data = cur.dest_states.values.Data(),
        *state_map_data = state_map_.Data(),
        *is_new_data = is_new.Data();

    auto lambda_set_dest_states1 = [=] __host__ __device__ (int32_t arc_idx012) {
       // arc_idx012 is an index into cur.dest_states
       int32_t state_idx01 = dest_states_row_ids2_data[arc_idx012],
             arc_idx01x = dest_states_row_splits2_data[state_idx01],
           arc_idx2 = arc_idx012 - arc_idx01x;
       // `fsas_state_idx01` is an idx01 into `fsas_`.
       int32_t fsas_state_idx01 = states_data[state_idx01],
           fsas_arc_idx01x = fsas_row_splits2_data[fsas_state_idx01],
           fsas_arc_idx012 = fsas_arc_idx01x + arc_idx2,
           fsas_dest_state_idx01 = fsas_arcs_data[fsas_arc_idx012].dest_state,
           symbol = fsas_arcs_data[fsas_arc_idx012].symbol;
       if (symbol == -1) {
         // dest-state is final-state, treat this specially as it has to be
         // numbered last so we can't figure out its numbering yet.
         dest_states_data[arc_idx012] = -1;
       } else {
         int32_t cur_state_idx1 = state_map_data[fsas_dest_state_idx01];
         if (cur_state_idx1 >= 0) {  // This is an already-processed state,
                                     // i.e. we know its numbering within this
                                     // output FSA
           dest_states_data[arc_idx012] = cur_state_idx1;
         } else {
           // temporarily set it to -2 - arc_idx012; if there are multiple arcs
           // going to this state, one will (arbitrarily) "win" and we'll use
           // this to decide the numbering of the new state.
           state_map_data[fsas_dest_state_idx01] = -2 - arc_idx012;
           // temporarily set `dest_states_data` to a number < -1 from which
           // we can work out the corresponding index into `state_map`.
           dest_states_data[arc_idx012] = -2 - fsas_dest_state_idx01;
         }
       }
    };
    Eval(c_, tot_arcs, lambda_set_dest_states1);

    auto lambda_set_is_new = [=] __host__ __device__ (int32_t arc_idx012) {
       int32_t dest = dest_states_data[arc_idx012];
       // Previously, we have set the elements of `dest_states` to -2 -
       // arc_idx012, for arcs whose dest-state didn't already have an index
       // allocated.  That creates a race condition; this code detects
       // which arcs 'won' the race.
       is_new_data[arc_idx012] = (dest < -1 && arc_idx012 == -(dest + 2));
    };
    Eval(c_, tot_arcs, lambda_set_is_new);

    // Now `is_new` contains a numbering for the new states, that's
    // 0, 1, 2, .. for this iteration of the algorithm (covers all FSAs).
    ExclusiveSum(is_new, &is_new);
    int32_t num_new_states = is_new[tot_arcs];

    if (num_new_states == 0) {
      // We're done!
      return std::unique_ptr<IterInfo>(nullptr);
    }

    int32_t num_fsas = fsas_.Dim0();
    std::unique_ptr<IterInfo> ans = std::make_unique<IterInfo>();
    ans->offsets = state_offsets_.Clone();
    std::vector<int32_t> sizes = {num_fsas, num_new_states};
    ans->states = RaggedFromTotSizes<int32_t>(c_, sizes);
    
    int32_t *new_states_row_splits1_data = ans->states.RowSplits(1).Data();
    const int32_t *states_row_splits1_data = cur.states.RowSplits(1).Data(),
      *dest_state_row_splits2_data = dest_state_row_splits2.Data();

    auto lambda_set_new_states_row_splits1 = [=] __host__ __device__ (int32_t fsa_idx) -> void {
      int32_t state_idx0x = states_row_splits1_data[fsa_idx],
          arc_idx0xx = dest_state_row_splits2_data[state_idx0x],
          // `is_new_data` currently contains the exclusive-sum of the `is_new`
          // info.
          new_state_idx0x = is_new_data[arc_idx0xx];
      K2_CHECK_LE(new_state_idx0x, state_idx0x);
      new_states_row_splits1_data[fsa_idx] = new_state_idx0x;
    };
    Eval(c_, num_fsas + 1, lambda_set_new_states_row_splits1);

    // Set the row_ids of `ans->states`, its values, and the elements of
    // `dest_states.data` and `state_map_`.  Note: dest_states and state_map_
    // both contain the same type of value (an idx01 into fsas_), just indexed
    // differently.
    int32_t *new_states_row_ids1_data = ans->states.RowIds(1).Data(),
        *new_states_values_data = ans->states.values.Data(),
         *state_offsets_data = state_offsets_.Data();
    auto lambda_set_new_states_etc = [=] __host__ __device__ (int32_t arc_idx012) -> void {
       int32_t dest = dest_states_data[arc_idx012];
       if (dest < -1) {
         // The purpose of the next two lines is to work out fsa_idx0 (the FSA index)
         // so we can set the row_ids of `ans`
         int32_t state_idx01 = dest_states_row_ids2_data[arc_idx012],
             fsa_idx0 = dest_states_row_ids1_data[state_idx01];

         // note: this_arc_idx012 may or may not be equal to arc_idx012, depending
         // whether this arc's thread 'won the race' in a previous lambda.
         int32_t this_arc_idx012 = -(dest + 2);
         // `new_state_idx01` is an index into `ans` which will become
         // the next IterInfo::states.
         int32_t new_state_idx01 = is_new_data[this_arc_idx012];

         new_states_row_ids1_data[new_state_idx01] = fsa_idx0;
         int32_t fsa_dest_state_idx01 = fsas_arcs_data[arc_idx012].dest_state;
         new_states_values_data[new_state_idx01] = fsa_dest_state_idx01;
         int32_t new_state_idx0x = new_states_row_splits1_data[fsa_idx0],
             new_state_idx1 = new_state_idx01 - new_state_idx0x,
             new_state_idx1_global = state_offsets_data[fsa_idx0] +
                   new_state_idx1;
         // `global` means: not just on this iter, the idx1 within the returned
         // FSA.
         dest_states_data[arc_idx012] = new_state_idx1_global;
         state_map_data[fsa_dest_state_idx01] = new_state_idx1_global;
       }
    };
    Eval(c_, tot_arcs, lambda_set_new_states_etc);

    auto lambda_increase_state_offsets = [=] __host__ __device__ (int32_t fsa_idx0) -> void {
      int32_t num_new_states = new_states_row_splits1_data[fsa_idx0+1] -
         new_states_row_splits1_data[fsa_idx0];
      state_offsets_data[fsa_idx0] += num_new_states;
    };
    Eval(c_, num_fsas, lambda_increase_state_offsets);

    K2_DCHECK(ans->states.Validate());
    return ans;
  }

  /*
    Called after all iterations of the algorithm, it returns one last IterInfo
    object that represents the final-states.  (We previously avoided giving a
    number to the final-states because it is required to be last).  Each FSA
    will have one final-state if that input FSA had at least one state,
    otherwise no final-state.
  */
  std::unique_ptr<IterInfo> GetLastIterInfo() {
    std::unique_ptr<IterInfo> ans = std::make_unique<IterInfo>();
    ans->offsets = state_offsets_.Clone();

    int32_t num_fsas = fsas_.Dim0();
    Array1<int32_t> num_final_states(c_, num_fsas + 1);
    int32_t *num_final_states_data = num_final_states.Data();
    const int32_t *fsas_row_splits1 = fsas_.RowSplits(1).Data();
    auto lambda_set_num_final_states = [=] __host__ __device__ (int32_t fsa_idx0) {
      int32_t num_final_states = (fsas_row_splits1[fsa_idx0 + 1] >
                                  fsas_row_splits1[fsa_idx0]);
      num_final_states_data[fsa_idx0] = num_final_states;
    };
    Eval(c_, num_fsas, lambda_set_num_final_states);
    ExclusiveSum(num_final_states, &num_final_states);
    Array1<int32_t> &states_row_splits1 = num_final_states;
    int32_t tot_final_states = states_row_splits1[num_fsas];

    ans->states = Ragged<int32_t>(RaggedShape2(&states_row_splits1,
                                          nullptr, tot_final_states),
                                  Array1<int32_t>(c_, tot_final_states));
    const int32_t *ans_states_row_ids1_data = ans->states.RowIds(1).Data();
    int32_t *ans_states_data = ans->states.values.Data();
    auto lambda_set_ans_states = [=] __host__ __device__ (int32_t ans_states_idx01) {
       int32_t fsa_idx0 = ans_states_row_ids1_data[ans_states_idx01];
       // The final-state is the last state of the corresponding input FSA.
       // Note: this would actually only ever be accessed in checking code, as
       // this state has no arcs leaving it.
       int32_t fsas_state_idx01 = fsas_row_splits1[fsa_idx0 + 1] - 1;
       ans_states_data[ans_states_idx01] = fsas_state_idx01;
    };

    // These final states all have zero arcs leaving them so ans->dest_states has no
    // elements.
    Array1<int32_t> temp_row_splits(c_, tot_final_states + 1, 0),
      temp_row_ids(c_, 0);
    RaggedShape dest_states_shape_part = RaggedShape2(&temp_row_splits,
                                                      &temp_row_ids, 0);
    ans->dest_states = Ragged<int32_t>(ComposeRaggedShapes(ans->states.shape,
                                                           dest_states_shape_part),
                                       Array1<int32_t>(c_, 0));
    return ans;
  }

  // called after iters_ is set up, it formats the output.
  // must only be called once.
  FsaVec FormatOutput(Array1<int32_t> *arc_map) {
    int32_t num_fsas = fsas_.Dim0(),
        num_iters = iters_.size();

    std::vector<const RaggedShape*> dest_states_shapes(num_iters);
    std::vector<const int32_t*> dest_states_ptrs_vec(num_iters),
        states_row_splits1_ptrs_vec(num_iters),
        dest_states_row_splits2_ptrs_vec(num_iters),
        states_ptrs_vec(num_iters);

    for (size_t i = 0; i < num_iters; i++) {
      dest_states_shapes[i] = &(iters_[i]->dest_states.shape);
      dest_states_ptrs_vec[i] = iters_[i]->dest_states.values.Data();
      states_row_splits1_ptrs_vec[i] = iters_[i]->states.RowSplits(1).Data();
      dest_states_row_splits2_ptrs_vec[i] = iters_[i]->dest_states.RowSplits(2).Data();
      states_ptrs_vec[i] = iters_[i]->states.values.Data();
    }

    // Axes of `stacked_shape` will be: [fsa_index][iter][state][arc]
    RaggedShape stacked_shape = Stack(1, num_iters, &(dest_states_shapes[0]));
    // Remove the axis with index `iter`.
    Ragged<Arc> ans(RemoveAxis(stacked_shape, 1),
                    Array1<Arc>(c_, stacked_shape.NumElements()));

    Array1<const int32_t*> dest_states_ptrs(c_, dest_states_ptrs_vec),
      states_row_splits1_ptrs(c_, states_row_splits1_ptrs_vec),
      dest_states_row_splits2_ptrs(c_, dest_states_row_splits2_ptrs_vec),
      states_ptrs(c_, states_ptrs_vec);
    const int32_t **dest_states_ptrs_data = dest_states_ptrs.Data(),
      **states_row_splits1_ptrs_data = states_row_splits1_ptrs.Data(),
      **dest_states_row_splits2_ptrs_data = dest_states_row_splits2_ptrs.Data(),
      **states_ptrs_data = states_ptrs.Data();

    Arc *ans_arcs_data = ans.values.Data();
    Arc *fsas_arcs_data = fsas_.values.Data();

    const int32_t *stacked_row_ids3_data = stacked_shape.RowIds(3).Data(),
        *stacked_row_ids2_data = stacked_shape.RowIds(2).Data(),
        *stacked_row_ids1_data = stacked_shape.RowIds(1).Data(),
        *ans_row_splits1_data = ans.RowSplits(1).Data(),
        *stacked_row_splits1_data = stacked_shape.RowSplits(1).Data(),
        *stacked_row_splits2_data = stacked_shape.RowSplits(2).Data(),
        *stacked_row_splits3_data = stacked_shape.RowSplits(3).Data(),
        *fsas_row_splits2_data = fsas_.RowSplits(2).Data();

    int32_t *arc_map_data = nullptr;
    if (arc_map) {
      *arc_map = Array1<int32_t>(c_, ans.values.Dim());
      arc_map_data = arc_map->Data();
    }

    // stacked_shape is [fsa_index][iter][state][arc]
    auto lambda_set_arcs = [=] __host__ __device__ (int32_t arc_idx0123) {
      // unless otherwise stated these indexes are into `stacked`, which has one
      // more axis than `ans` (i.e. the `iter` axis).  so 0123==(fsa,iter,state,arc),
      // where the state and arc indexes are respectively
      // idx1's into IterInfo::{state,dest_state}
      // and idx2's into IterInfo::dest_state.
      int32_t state_idx012 = stacked_row_ids3_data[arc_idx0123],
          state_idx012x = stacked_row_splits3_data[state_idx012],
          // fsaiter here means `fsa and iteration combined`
          fsaiter_idx01 = stacked_row_ids2_data[state_idx012],
          fsaiter_idx01x = stacked_row_splits2_data[fsaiter_idx01],
          fsaiter_idx01xx = stacked_row_splits3_data[fsaiter_idx01x],
          fsa_idx0 = stacked_row_ids1_data[fsaiter_idx01],
          fsa_idx0x = stacked_row_splits1_data[fsa_idx0],
          // could also do fsa_idx0xx == stacked_row_splits2_data[fsa_idx0x],
          // but the next line can be executed sooner.
          fsa_idx0xx = ans_row_splits1_data[fsa_idx0],
          iter_idx1 = fsa_idx0x - fsaiter_idx01,
          state_idx12 = state_idx012 - fsa_idx0xx,  // == state_idx1 into `ans`.
          state_idx2 = state_idx012 - fsaiter_idx01x,  // state index given (iter and FSA)
          arc_idx23 = arc_idx0123 - fsaiter_idx01xx,   // arc index given (iter and FSA)
          arc_idx3 = arc_idx0123 - state_idx012x;

      // `states_fsa_idx0x` is the idx0x into the `states` and `dest_states`
      // tensors for the iteration corresponding to `iter_idx`, for this fsa_idx0.
      int32_t states_fsa_idx0x = states_row_splits1_ptrs_data[iter_idx1][fsa_idx0],
          dest_states_fsa_idx0xx =
                 dest_states_row_splits2_ptrs_data[iter_idx1][states_fsa_idx0x];

      // fsas_state_idx01 is an idx01 into fsas_, as stored in IterInfo::state..
      // note: `state_idx2` would be called `state_idx1` if we named it relative
      // to IterInfo::state, and `arc_idx3` would be called `arc_idx2` if we 
      // named it relative to fsas_.
      int32_t fsas_state_idx01 = states_ptrs_data[iter_idx1][
            states_fsa_idx0x + state_idx2],
          fsas_state_idx01x = fsas_row_splits2_data[fsas_state_idx01],
          fsas_arc_idx012 = fsas_state_idx01x + arc_idx3;

      // below, the idx0xx + idx23 is not an error though it may look like one...
      // the arc_idx23 into `stacked_shape` corresponds to arc_idx12 into
      // IterInfo::dest_states.
      // here `dest_state_idx1` is the state_idx1 into the output FsaVec.
      int32_t dest_state_idx1 = dest_states_ptrs_data[iter_idx1][
          dest_states_fsa_idx0xx + arc_idx23];

      Arc arc = fsas_arcs_data[fsas_arc_idx012];
      arc.src_state = state_idx12;  // state_idx12 into `stacked` == state_idx1 into `ans`
      if (dest_state_idx1 >= 0) {
        arc.dest_state = dest_state_idx1;  // is an idx1 into `ans`.
      } else {  // arc to final-state
        K2_CHECK_EQ(arc.dest_state, -1);
        // the following 2 variables are named w.r.t. the indexing into `ans`.
        int32_t ans_fsa_idx0x_next = stacked_row_splits1_data[fsa_idx0],
            ans_fsa_idx0x = stacked_row_splits1_data[fsa_idx0 + 1],
            ans_final_state_idx1 = (ans_fsa_idx0x_next - ans_fsa_idx0x) - 1;
        arc.dest_state = ans_final_state_idx1;
      }
      ans_arcs_data[arc_idx0123] = arc;
      if (arc_map_data != nullptr) {
        arc_map_data[arc_idx0123] = fsas_arc_idx012;
      }
    };
    Eval(c_, ans.values.Dim(), lambda_set_arcs);
    return ans;
  }

  /* Does the main work of top-sorting and returns the resulting FSAs.
        @param [out] arc_map  if non-NULL, the map from (arcs in output)
                     to (corresponding arcs in input) is written to here.
        @return   Returns the top-sorted FsaVec.  (Note: this may have
                 fewer states than the input if there were unreachable
                 states.)
   */
  FsaVec TopSort(Array1<int32_t> *arc_map) {
    iters_.push_back(GetFirstIterInfo());
    while (iters_.back() != nullptr) {
      iters_.push_back(ComputeDestStates(*iters_.back()));
    }
    iters_.back() = GetLastIterInfo();  // last one was nullptr.
    return FormatOutput(arc_map);
  }

  ContextPtr c_;
  FsaVec &fsas_;
  std::vector<std::unique_ptr<IterInfo> > iters_;

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


void TopSort(FsaVec &src, FsaVec *dest, Array1<int32_t> *arc_map) {
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK_LE(src.NumAxes(), 3);
  if (src.NumAxes() == 2) {
    // Turn single Fsa into FsaVec.
    const Fsa *srcs = &src;
    FsaVec src_vec = CreateFsaVec(1, &srcs),
      dest_vec;
    // Recurse..
    TopSort(src_vec, &dest_vec, arc_map);
    *dest = GetFsaVecElement(dest_vec, 0);
    return;
  }
  TopSorter sorter(src);
  *dest = sorter.TopSort(arc_map);
}

}

