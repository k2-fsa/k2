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



/*  Returns a renumbered version of the FsaVec `src`.
      @param [in] src    An FsaVec, assumed to be valid, with NumAxes() == 3
      @param [in] order  An ordering of states in `src`.  Does not necessarily
                    have to contain all states in `src`.  The FSAs must
                    not be reordered by this (i.e. if we get the old fsa-index
                    of elements of `order`, they must still be non-decreasing).
      @param [out] arc_map  If non-NULL, this will be set to a new Array1 that
                   maps from arcs in the returned FsaVec to the original arc-index
                   in `fsas`.
      @return  Returns renumbered FSA.

  NOTE (Dan): we could declare this in a header to make it easier to test.
*/
  static FsaVec RenumberFsaVec(FsaVec &fsas, const Array1<int32_t> &order,
                               Array1<int32_t> *arc_map) {
  Context c = fsas.Context();
  K2_CHECK_LE(order.NumElements(), fsas.NumElements());
  Array1<int32_t> old2new_map(c, fsas.TotSize(1));
  if (order.NumElements() != fsas.NumElements()) {
    old2new_map = -1;
  }
  int32_t new_num_states = order.Dim(), num_fsas = fsas.Dim0();
  Array1<int32_t> num_arcs(c, new_num_states + 1);
  const int32_t *order_data = order.Data(),
    *fsas_row_splits1_data = fsas.RowSplits(1).Data();
    *fsas_row_splits2_data = fsas.RowSplits(2).Data();
  int32_t *old2new_data = old2new_map.Data(),
    *num_arcs_data = num_arcs.Data();
  auto lambda_set_old2new_and_num_arcs = [=] __host__ __device__ (int32_t new_state_idx01) -> void {
    int32_t old_state_idx01 = order_data[new_state_idx01];
    old2new_data[old_state_idx01] = new_state_idx01;
    int32_t num_arcs = fsas_row_splits2_data[old_state_idx01+1] -
                       fsas_row_splits2_data[old_state_idx01];
    num_arcs_data[new_state_idx01] = num_arcs;
  };
  Eval(c, new_num_states, lambda_set_old2new_and_num_arcs);
  

  Array1<int32_t> new_row_splits1, new_row_ids1;
  if (order.Dim() == fsas.Dim()) {
    new_row_splits1 = fsas.RowSplits(1);
    new_row_ids1 = fsas.RowIds(1);
  } else {
    new_row_ids1 = fsas.RowIds(1)[order];
    new_row_splits1 = fsas.Array1<int32_t>(c, num_fsas + 1);
    RowIdsToRowSplits(new_row_splits1, &new_row_ids1);
  }

  ExclusiveSum(num_arcs, &num_arcs);
  RaggedShape ans_shape = Ragged3(&new_row_splits1, &new_row_ids1, -1,
                                  &num_arcs, nullptr, -1);
  const int32_t *ans_row_ids2 = ans_shape.RowIds(2).Data(),
    *ans_row_ids1 = ans_shape.RowIds(1).Data(),
    *ans_row_splits1 = ans_shape.RowSplits(1).Data(),
    *ans_row_splits2 = ans_shape.RowSplits(2).Data();
  int32_t ans_num_arcs = ans_shape.NumElements();
  Array1<Arc> ans_arcs(c, ans_num_arcs);
  int32_t *arc_map_data;
  if (arc_map) {
    *arc_map = Array1<int32_t>(c, ans_num_arcs);
    arc_map_data = arc_map->Data();
  } else {
    arc_map_data = nullptr;
  }
  const Arc *fsas_arcs = fsas.values.Data();
  Arc *ans_arcs_data = ans_arcs.Data();
  auto lambda_set_arcs = [=] __host__ __device__ (int32_t ans_idx012) -> void {
    int32_t ans_idx01 = ans_row_ids2[ans_idx012], // state index
      ans_idx01x = ans_row_splits2[ans_idx01],
      ans_idx0 = ans_row_ids1[ans_idx01],  // FSA index
      ans_idx0x = ans_row_splits1[ans_idx0],
      ans_idx1 = ans_idx01 - ans_idx0x,
      ans_idx2 = ans_idx012 - ans_idx01x,
      fsas_idx01 = order_data[ans_idx01],
      fsas_idx01x = fsas_row_splits2_data[fsas_idx01],
      fsas_idx012 = fsas_idx01x + ans_idx2;
    Arc arc = fsas_arcs[fsas_idx012];
    int32_t fsas_src_idx1 = arc.src_state,
       fsas_dest_idx1 = arc.dest_state,
       fsas_idx0x = fsas_row_splits1_data[ans_idx0],
       fsas_src_idx01 = fsas_idx0x + fsas_src_idx1,
       fsas_dest_idx01 = fsas_idx01 + fsas_dest_idx1;
    K2_CHECK_EQ(old2new_data[fsas_src_idx01], ans_idx01);
    int32_t ans_dest_idx01 = old2new_data[fsas_dest_idx01],
      ans_dest_idx1 = ans_dest_idx01 - ans_idx0x;
    arc.src_state = ans_idx1;
    ans.dest_state = ans_dest_idx1;
    ans_arcs_data[ans_idx012] = arc;
    if (arc_map_data)
      arc_map_data[ans_idx012] = fsas_idx012;
  };
  Eval(c, ans_shape.NumElements(), lambda_set_arcs);
  return FsaVec(ans_shape, ans_arcs);
}


  
class TopSorter {
 public:
  /**
     Topological sorter object.  You should call TopSort() after
     constructing it.  Please see TopSort() declaration in header for
     high-level overview of the algorithm.

       @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  TopSorter(FsaVec &fsas)
    : c_(fsas.Context()), fsas_(fsas) {
    K2_CHECK_EQ(fsas_.NumAxes(), 3);
    int32_t num_fsas = fsas_.shape.TotSize(0),
      num_states = fsas_.shape.TotSize(1);

    // Get in-degrees of states.
    int32_t num_arcs = fsas_.NumElements();
    Array1<int32_t> dest_states_idx01(c_, num_arcs);
    int32_t *dest_states_idx01_data = dest_states_idx01.Data();
    Array1<int32_t> arc_to_start_state_idx01 =
      fsas_.RowSplits(1)[fsas_.RowIds(1)[fsas_.RowIds(2)]];
    
    const int32_t *fsas_row_splits1 = fsas_.RowSplits(1).Data(),
      *fsas_row_ids1 = fsas_.RowIds(1).Data(),
      *fsas_row_ids2 = fsas_.RowIds(2).Data();
    const Arc *arcs_data = fsas_.values.Data();
    auto lambda_set_dest_states_idx01 = [=] __host__ __device__ (int32_t arc_idx012) -> void {
      int32_t state_idx01 = fsas_row_ids2[arc_idx012],
        fsa_idx0 = fsas_row_ids1[state_idx01],
        start_state_idx0x = fsas_row_splits1[fsa_idx0],
        dest_state_idx1 = arcs[arc_idx012].dest_state,
        dest_state_idx01 = start_state_idx0x + dest_state_idx;
      if (dest_state_idx01 == state_idx01) {
        // If it was a self-loop, pretend it was to the final-state.  This will
        // allow the algorithm to work in the presence of self-loops (it's actually
        // equivalent to not having the arc at all)
        int32_t final_state_idx01 = fsas_row_splits1[fsa_idx0 + 1] - 1;
        dest_state_idx01 = final_state_idx01;
      }
      dest_states_idx01_data[arc_idx012] = dest_state_idx01;
    };
    Eval(c_, num_arcs, lambda_set_dest_states_idx01);

    dest_states_ = Ragged<int32_t>(fsas_.shape, dest_states_idx01);

    state_in_degree_ = GetCounts(dest_states_, num_arcs);
    int32_t *state_in_degree_data = state_in_degree_.Data();

    // Increment the in-degree of final-states
    auto lambda_inc_final_state_in_degree = [=] __host__ __device__ (int32_t fsa_idx0) -> void {
      int32_t final_state = fsas_row_splits1[fsa_idx0 + 1] - 1;
    }
    Eval(c_, num_fsas, lambda_inc_final_state_in_degree);
  }



  int32_t NumFsas() { return fsas_.Dim0(); }


  /*
    Return the ragged array containing the states active on the 1st iteration of
    the algorithm.  These just correspond to the start-states of all 
    the FSAs, and also the final-states for all FSAs in which final-states
    had in-degree zero (no arcs entering them).

    Note: in the originally published algorithm we start with all states
    that have in-degree zero, but in the context of this toolkit there
    is (I believe) no use in states that aren't accessible from the start
    state, so we remove them.
   */
  std::unique_ptr<Ragged<int32_t> > GetInitialBatch() {
    // Initialize it with a list of all states that currently have zero
    // in-degree.
    int32_t num_states = state_in_degree_.Dim();
    Renumbering state_renumbering(num_states);
    // NOTE: this is not very optimal given that we're keeping only a small
    // number of states, but at this point I dont want to optimize too heavily.
    // The (dest_states_data[i] != i) part is to avoid self-loops.
    char *keep_data = state_renumbering.Keep().Data();
    const int32_t *state_in_degree_data = state_in_degree_.values.Data(),
      *fsas_row_ids1 = fsas_.RowIds(1),
      *fsas_row_splits1 = fsas_.RowSplits(1);
    auto lambda_set_keep = [=] __host__ __device__ (int32_t fsas_idx01) -> void {
      // Make this state a member of the initial batch if it has zero in-degree
      // (note: this won't include final states, as we incremented their in-degree
      // to avoid them appearing here.)
      keep_data[fsas_idx01] = state_in_degree_data[fsas_idx01] == 0;
    };
    Eval(c_, num_states, lambda_set_keep);

    Array1<int32_t> first_iter_values = renumbering.New2Old();
    Array1<int32_t> first_iter_row_ids = fsas_.RowIds(1)[first_iter_values];
    return std::make_unique<Ragged<int32_t> >(
      RaggedShape2(nullptr, &first_iter_row_ids,
                   first_iter_row_ids.Dim()),
      first_iter_values);
  }

  /*
    Computes the next batch of states 
         @param [in] cur_states  Ragged array with 2 axes, containing state-indexes
                              (idx01) into fsas_.  These are states which already
                               have in-degree 0
         @return   Returns the states which, after processing.
   */
  std::unique_ptr<Ragged<int32_t> > GetNextBatch(Ragged<int32_t> &cur_states) {
    // Process arcs leaving all states in `cur`

    // First figure out how many arcs leave each state.
    Array1<int32_t> arcs_per_state(cur_states.NumElements() + 1);
    int32_t *arcs_per_state_data = arcs_per_state.Data();
    const int32_t *states_data = cur_states.values.Data(),
      *fsas_row_splits2_data = fsas_.RowSplits(2).Data();
    auto lambda_set_arcs_per_state = [=] __host__ __device__ (int32_t states_idx01) {
      int32_t fsas_idx01 = states_data[state_idx01],
        num_arcs = fsas_row_splits2_data[fsas_idx01 + 1] -
          fsas_row_splits2_data[fsas_idx01];
      arcs_per_state_data[states_idx01] = num_arcs;
    };
    Eval(c_, cur_states.NumElements(), lambda_set_arcs_per_state);
    ExclusiveSum(arcs_per_state, &arcs_per_state);

    RaggedShape arcs_shape = ComposeRaggedShape(cur_states.shape,
                                                RaggedShape2(&arcs_per_state, nullptr, -1));

    // Each arc that generates a new state (i.e. for which arc_renumbering.Keep[i] == true)
    // will write the state-id to here (as an idx01 into fsas_).  Other elements will
    // be undefined.
    Array<int32_t> next_iter_states(c_, arcs_shape.NumElements());

    // We'll be figuring out which of these arcs leads to a state that now has
    // in-degree 0.  (If >1 arc goes to such a state, only one will 'win',
    // arbitrarily).
    Renumbering arc_renumbering(arcs_shape.NumElements());

    const int32_t *arcs_row_ids2 = arcs_shape.RowIds(2).Data(),
      *arcs_row_splits2 = arcs_shape.RowSplits(2).Data(),
      *fsas_row_splits1 = fsas_.RowSplits(1).Data(),
      *fsas_row_splits2 = fsas_.RowSplits(2).Data(),
      *dest_states_data = dest_states_.values.Data();
    char *keep_arc_data = arc_renumbering.Keep().Data();
    int32_t *state_in_degree = state_in_degree_.Data(),
      *next_iter_states_data = next_iter_states.Data();
    auto lambda_set_arc_renumbering = __host__ __device__ (int32_t arcs_idx012) {
      // note: the prefix `arcs_` means it is an idxXXX w.r.t. `arcs_shape`.
      // the prefix `fsas_` means the variable is an idxXXX w.r.t. `fsas_`.
      int32_t arcs_idx01 = arcs_row_ids2[arcs_idx012],
        arcs_idx01x = arcs_row_splits2[arcs_idx01],
        arcs_idx2 = arcs_idx012 - arcs_idx01x,
        fsa_idx0 = arcs_row_ids1[arcs_idx01],
        fsas_idx01 = states_data[arcs_idx012],  // a state index
        fsas_idx01x = fsas_row_splits2[fsas_idx01],
        fsas_idx012 = fsas_idx012 + arcs_idx2,
        fsas_dest_state_idx01 = dest_states_data[fsas_idx012];

      if ((keep_arc_data = AtomicDecAndCompareZero(state_in_degree + fsas_dest_state))) {
        next_iter_states_data[arcs_idx012] = fsas_idx01;
      }
    };

    Array1<int32_t> new2old_map = arc_renumbering.New2Old();
    if (new2old_map.Dim() == 0) {
      // There are no new states.  This means we terminated.  We'll check from calling code
      // that we processed all arcs.
      return nullptr;
    }
    // `new_states` will contain state-ids which are idx01's into `fsas_`.
    Array1<int32_t> new_states = next_iter_states_data[new2old_map];
    Array1<int32_t> new_states_row_ids(new_states.Dim());  // will map to FSA index
    const int32_t *new2old_map_data = new2old_map.Data();
    int32_t *new_states_row_ids_data = new_states_row_ids.Data();
    auto lambda_set_row_ids = [=] __host__ __device__ (int32_t new_state_idx) -> void {
      int32_t old_arcs_idx012 = new2old_map_data[new_state_idx],
        old_arcs_idx01 = arcs_row_splits2[old_arcs_idx012],  // state index
        old_arcs_idx0 = arcs_row_splits1[old_arcs_idx01]; // FSA index
      new_states_row_ids_data[new_state_idx] = old_arcs_idx0;
    }
    
    std::unique_ptr<Ragged<int32_t> > ans = std::make_unique<Ragged<int32_t> >(
       RaggedShape2(nullptr, &new_states_row_ids_data, -1),
       new_states);
    // The following will ensure the answer has deterministic numbering
    SortSublists(ans.get(), nullptr);
    return ans;
  }

  /*
    Returns the final batch of states.  This will include all final-states that
    existed in the original FSAs, i.e. at most one per input.  We treat them
    specially because we can't afford the final-state to not be the last state
    (this is only an issue because we support input where not all states were
    reachable from the start state).
   */
  std::unique_ptr<Ragged<int32_t> > GetFinalBatch() {
    int32_t num_fsas = NumFsas() + 1;
    int32_t *fsas_row_splits1 = fsas_.RowSplits(1);
    Array<int32_t> has_final_state(num_fsas + 1);
    int32_t *has_final_state_data = has_final_state.Data();
    auto lambda_set_has_final_state = [=] __host__ __device__ (int32_t i) -> void {
      int32_t split = fsas_row_splits1[i], next_split = fsas_row_splits1[i+1];
      has_final_state_data[i] = (next_split > split);
    };
    Eval(c_, num_fsas, lambda_set_has_final_state);
    ExclusiveSum(has_final_state, &has_final_state);

    int32_t n = has_final_state[num_fsas];
    std::unique_ptr<Ragged<int32_t> > ans = std::make_unique<Ragged<int32_t> >(
      RaggedShape2(&has_final_state, nullptr, n),
      Array1<int32_t>(c_, n));
    int32_t *ans_data = ans->values.Data();
    const int32_t *ans_row_ids_data = ans->values.RowIds();
    auto lambda_set_final_state = [=] __host__ __device__ (int32_t i) -> void {
      int32_t fsa_idx0 = ans_row_ids_data[i],
      final_state = fsas_row_splits1[fsa_idx0+1] - 1;
      // If the following fails, it likely means an input FSA was invalid (e.g.
      // had exactly one state, which is not allowed).  Either that, or a code
      // error.
      K2_CHECK_GT(final_state, fsas_row_splits1[fsa_idx0]);
      ans_data[i] = final_state;
    };
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

    std::vector<std::unique_ptr<Ragged<int32_t> > > iters;
    iters.push_back(GetFirstBatch());
    while (iters.back() != nullptr)
      iters.push_back(GetNextBatch(*iters.back()));
    // note: below, we're overwriting nullptr.
    iters.back() = GetFinalBatch();

    // Need raw pointers for Stack().
    std::vector<Ragged<int32_t>* > iters_ptrs(iters.size());
    for (size_t i = 0; < iters.size(); i++)
      iters_ptrs[i] = iters[i].get();
    RaggedShape all_states = Append(1, static_cast<int32_t>(iters.size()),
                                    iters.data());
    K2_CHECK_EQ(all_states.NumElements(), fsas_.TotSize(1))
      << "likely code error";

    return RenumberFsaVec(fsas_, all_states, arc_map);
  }

  ContextPtr c_;
  FsaVec &fsas_;

  // The remaining in-degree of each state (state_in_degree_.Dim() ==
  // fsas_.NumElements()), i.e. number of incoming arcs (except those from states
  // that were already processed).
  Array1<int32_t> state_in_degree_;

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


