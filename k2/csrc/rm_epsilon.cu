/**
 * @brief
 * remove epsilon
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/rm_epsilon.h"

namespace k2 {
void ComputeEpsilonSubset(FsaVec &src, FsaVec *dest, Array1<int32_t> *state_map,
                          Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(dest != nullptr && state_map != nullptr && arc_map != nullptr);
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr &c = src.Context();
  int32_t num_fsas = src.Dim0(), num_states = src.TotSize(1),
          num_arcs = src.TotSize(2);
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data(),
                *src_row_ids1_data = src.RowIds(1).Data(),
                *src_row_splits2_data = src.RowSplits(2).Data(),
                *src_row_ids2_data = src.RowIds(2).Data();
  const Arc *src_arcs_data = src.values.Data();
  // only keep states with epsilons entering them or leaving them
  Renumbering state_renumbering(c, num_states, true);
  // `state_renumbering.Keep()` has been initialized with 0s as we only set 1s
  // in below lambda.
  char *state_keep_data = state_renumbering.Keep().Data();
  // only keep epsilon arcs
  Renumbering arc_renumbering(c, num_arcs);
  char *arc_keep_data = arc_renumbering.Keep().Data();
  K2_EVAL(
      c, num_arcs, lambda_set_keep, (int32_t arc_idx012)->void {
        int32_t state_idx01 = src_row_ids2_data[arc_idx012],
                fsa_idx0 = src_row_ids1_data[state_idx01];
        // note start_state is idx0x
        int32_t start_state_this_fsa = src_row_splits1_data[fsa_idx0],
                start_state_next_fsa = src_row_splits1_data[fsa_idx0 + 1],
                first_arc_idx0xx_this_fsa =
                    src_row_splits2_data[start_state_this_fsa];
        Arc cur_arc = src_arcs_data[arc_idx012];
        // we only keep epsilon arcs
        arc_keep_data[arc_idx012] = (cur_arc.label == 0);
        if (cur_arc.label == 0) {
          // we keep any state who has entering or leaving epsilon arcs.
          int32_t cur_arc_src_state_idx01 =
              start_state_this_fsa + cur_arc.src_state;
          int32_t cur_arc_dest_state_idx01 =
              start_state_this_fsa + cur_arc.dest_state;
          state_keep_data[cur_arc_src_state_idx01] = 1;
          state_keep_data[cur_arc_dest_state_idx01] = 1;
        }
        // We always keep start state and final state for each non-empty fsa,
        // but only set `state_keep_data` when we process the first arc
        // of this fsa.
        if (start_state_next_fsa > start_state_this_fsa &&
            arc_idx012 == first_arc_idx0xx_this_fsa) {
          state_keep_data[start_state_this_fsa] = 1;
          state_keep_data[start_state_next_fsa - 1] = 1;
        }
      });

  Array1<int32_t> state_new_to_old = state_renumbering.New2Old();
  Array1<int32_t> state_old_to_new = state_renumbering.Old2New();
  Array1<int32_t> arc_new_to_old = arc_renumbering.New2Old();
  const int32_t *state_old_to_new_data = state_old_to_new.Data(),
                *arc_new_to_old_data = arc_new_to_old.Data();

  // get row_splits1 and row_ids of dest
  Array1<int32_t> dest_row_splits1 = state_old_to_new[src.RowSplits(1)];
  Array1<int32_t> dest_row_ids1 = src.RowIds(1)[state_new_to_old];

  // get arcs and row_ids2 of dest
  int32_t dest_num_arcs = arc_renumbering.NumNewElems();
  Array1<int32_t> dest_row_ids2 = Array1<int32_t>(c, dest_num_arcs);
  int32_t *dest_row_ids2_data = dest_row_ids2.Data();
  Array1<Arc> dest_arcs = Array1<Arc>(c, dest_num_arcs);
  Arc *dest_arcs_data = dest_arcs.Data();
  K2_EVAL(
      c, dest_num_arcs, lambda_set_dest_arc_and_row_ids2,
      (int32_t dest_arc_idx012)->void {
        int32_t src_arc_idx012 = arc_new_to_old_data[dest_arc_idx012],
                src_state_idx01 = src_row_ids2_data[src_arc_idx012],
                dest_state_idx01 = state_old_to_new_data[src_state_idx01];
        // set row_ids2 of dest
        dest_row_ids2_data[dest_arc_idx012] = dest_state_idx01;
        int32_t fsa_idx0 = src_row_ids1_data[src_state_idx01];
        int32_t src_start_state_idx0x = src_row_splits1_data[fsa_idx0],
                dest_start_state_idx0x =
                    state_old_to_new_data[src_start_state_idx0x];
        Arc cur_src_arc = src_arcs_data[src_arc_idx012];
        K2_DCHECK_EQ(cur_src_arc.label, 0);
        int32_t cur_src_arc_dest_state_idx01 =
            cur_src_arc.dest_state + src_start_state_idx0x;
        int32_t cur_dest_arc_dest_state_idx01 =
            state_old_to_new_data[cur_src_arc_dest_state_idx01];
        dest_arcs_data[dest_arc_idx012] =
            Arc(dest_state_idx01 - dest_start_state_idx0x,
                cur_dest_arc_dest_state_idx01 - dest_start_state_idx0x, 0,
                cur_src_arc.score);
      });
  *state_map = state_new_to_old;
  *arc_map = arc_new_to_old;
  RaggedShape dest_shape = RaggedShape3(&dest_row_splits1, &dest_row_ids1, -1,
                                        nullptr, &dest_row_ids2, dest_num_arcs);
  *dest = FsaVec(dest_shape, dest_arcs);
}

void ComputeNonEpsilonSubset(FsaVec &src, FsaVec *dest, Renumbering *state_map,
                             Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(dest != nullptr && state_map != nullptr && arc_map != nullptr);
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr &c = src.Context();
  int32_t num_fsas = src.Dim0(), num_states = src.TotSize(1),
          num_arcs = src.TotSize(2);
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data(),
                *src_row_ids1_data = src.RowIds(1).Data(),
                *src_row_splits2_data = src.RowSplits(2).Data(),
                *src_row_ids2_data = src.RowIds(2).Data();
  const Arc *src_arcs_data = src.values.Data();
  // only keep states with non-epsilons entering them or leaving them
  *state_map = Renumbering(c, num_states, true);
  Renumbering &state_renumbering = *state_map;
  // `state_renumbering.Keep()` has been initialized with 0s as we only set 1s
  // in below lambda.
  char *state_keep_data = state_renumbering.Keep().Data();
  // only keep non-epsilon arcs
  Renumbering arc_renumbering(c, num_arcs);
  char *arc_keep_data = arc_renumbering.Keep().Data();
  K2_EVAL(
      c, num_arcs, lambda_set_keep, (int32_t arc_idx012)->void {
        int32_t state_idx01 = src_row_ids2_data[arc_idx012],
                fsa_idx0 = src_row_ids1_data[state_idx01];
        // note start_state is idx0x
        int32_t start_state_this_fsa = src_row_splits1_data[fsa_idx0],
                start_state_next_fsa = src_row_splits1_data[fsa_idx0 + 1],
                first_arc_idx0xx_this_fsa =
                    src_row_splits2_data[start_state_this_fsa];
        Arc cur_arc = src_arcs_data[arc_idx012];
        // we only keep non-epsilon arcs
        arc_keep_data[arc_idx012] = (cur_arc.label != 0);
        if (cur_arc.label != 0) {
          // we keep any state who has entering or leaving non-epsilon arcs.
          int32_t cur_arc_src_state_idx01 =
              start_state_this_fsa + cur_arc.src_state;
          int32_t cur_arc_dest_state_idx01 =
              start_state_this_fsa + cur_arc.dest_state;
          state_keep_data[cur_arc_src_state_idx01] = 1;
          state_keep_data[cur_arc_dest_state_idx01] = 1;
        }
        // We always keep start state and final state for each non-empty fsa,
        // but only set `state_keep_data` when we process the first arc
        // of this fsa.
        if (start_state_next_fsa > start_state_this_fsa &&
            arc_idx012 == first_arc_idx0xx_this_fsa) {
          state_keep_data[start_state_this_fsa] = 1;
          state_keep_data[start_state_next_fsa - 1] = 1;
        }
      });

  Array1<int32_t> state_new_to_old = state_renumbering.New2Old();
  Array1<int32_t> state_old_to_new = state_renumbering.Old2New();
  Array1<int32_t> arc_new_to_old = arc_renumbering.New2Old();
  const int32_t *state_old_to_new_data = state_old_to_new.Data(),
                *arc_new_to_old_data = arc_new_to_old.Data();

  // get row_splits1 and row_ids of dest
  Array1<int32_t> dest_row_splits1 = state_old_to_new[src.RowSplits(1)];
  Array1<int32_t> dest_row_ids1 = src.RowIds(1)[state_new_to_old];

  // get arcs and row_ids2 of dest
  int32_t dest_num_arcs = arc_renumbering.NumNewElems();
  Array1<int32_t> dest_row_ids2 = Array1<int32_t>(c, dest_num_arcs);
  int32_t *dest_row_ids2_data = dest_row_ids2.Data();
  Array1<Arc> dest_arcs = Array1<Arc>(c, dest_num_arcs);
  Arc *dest_arcs_data = dest_arcs.Data();
  K2_EVAL(
      c, dest_num_arcs, lambda_set_dest_arc_and_row_ids2,
      (int32_t dest_arc_idx012)->void {
        int32_t src_arc_idx012 = arc_new_to_old_data[dest_arc_idx012],
                src_state_idx01 = src_row_ids2_data[src_arc_idx012],
                dest_state_idx01 = state_old_to_new_data[src_state_idx01];
        // set row_ids2 of dest
        dest_row_ids2_data[dest_arc_idx012] = dest_state_idx01;
        int32_t fsa_idx0 = src_row_ids1_data[src_state_idx01];
        int32_t src_start_state_idx0x = src_row_splits1_data[fsa_idx0],
                dest_start_state_idx0x =
                    state_old_to_new_data[src_start_state_idx0x];
        Arc cur_src_arc = src_arcs_data[src_arc_idx012];
        K2_DCHECK_NE(cur_src_arc.label, 0);
        int32_t cur_src_arc_dest_state_idx01 =
            cur_src_arc.dest_state + src_start_state_idx0x;
        int32_t cur_dest_arc_dest_state_idx01 =
            state_old_to_new_data[cur_src_arc_dest_state_idx01];
        dest_arcs_data[dest_arc_idx012] =
            Arc(dest_state_idx01 - dest_start_state_idx0x,
                cur_dest_arc_dest_state_idx01 - dest_start_state_idx0x,
                cur_src_arc.label, cur_src_arc.score);
      });
  *arc_map = arc_new_to_old;
  RaggedShape dest_shape = RaggedShape3(&dest_row_splits1, &dest_row_ids1, -1,
                                        nullptr, &dest_row_ids2, dest_num_arcs);
  *dest = FsaVec(dest_shape, dest_arcs);
}

void MapFsaVecStates(FsaVec &src, Array1<int32_t> &state_row_splits,
                     Array1<int32_t> &state_row_ids,
                     const Array1<int32_t> &state_map, FsaVec *dest,
                     Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(dest != nullptr && arc_map != nullptr);
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr c = GetContext(src, state_row_splits, state_row_ids, state_map);
  int32_t num_fsas = src.Dim0(), src_num_states = src.TotSize(1),
          src_num_arcs = src.TotSize(2);
  K2_CHECK_EQ(src_num_states, state_map.Dim());
  int32_t dest_num_states = state_row_ids.Dim();
  const int32_t *state_map_data = state_map.Data();
#if !defined(NDEBUG)
  ValidateRowSplitsAndIds(state_row_splits, state_row_ids);
  // check if state_map is valid or not
  Array1<int32_t> state_map_status(c, 1, 0);
  int32_t *state_map_status_data = state_map_status.Data();
  K2_EVAL(
      c, src_num_states, lambda_check_state_map,
      (int32_t src_state_idx01)->void {
        int32_t dest_state_idx01 = state_map_data[src_state_idx01];
        if (dest_state_idx01 != -1 &&
            (dest_state_idx01 < 0 || dest_state_idx01 >= dest_num_states))
          state_map_status_data[0] = 1;
      });
  K2_CHECK_EQ(state_map_status[0], 0);
#endif

  const int32_t *src_row_splits1_data = src.RowSplits(1).Data(),
                *src_row_ids1_data = src.RowIds(1).Data(),
                *src_row_splits2_data = src.RowSplits(2).Data(),
                *src_row_ids2_data = src.RowIds(2).Data();
  const Arc *src_arcs_data = src.values.Data();
  // only keep arcs whose src_state and dest_state are both kept in dest
  Renumbering arc_renumbering(c, src_num_arcs);
  char *arc_keep_data = arc_renumbering.Keep().Data();
  K2_EVAL(
      c, src_num_arcs, lambda_set_keep, (int32_t arc_idx012)->void {
        int32_t state_idx01 = src_row_ids2_data[arc_idx012],
                fsa_idx0 = src_row_ids1_data[state_idx01],
                start_state_idx0x = src_row_splits1_data[fsa_idx0];
        const Arc &cur_arc = src_arcs_data[arc_idx012];
        // noted src_state and dest_state are idx1, we need to convert them to
        // idx01 as state_map is indexed by idx01.
        arc_keep_data[arc_idx012] =
            (state_map_data[start_state_idx0x + cur_arc.src_state] != -1 &&
             state_map_data[start_state_idx0x + cur_arc.dest_state] != -1);
      });

  Array1<int32_t> arc_old_to_new = arc_renumbering.Old2New();
  Array1<int32_t> arc_new_to_old = arc_renumbering.New2Old();
  const int32_t *arc_new_to_old_data = arc_new_to_old.Data();
  // Suppose we generate an FsaVec `temp` with only arcs kept above, then
  // `temp_row_splits2` is the row_splits2 of `temp`. Let dest_state_idx01 =
  // state_map[temp_state_idx01] (note temp_state_idx01 is state_idx01 in temp,
  // it equals to the corresponding state_idx01 in src (i.e. src_state_idx01)
  // as we did not delete any state in `src` while generating temp), then
  // arc_nums leaving dest_state_idx01 will equal to arc_nums leaving
  // temp_state_idx01.
  Array1<int32_t> temp_row_splits2 = arc_old_to_new[src.RowSplits(2)];
  Array1<int32_t> temp_row_ids2 = src.RowIds(2)[arc_new_to_old];
  // init dest_row_splits2 with `0` as we only set arc_nums for those kept
  // states in `state_map`, note the number of states in dest may be smaller or
  // larger than num_states in src.
  Array1<int32_t> dest_row_splits2(c, dest_num_states + 1, 0);
  int32_t *dest_row_splits2_data = dest_row_splits2.Data();
  const int32_t *temp_row_splits2_data = temp_row_splits2.Data(),
                *temp_row_ids2_data = temp_row_ids2.Data();
  K2_EVAL(
      c, src_num_states, lambda_set_arc_nums_in_each_state,
      (int32_t temp_state_idx01)->void {
        int32_t dest_state_idx01 = state_map_data[temp_state_idx01];
        if (dest_state_idx01 != -1)
          dest_row_splits2_data[dest_state_idx01] =
              temp_row_splits2_data[temp_state_idx01 + 1] -
              temp_row_splits2_data[temp_state_idx01];
      });
  ExclusiveSum(dest_row_splits2.Arange(0, dest_num_states), &dest_row_splits2);

  int32_t dest_num_arcs = arc_renumbering.NumNewElems();
  // TODO(haowen): remove below check after testing
  K2_DCHECK_EQ(dest_num_arcs, dest_row_splits2.Back());
  Array1<Arc> dest_arcs(c, dest_num_arcs);
  *arc_map = Array1<int32_t>(c, dest_num_arcs);
  Arc *dest_arcs_data = dest_arcs.Data();
  int32_t *arc_map_data = arc_map->Data();
  const int32_t *dest_row_splits1_data = state_row_splits.Data(),
                *dest_row_ids1_data = state_row_ids.Data();
  K2_EVAL(
      c, dest_num_arcs, lambda_set_dest_arc_and_arc_map,
      (int32_t temp_arc_idx012)->void {
        // note `temp_arc_idx012` is arc_idx012 corresponds to FsaVec `temp` we
        // declare in comments above.
        int32_t temp_state_idx01 = temp_row_ids2_data[temp_arc_idx012],
                temp_arc_idx01x = temp_row_splits2_data[temp_state_idx01],
                temp_arc_idx2 = temp_arc_idx012 - temp_arc_idx01x;
        int32_t src_arc_idx012 = arc_new_to_old_data[temp_arc_idx012];
        const Arc &src_arc = src_arcs_data[src_arc_idx012];
        // noted below we use temp_state_idx01 as src_state_idx01 as they
        // equal to each other (recalling that we didn't delete any state in
        // `src` when generating `temp`.
        int32_t fsa_idx0 = src_row_ids1_data[temp_state_idx01],
                src_state_idx0x = src_row_splits1_data[fsa_idx0];
        // TODO(haowen): remove below check after testing
        int32_t src_arc_src_state_idx01 = src_state_idx0x + src_arc.src_state;
        K2_DCHECK_EQ(src_arc_src_state_idx01, temp_state_idx01);
        int32_t src_arc_dest_state_idx01 = src_state_idx0x + src_arc.dest_state;
        int32_t dest_state_idx01 = state_map_data[temp_state_idx01],
                dest_arc_idx01x = dest_row_splits2_data[dest_state_idx01],
                dest_arc_idx012 = dest_arc_idx01x + temp_arc_idx2,
                dest_fsa_idx0 = dest_row_ids1_data[dest_state_idx01],
                dest_state_idx0x = dest_row_splits1_data[dest_fsa_idx0];
        dest_arcs_data[dest_arc_idx012] =
            Arc(dest_state_idx01 - dest_state_idx0x,
                state_map_data[src_arc_dest_state_idx01] - dest_state_idx0x,
                src_arc.label, src_arc.score);
        arc_map_data[dest_arc_idx012] = src_arc_idx012;
      });
  RaggedShape dest_shape =
      RaggedShape3(&state_row_splits, &state_row_ids, -1, &dest_row_splits2,
                   nullptr, dest_num_arcs);
  *dest = FsaVec(dest_shape, dest_arcs);
}

void ComputeEpsilonClosure(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                           Ragged<int32_t> *arc_map) {
  K2_LOG(FATAL) << "Not Implemented!";
}

void ComputeEpsilonClosureOneIter(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                                  Ragged<int32_t> *arc_map) {
  K2_LOG(FATAL) << "Not Implemented!";
}

void RemoveEpsilonsIterativeTropical(FsaVec &src_fsa, FsaVec *dest_fsa,
                                     Ragged<int32_t> *arc_map_out) {
  ContextPtr c = GetContext(src_fsa, *dest_fsa, *arc_map_out);
  Array1<int32_t> epsilons_state_map, epsilons_arc_map;
  FsaVec epsilon_fsa;
  ComputeEpsilonSubset(src_fsa, &epsilon_fsa, &epsilons_state_map,
                       &epsilons_arc_map);

  FsaVec epsilon_fsa_closure;
  Ragged<int32_t> epsilon_closure_arc_map;
  ComputeEpsilonClosure(epsilon_fsa, &epsilon_fsa_closure,
                        &epsilon_closure_arc_map);
  // make epsilon_closure_arc_map refer back to 'src_fsa'.
  epsilon_closure_arc_map.values =
      epsilons_arc_map[epsilon_closure_arc_map.values];

  FsaVec non_epsilon_fsa;
  Renumbering non_epsilon_state_renumbering;
  Array1<int32_t> non_epsilon_arc_map;
  ComputeNonEpsilonSubset(src_fsa, &non_epsilon_fsa,
                          &non_epsilon_state_renumbering, &non_epsilon_arc_map);

  // Combine the info in epsilons_state_map
  // and non_epsilon_state_renumbering.Old2New(),
  // to create a state-map from the states in (epsilon_fsa or
  // epsilon_fsa_closure) to those in non_epsilon_fsa (or -1 for those states
  // which are not present in non_epsilon_fsa.
  Array1<int32_t> epsilon_to_noneps_state_map(c, epsilon_fsa.TotSize(1));
  // [lambda here to set epsilon_to_noneps_state_map)

  // `epsilon_closure_mapped` will have (a subset of) the arcs of the
  // epsilon-closure FSA in the same numbering as those of non_epsilon_fsa.
  FsaVec epsilon_closure_mapped;
  Array1<int32_t> epsilon_closure_mapped_arc_map1;
  MapFsaVecStates(epsilon_fsa_closure, non_epsilon_fsa.RowSplits(1),
                  non_epsilon_fsa.RowSplits(1), epsilon_to_noneps_state_map,
                  &epsilon_closure_mapped, &epsilon_closure_mapped_arc_map1);

  // arc_map from epsilon_closure_mapped back to `src_fsa`.
  Ragged<int32_t> epsilon_closure_mapped_arc_map =
      Index(epsilon_closure_arc_map, epsilon_closure_mapped_arc_map1);

  // we will need the row_splits of this to get the number of non-epsilon arcs
  // entering each state in non_epsilon_fsa.
  Ragged<int32_t> non_epsilon_incoming_arcs =
      GetIncomingArcs(non_epsilon_fsa, GetDestStates(non_epsilon_fsa, true));

  // epsilon_prec_renumbering will remap the arcs in epsilon_closure_mapped to
  // the subset of arcs that we'll combine with the *preceding* arc (i.e.
  // entering their src_state).
  Renumbering epsilon_prec_renumbering(c, epsilon_closure_mapped.NumElements());
  char *epsilon_prec_renumbering_keep_data =
      epsilon_prec_renumbering.Keep().Data();

  Array1<int32_t> epsilon_num_foll_arcs(
      c, epsilon_closure_mapped.NumElements() + 1);
  int32_t *epsilon_num_foll_arcs_data = epsilon_num_foll_arcs.Data();

  // Lambda:
  //   For each epsilon arc in epsilon_closure_mapped, we'll decide whether to
  //   combine it with *following* non-epsilon arcs or *preceding* non-epsilon
  //   arcs. We combine it with *following* non-epsilon arcs if it is leaving
  //   from the start-state or if the num-non-epsilon-arcs leaving its dest
  //   state is less than the num-non-epsilon-arcs entering its src state.
  //
  //   If we decided to combine it with following non-epsilon arcs then we set
  //   epsilon_num_foll_arcs_data to the number of non-epsilon-arcs leaving
  //   the dest-state, and set epsilon_prec_renumbering_keep_data to 0.
  //   Else (combining with preceding arcs) we set epsilon_num_foll_arcs_data to
  //   0 and set epsilon_prec_renumbering_keep_data to 1.

  // `combined_foll` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs following them.
  FsaVec combined_foll;
  Ragged<int32_t> combined_foll_arc_map;
  {  // This block will set combined_foll and combined_foll_arc_map
    ExclusiveSum(epsilon_num_foll_arcs, &epsilon_num_foll_arcs);
    Array1<int32_t> &foll_row_splits = epsilon_num_foll_arcs;
    int32_t num_arcs = foll_row_splits.Back();
    Array1<int32_t> foll_row_ids(c, num_arcs);
    RowSplitsToRowIds(foll_row_splits, &foll_row_ids);
    // This shape just says, for each arc in epsilon_closure_mapped
    // that is to be combined with following arcs, how many following
    // arcs it is combined with (else 0).
    RaggedShape foll_shape =
        RaggedShape2(&foll_row_splits, &foll_row_ids, num_arcs);

    // foll_non_eps_arc_idx will be set in the lambda to the arc-index within
    // non_epsilon_fsa of the following arc which we're combining this epsilon
    // arc with
    Array1<int32_t> foll_non_eps_arc_idx(c, num_arcs);
    Array1<Arc> arcs(c, num_arcs);
    {
      // lambda that sets foll_non_eps_arc_idx and arcs.
    }

    RaggedShape epsilon_closure_combined_with_fool =
        ComposeRaggedShapes(epsilon_closure_mapped.shape, foll_shape);
    RaggedShape foll_fsa_shape =
        RemoveAxis(epsilon_closure_combined_with_fool, 2);

    combined_foll = FsaVec(foll_fsa_shape, arcs);

    Ragged<int32_t> epsilon_closure_mapped_arc_map_foll =
        Index(epsilon_closure_mapped_arc_map, foll_row_ids);
    combined_foll_arc_map =
        AddSuffixToRagged(epsilon_closure_mapped_arc_map_foll,
                          non_epsilon_arc_map[foll_non_eps_arc_idx]);
  }

  FsaVec epsilon_closure_prec =
      SubsampleRagged(epsilon_closure_mapped, epsilon_prec_renumbering);
  Ragged<int32_t> epsilon_closure_prec_arc_map =
      Index(epsilon_closure_mapped_arc_map, epsilon_prec_renumbering.New2Old());

  // `combined_prec` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs preceding them.
  FsaVec combined_prec;
  Ragged<int32_t> combined_prec_arc_map;

  {  //  This block will set combined_prec and combined_prec_arc_map
    //  nonepsilon_num_foll_eps[i] tells us, for each arc in non_epsilon_fsa,
    // how many epsilon arcs leave the state that follows it.
    Array1<int32_t> nonepsilon_num_foll_eps(c,
                                            non_epsilon_fsa.NumElements() + 1);
    // will set nonepsilon_num_foll_eps using a lambda that uses the row-ids
    // (etc.) of epsilon_closure_prec.

    ExclusiveSum(nonepsilon_num_foll_eps, &nonepsilon_num_foll_eps);

    // The basic logic of this block will be similar to the block in
    // which we set combined_foll, except the order of non-epsilon and
    // epsilon are different.  By indexing/numbering things by the *first*
    // of the two arcs (rather than, say, by the epsilon) we ensure
    // that there is no need to re-sort the arcs, which could be slow.
  }

  // NOTE: important that order matches with `arc_maps` below.
  FsaVec *vecs[3] = {&combined_foll, &combined_prec, &non_epsilon_fsa};
  int32_t axis = 2;

  // Caution: currently Append() does not support axis > 1; but actually this is
  // not a fundamental limitation because it doesn't interact non-trivially with
  // the earlier axes, as long as they are identical among all inputs.  For
  // instance, we could do RemoveAxis() to remove axis 0 of all the inputs, then
  // Append() with axis 1, then do ComposeRaggedShapes() to combine with axis 0
  // of the inputs (we could obtain that by doing RemoveAxis(one_of_the_inputs,
  // 2)).  We just require that the earlier axes all be the same, which they are
  // here.  I'm not saying we need such a recursive implementation, necessarily;
  // only that there is not a fundamental reason why Append() can't work in this
  // case.
  Array1<uint32_t> arcs_merge_map;
  FsaVec dest_fsa_unsorted = Append(axis, 2, vecs, &arcs_merge_map);

  Ragged<int32_t> non_epsilon_arc_map_ragged(
      RegularRaggedShape(c, non_epsilon_fsa.NumElements(),
                         non_epsilon_fsa.NumElements()),
      non_epsilon_arc_map);

  Ragged<int32_t> dest_unsorted_arc_map;
  if (arc_map_out != nullptr) {
    // This block creates 'dest_unsorted_arc_map' which combines
    // combined_foll_arc_map, combined_prec_arc_map and
    // non_epsilon_arc_map_ragged

    // NOTE: important that order matches with `vecs` above.
    Ragged<int32_t> *arc_maps_full[] = {&combined_foll_arc_map,
                                        &combined_prec_arc_map,
                                        &non_epsilon_arc_map_ragged};

    dest_unsorted_arc_map = Merge(3, arc_maps_full, arcs_merge_map, nullptr);
  }

  int32_t props = GetFsaBasicProperties(dest_fsa_unsorted);
  if (props & kFsaPropertiesArcSorted == 0) {
    // `dest_fsa_unsorted` was not arc sorted.
    Array1<int32_t> arcsort_arc_map;
    ArcSort(dest_fsa_unsorted, dest_fsa, &arcsort_arc_map);
    if (arc_map_out != nullptr)
      *arc_map_out = Index(dest_unsorted_arc_map, arcsort_arc_map);
  } else {
    *dest_fsa = dest_fsa_unsorted;
    if (arc_map_out != nullptr) *arc_map_out = dest_unsorted_arc_map;
  }
}

}  // namespace k2
