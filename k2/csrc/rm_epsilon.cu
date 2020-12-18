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

namespace {
// Will be used in `ComputeEpsilonClosureOneIter` below
struct ArcComparer {
  __host__ __device__ __forceinline__ bool operator()(
      const k2::Arc &lhs, const k2::Arc &rhs) const {
    // Compares `dest_state` first, then `score` (in ascending order);
    if (lhs.dest_state != rhs.dest_state)
      return lhs.dest_state < rhs.dest_state;
    else
      return lhs.score > rhs.score;
  }
};

}  // namespace

namespace k2 {
/*
  Given epsilon-closure FSA `epsilon_fsa_closure` and non-epsilon FSA
  `non_epsilon_fsa`, this function will create an FSA `epsilon_closure_mapped`
  which have all arcs in `epsilon_fsa_closure` whose src_state and dest_state
  are both kept in `non_epsilon_fsa`.

     @param [in] epsilon_fsa_closure   Epsilon-closure FSA, returned
                      by ComputeEpsilonClosure.
     @param [in] epsilon_closure_state_map  The state map from
                      `epsilon_fsa_closure` to the original Fsa
                      (i.e. the input Fsa of `ComputeEpsilonSubset` and
                      `ComputeNonEpsilonSubset`).
                      Noted we'll call the original Fsa `src` below.
     @param [in] epsilon_closure_arc_map  The arc map from
                      `epsilon_fsa_closure` to `src` Fsa.
     @param [in] non_epsilon_fsa  Non-epsilon Fsa, returned by
                       ComputeNonEpsilonSubset.
     @param [in] non_epsilon_state_renumbering  The renumbering object from
                       states in `non_epsilon_fsa` to states in `src`.
     @param [out] epsilon_closure_mapped  The output Fsa, will have (a subset
                       of) the arcs of `epsilon_fsa_closure` whose src_state
                       and dest_state are both kept in `non_epsilon_fsa`.
                       Will have the same states numbering as `non_epsilon_fsa`.
    @param [out] epsilon_closure_mapped_arc_map  The arc map from
                       `epsilon_closure_mapped` to `src`.
*/
static void GetEpsilonClosureMapped(
    FsaVec &epsilon_fsa_closure,
    const Array1<int32_t> &epsilon_closure_state_map,
    Ragged<int32_t> &epsilon_closure_arc_map, FsaVec &non_epsilon_fsa,
    Renumbering &non_epsilon_state_renumbering, FsaVec *epsilon_closure_mapped,
    Ragged<int32_t> *epsilon_closure_mapped_arc_map) {
  ContextPtr &c = epsilon_fsa_closure.Context();
  // Combine the info in epsilon_closure_state_map and
  // non_epsilon_state_renumbering.Old2New(), to create a state-map from the
  // states in epsilon_fsa_closure to those in non_epsilon_fsa (or -1 for those
  // states which are not present in non_epsilon_fsa.
  Array1<int32_t> epsilon_to_noneps_state_map(c,
                                              epsilon_fsa_closure.TotSize(1));
  int32_t *epsilon_to_noneps_state_map_data =
      epsilon_to_noneps_state_map.Data();
  const int32_t *epsilons_state_map_data = epsilon_closure_state_map.Data(),
                *non_epsilon_state_map_old_to_new_data =
                    non_epsilon_state_renumbering.Old2New().Data();
  const char *non_epsilon_state_renumbering_keep_data =
      non_epsilon_state_renumbering.Keep().Data();
  K2_EVAL(
      c, epsilon_to_noneps_state_map.Dim(),
      lambda_set_epsilon_to_noneps_state_map,
      (int32_t epsilon_fsa_state_idx01)->void {
        int32_t src_fsa_state_idx01 =
            epsilons_state_map_data[epsilon_fsa_state_idx01];
        epsilon_to_noneps_state_map_data[epsilon_fsa_state_idx01] =
            non_epsilon_state_renumbering_keep_data[src_fsa_state_idx01] == 1
                ? non_epsilon_state_map_old_to_new_data[src_fsa_state_idx01]
                : -1;
      });
  Array1<int32_t> epsilon_closure_mapped_arc_map1;
  MapFsaVecStates(epsilon_fsa_closure, non_epsilon_fsa.RowSplits(1),
                  non_epsilon_fsa.RowIds(1), epsilon_to_noneps_state_map,
                  epsilon_closure_mapped, &epsilon_closure_mapped_arc_map1);
  // arc_map from epsilon_closure_mapped back to `src` Fsa.
  *epsilon_closure_mapped_arc_map =
      Index(epsilon_closure_arc_map, epsilon_closure_mapped_arc_map1);
}

/*
  For each epsilon arc in epsilon_closure_mapped, we'll decide whether to
  combine it with *following* non-epsilon arcs or *preceding* non-epsilon
  arcs. We combine it with *following* non-epsilon arcs if it is leaving
  from the start-state or if the num-non-epsilon-arcs leaving its dest
  state is less than the num-non-epsilon-arcs entering its src state.

  If we decided to combine it with following non-epsilon arcs then we set
  epsilon_num_foll_arcs_data to the number of non-epsilon-arcs leaving
  the dest-state, and set epsilon_prec_renumbering_keep_data to 0.
  Else (combining with preceding arcs) we set epsilon_num_foll_arcs_data to
  0 and set epsilon_prec_renumbering_keep_data to 1.

     @param [in] epsilon_closure_mapped  Fsa returned by
                       GetEpsilonClosureMapped. It has the same states numbering
                       as `non_epsilon_fsa`.
     @param [in] non_epsilon_fsa  Non-epsilon Fsa, returned by
                       ComputeNonEpsilonSubset.
     @param [out] epsilon_prec_renumbering  For each epsilon arc `i` in
                       `epsilon_closure_mapped`, if we decide to combine it with
                       preceding non-epsilon arcs, we'll set
                       epsilon_prec_renumbering.Keep()[i] to 1;
                       Otherwise set it to 0.
     @param [out] foll_shape At exit foll_shape.NumAxes() == 2. For each arc `i`
                       in epsilon_closure_mapped that is to be combined with
                       following arcs, foll_shape.RowSplits(1)[i+1] -
                       foll_shape.RowSplits(1)[i] is the number of following
                       arcs it is combined with.
*/
static void DecideCombineWithFollowingOrPreceding(
    FsaVec &epsilon_closure_mapped, FsaVec &non_epsilon_fsa,
    Renumbering *epsilon_prec_renumbering, RaggedShape *foll_shape) {
  ContextPtr &c = epsilon_closure_mapped.Context();
  // we will need the row_splits of this to get the number of non-epsilon arcs
  // entering each state in non_epsilon_fsa.
  Ragged<int32_t> non_epsilon_incoming_arcs =
      GetIncomingArcs(non_epsilon_fsa, GetDestStates(non_epsilon_fsa, true));

  int32_t epsilon_closure_mapped_num_arcs =
      epsilon_closure_mapped.NumElements();
  *epsilon_prec_renumbering = Renumbering(c, epsilon_closure_mapped_num_arcs);
  char *epsilon_prec_renumbering_keep_data =
      epsilon_prec_renumbering->Keep().Data();

  Array1<int32_t> epsilon_num_foll_arcs(c, epsilon_closure_mapped_num_arcs + 1);
  int32_t *epsilon_num_foll_arcs_data = epsilon_num_foll_arcs.Data();
  const int32_t *epsilon_closure_mapped_row_splits1_data =
                    epsilon_closure_mapped.RowSplits(1).Data(),
                *epsilon_closure_mapped_row_ids1_data =
                    epsilon_closure_mapped.RowIds(1).Data(),
                *epsilon_closure_mapped_row_ids2_data =
                    epsilon_closure_mapped.RowIds(2).Data(),
                *non_epsilon_fsa_row_splits2_data =
                    non_epsilon_fsa.RowSplits(2).Data(),
                *non_epsilon_incoming_arcs_row_splits2_data =
                    non_epsilon_incoming_arcs.RowSplits(2).Data();
  const Arc *epsilon_closure_mapped_arcs_data =
                epsilon_closure_mapped.values.Data(),
            *non_epsilon_fsa_arcs_data = non_epsilon_fsa.values.Data();
  K2_EVAL(
      c, epsilon_closure_mapped_num_arcs,
      lambda_set_combined_with_preceding_or_following,
      (int32_t arc_idx012)->void {
        int32_t state_idx01 = epsilon_closure_mapped_row_ids2_data[arc_idx012],
                fsa_idx0 = epsilon_closure_mapped_row_ids1_data[state_idx01],
                start_state_idx0x =
                    epsilon_closure_mapped_row_splits1_data[fsa_idx0];
        const Arc &cur_arc = epsilon_closure_mapped_arcs_data[arc_idx012];
        int32_t dest_state_idx01 = start_state_idx0x + cur_arc.dest_state;
        // note `epsilon_closure_mapped` shares the same row_splits1 and
        // with `non_epsilon_fsa` and `non_epsilon_incoming_arcs`, so the order
        // of states (i.e. state_idx01) in them are same.
        int32_t num_non_eps_arcs_entering_src_state =
            non_epsilon_incoming_arcs_row_splits2_data[state_idx01 + 1] -
            non_epsilon_incoming_arcs_row_splits2_data[state_idx01];
        int32_t num_non_eps_arcs_leaving_dest_state =
            non_epsilon_fsa_row_splits2_data[dest_state_idx01 + 1] -
            non_epsilon_fsa_row_splits2_data[dest_state_idx01];
        if (state_idx01 == start_state_idx0x ||
            num_non_eps_arcs_leaving_dest_state <
                num_non_eps_arcs_entering_src_state) {
          // We'll combine this arc with *following* non-epsilon arcs
          epsilon_prec_renumbering_keep_data[arc_idx012] = 0;
          epsilon_num_foll_arcs_data[arc_idx012] =
              num_non_eps_arcs_leaving_dest_state;

        } else {
          epsilon_prec_renumbering_keep_data[arc_idx012] = 1;
          epsilon_num_foll_arcs_data[arc_idx012] = 0;
        }
      });
  ExclusiveSum(epsilon_num_foll_arcs.Arange(0, epsilon_closure_mapped_num_arcs),
               &epsilon_num_foll_arcs);
  Array1<int32_t> &foll_row_splits = epsilon_num_foll_arcs;
  int32_t num_arcs = foll_row_splits.Back();
  Array1<int32_t> foll_row_ids(c, num_arcs);
  RowSplitsToRowIds(foll_row_splits, &foll_row_ids);
  *foll_shape = RaggedShape2(&foll_row_splits, &foll_row_ids, num_arcs);
}

/*
  Combine (a subset of) epsilon arcs in `epsilon_closure_mapped` with
  following non-epsilon arcs in `non_epsilon_fsa`.
     @param [in] epsilon_closure_mapped  Fsa returned by
                       GetEpsilonClosureMapped. It has the same states numbering
                       as `non_epsilon_fsa`.
     @param [in] epsilon_closure_mapped_arc_map  Arc map from arcs in
                       epsilon_clousre_mapped to arcs in the original Fsa, i.e.
                       the src Fsa we call `RemoveEpsilonsIterativeTropical`
                       with.
     @param [in] non_epsilon_fsa  Non-epsilon Fsa, returned by
                       ComputeNonEpsilonSubset.
     @param [in] non_epsilon_arc_map  Arc map from arcs in non_epsilon_fsa to
                       arcs in the original Fsa.
     @param [in] foll_shape Returned by DecideCombineWithFollowingOrPreceding.
                       For each arc `i` in epsilon_closure_mapped that is to be
                       combined with following arcs,
                       foll_shape.RowSplits(1)[i+1] - foll_shape.RowSplits(1)[i]
                       is the number of following arcs it is combined with.
     @param [out] combined_foll The combined Fsa, with the same state numbering
                       as `non_epsilon_fsa`, containing the arcs which arose by
                       combining epsilon arcs with non-epsilon arcs following
                       them.
     @param [out] combined_foll_arc_map The arc map of `combined_foll`, from
                       arcs idx012 in `combined_foll` to the original Fsa.
*/
static void CombineWithFollowingNonEpsilonArcs(
    FsaVec &epsilon_closure_mapped,
    Ragged<int32_t> &epsilon_closure_mapped_arc_map, FsaVec &non_epsilon_fsa,
    const Array1<int32_t> &non_epsilon_arc_map, RaggedShape &foll_shape,
    FsaVec *combined_foll, Ragged<int32_t> *combined_foll_arc_map) {
  ContextPtr &c = non_epsilon_fsa.Context();
  const int32_t *epsilon_closure_mapped_row_splits1_data =
                    epsilon_closure_mapped.RowSplits(1).Data(),
                *epsilon_closure_mapped_row_ids1_data =
                    epsilon_closure_mapped.RowIds(1).Data(),
                *epsilon_closure_mapped_row_ids2_data =
                    epsilon_closure_mapped.RowIds(2).Data(),
                *non_epsilon_fsa_row_splits2_data =
                    non_epsilon_fsa.RowSplits(2).Data();
  const Arc *epsilon_closure_mapped_arcs_data =
                epsilon_closure_mapped.values.Data(),
            *non_epsilon_fsa_arcs_data = non_epsilon_fsa.values.Data();

  int32_t num_arcs = foll_shape.NumElements();
  // foll_non_eps_arc_idx will be set in the lambda to the arc-index within
  // non_epsilon_fsa of the following arc which we're combining this epsilon
  // arc with
  Array1<int32_t> foll_non_eps_arc_idx(c, num_arcs);
  int32_t *foll_non_eps_arc_idx_data = foll_non_eps_arc_idx.Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();
  const int32_t *foll_row_splits_data = foll_shape.RowSplits(1).Data(),
                *foll_row_ids_data = foll_shape.RowIds(1).Data();
  K2_EVAL(
      c, num_arcs, lambda_set_foll_non_eps_arc_idx_and_arcs,
      (int32_t foll_idx01)->void {
        int32_t foll_idx0 = foll_row_ids_data[foll_idx01],
                foll_idx0x = foll_row_splits_data[foll_idx0],
                foll_idx1 = foll_idx01 - foll_idx0x;
        // foll_idx0 is arc_idx012 in epsilon_clousre_mapped
        int32_t state_idx01 = epsilon_closure_mapped_row_ids2_data[foll_idx0],
                fsa_idx0 = epsilon_closure_mapped_row_ids1_data[state_idx01],
                start_state_idx0x =
                    epsilon_closure_mapped_row_splits1_data[fsa_idx0];
        const Arc &cur_arc = epsilon_closure_mapped_arcs_data[foll_idx0];
        int32_t dest_state_idx01 = start_state_idx0x + cur_arc.dest_state;
        // foll_idx1 is the arc we'll combined with in those leaving arcs of
        // dest_state.
        int32_t leaving_arcs_idx01x =
                    non_epsilon_fsa_row_splits2_data[dest_state_idx01],
                leaving_arcs_idx012 = leaving_arcs_idx01x + foll_idx1;
        const Arc &cur_combined_arc =
            non_epsilon_fsa_arcs_data[leaving_arcs_idx012];
        arcs_data[foll_idx01] =
            Arc(cur_arc.src_state, cur_combined_arc.dest_state,
                cur_combined_arc.label, cur_arc.score + cur_combined_arc.score);
        foll_non_eps_arc_idx_data[foll_idx01] = leaving_arcs_idx012;
      });
  RaggedShape epsilon_closure_combined_with_fool =
      ComposeRaggedShapes(epsilon_closure_mapped.shape, foll_shape);
  RaggedShape foll_fsa_shape =
      RemoveAxis(epsilon_closure_combined_with_fool, 2);

  *combined_foll = FsaVec(foll_fsa_shape, arcs);

  Ragged<int32_t> epsilon_closure_mapped_arc_map_foll =
      Index(epsilon_closure_mapped_arc_map, foll_shape.RowIds(1));
  *combined_foll_arc_map =
      AddSuffixToRagged(epsilon_closure_mapped_arc_map_foll,
                        non_epsilon_arc_map[foll_non_eps_arc_idx]);
}

/*
  Combine epsilon arcs in `epsilon_closure_prec` with preceding non-epsilon arcs
  in `non_epsilon_fsa`.
     @param [in] epsilon_closure_prec  Contains arcs in `epsilon_closure_mapped`
                       returned by GetEpsilonClosureMapped that will be combined
                       with preceding non-epsilon arcs in `non_epsilon_fsa`.
                       It has the same states numbering as `non_epsilon_fsa`.
     @param [in] epsilon_closure_prec_arc_map  Arc map from arcs in
                       epsilon_clousre_prec to arcs in the original Fsa, i.e.
                       the src Fsa we call `RemoveEpsilonsIterativeTropical`
                       with.
     @param [in] non_epsilon_fsa  Non-epsilon Fsa, returned by
                       ComputeNonEpsilonSubset.
     @param [out] combined_prec The combined Fsa, with the same state numbering
                       as `non_epsilon_fsa`, containing the arcs which arose by
                       combining epsilon arcs with non-epsilon arcs preceding
                       them.
     @param [out] epsilon_closure_prec_arc_map_prec  NumAxes() == 2, for each
                       arc in `combined_prec`, it gives those arc map info
                       from `epsilon_closure_prec_arc_map`. Noted the caller
                       need to combined it with arc map info from
                       `non_epsilon_fsa` to get the complete arc map info
                       for `combined_prec`.
     @param [out] foll_shape At exit foll_shape.NumAxes() == 2. For each arc `i`
                       in non_epsilon_fsa that is to be combined with
                       following epsilon arcs, foll_shape.RowSplits(1)[i+1] -
                       foll_shape.RowSplits(1)[i] is the number of following
                       epsilon arcs it is combined with. If user has already get
                       `non_epsilon_arc_map` (i.e. arc map from
                       `non_epsilon_fsa` to the original fsa), then
                       `non_epsilon_arc_map[foll_shape.RowIds(1)]` will be the
                       part of arc map information from `non_epsilon_fsa` for
                       each arc in `combined_prec`, combined it with
                       `epsilon_closure_prec_arc_map`, user will get the
                       complete arc map info for `combined_prec`.
*/
static void CombineWithPrecedingNonEpsilonArcs(
    FsaVec &epsilon_closure_prec, Ragged<int32_t> &epsilon_closure_prec_arc_map,
    FsaVec &non_epsilon_fsa, FsaVec *combined_prec,
    Ragged<int32_t> *epsilon_closure_prec_arc_map_prec,
    RaggedShape *foll_shape) {
  ContextPtr &c = non_epsilon_fsa.Context();
  //  non_epsilon_num_foll_eps[i] tells us, for each arc in non_epsilon_fsa,
  // how many epsilon arcs leave the dest state of this arc.
  int32_t non_epsilon_fsa_num_arcs = non_epsilon_fsa.NumElements();
  Array1<int32_t> non_epsilon_num_foll_eps(c, non_epsilon_fsa_num_arcs + 1);
  int32_t *non_epsilon_num_foll_eps_data = non_epsilon_num_foll_eps.Data();
  const Arc *non_epsilon_fsa_arcs_data = non_epsilon_fsa.values.Data(),
            *epsilon_closure_prec_arcs_data =
                epsilon_closure_prec.values.Data();
  const int32_t *non_epsilon_fsa_row_ids2_data =
                    non_epsilon_fsa.RowIds(2).Data(),
                *non_epsilon_fsa_row_splits2_data =
                    non_epsilon_fsa.RowSplits(2).Data(),
                *non_epsilon_fsa_row_ids1_data =
                    non_epsilon_fsa.RowIds(1).Data(),
                *non_epsilon_fsa_row_splits1_data =
                    non_epsilon_fsa.RowSplits(1).Data(),
                *epsilon_closure_prec_row_splits2_data =
                    epsilon_closure_prec.RowSplits(2).Data();
  K2_EVAL(
      c, non_epsilon_fsa_num_arcs, lambda_set_non_epsilon_num_foll_eps,
      (int32_t arc_idx012)->void {
        int32_t dest_state_idx1 =
            non_epsilon_fsa_arcs_data[arc_idx012].dest_state;
        int32_t state_idx01 = non_epsilon_fsa_row_ids2_data[arc_idx012],
                fsa_idx0 = non_epsilon_fsa_row_ids1_data[state_idx01],
                start_state_idx0x = non_epsilon_fsa_row_splits1_data[fsa_idx0],
                dest_state_idx01 = start_state_idx0x + dest_state_idx1;
        // note epsilon_closure_prev shares the same row_splits1 and row_ids1
        // with non_epsilon_fsa, so states numbering in them are same.
        non_epsilon_num_foll_eps_data[arc_idx012] =
            epsilon_closure_prec_row_splits2_data[dest_state_idx01 + 1] -
            epsilon_closure_prec_row_splits2_data[dest_state_idx01];
      });

  ExclusiveSum(non_epsilon_num_foll_eps.Arange(0, non_epsilon_fsa_num_arcs),
               &non_epsilon_num_foll_eps);
  Array1<int32_t> &foll_row_splits = non_epsilon_num_foll_eps;
  int32_t num_arcs = foll_row_splits.Back();
  Array1<int32_t> foll_row_ids(c, num_arcs);
  RowSplitsToRowIds(foll_row_splits, &foll_row_ids);
  // This shape just says, for each arc in non_epsilon_fsa
  // that is to be combined with following epsilon arcs, how many following
  // epsilon arcs it is combined with (else 0).
  *foll_shape = RaggedShape2(&foll_row_splits, &foll_row_ids, num_arcs);

  // foll_eps_arc_idx will be set in the lambda to the arc-index within
  // epsilon_closure_prec of the following epsilon arc which we're combining
  // this non-epsilon arc (in non_epsilon_fsa) with
  Array1<int32_t> foll_eps_arc_idx(c, num_arcs);
  int32_t *foll_eps_arc_idx_data = foll_eps_arc_idx.Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();
  const int32_t *foll_row_splits_data = foll_row_splits.Data(),
                *foll_row_ids_data = foll_row_ids.Data();
  K2_EVAL(
      c, num_arcs, lambda_set_foll_eps_arc_idx_and_arcs,
      (int32_t foll_idx01)->void {
        int32_t foll_idx0 = foll_row_ids_data[foll_idx01],
                foll_idx0x = foll_row_splits_data[foll_idx0],
                foll_idx1 = foll_idx01 - foll_idx0x;
        // foll_idx0 is arc_idx012 in non_epsilon_fsa
        int32_t state_idx01 = non_epsilon_fsa_row_ids2_data[foll_idx0],
                fsa_idx0 = non_epsilon_fsa_row_ids1_data[state_idx01],
                start_state_idx0x = non_epsilon_fsa_row_splits1_data[fsa_idx0];
        const Arc &cur_arc = non_epsilon_fsa_arcs_data[foll_idx0];
        int32_t dest_state_idx01 = start_state_idx0x + cur_arc.dest_state;
        // foll_idx1 is the arc we'll combined with in those leaving arcs of
        // dest_state_idx01 in epsilon_closure_prec.
        int32_t leaving_arcs_idx01x =
                    epsilon_closure_prec_row_splits2_data[dest_state_idx01],
                leaving_arcs_idx012 = leaving_arcs_idx01x + foll_idx1;
        const Arc &cur_combined_arc =
            epsilon_closure_prec_arcs_data[leaving_arcs_idx012];
        arcs_data[foll_idx01] =
            Arc(cur_arc.src_state, cur_combined_arc.dest_state, cur_arc.label,
                cur_arc.score + cur_combined_arc.score);
        foll_eps_arc_idx_data[foll_idx01] = leaving_arcs_idx012;
      });
  RaggedShape epsilon_closure_combined_with_prec =
      ComposeRaggedShapes(non_epsilon_fsa.shape, *foll_shape);
  RaggedShape prec_fsa_shape =
      RemoveAxis(epsilon_closure_combined_with_prec, 2);

  *combined_prec = FsaVec(prec_fsa_shape, arcs);

  *epsilon_closure_prec_arc_map_prec =
      Index(epsilon_closure_prec_arc_map, foll_eps_arc_idx);
}

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
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(closure_fsa != nullptr && arc_map != nullptr);
  K2_CHECK_EQ(epsilon_fsa.NumAxes(), 3);

  // We repeatedly call ComputeEpsilonClosureOneIter() until there is no further
  // change in the FsaVec (this can be by simple comparison on arcs vector,
  // since thanks to sorting the order is deterministic).
  Array1<Arc> epsilon_fsa_arcs = epsilon_fsa.values;
  ComputeEpsilonClosureOneIter(epsilon_fsa, closure_fsa, arc_map);
  Array1<Arc> closure_fsa_arcs = closure_fsa->values;
  // note function `Equal` for Array1 requires the input two arrays have the
  // same size.
  while (epsilon_fsa_arcs.Dim() != closure_fsa_arcs.Dim() ||
         !Equal(epsilon_fsa_arcs, closure_fsa_arcs)) {
    epsilon_fsa_arcs = closure_fsa_arcs;
    FsaVec cur_iter_closure_fsa;
    Ragged<int32_t> cur_iter_arc_map;
    ComputeEpsilonClosureOneIter(*closure_fsa, &cur_iter_closure_fsa,
                                 &cur_iter_arc_map);
    closure_fsa_arcs = cur_iter_closure_fsa.values;
    *closure_fsa = cur_iter_closure_fsa;
    *arc_map = ComposeArcMaps(*arc_map, cur_iter_arc_map);
  }

  // delete all epsilon self cycle.
  int32_t num_arcs = closure_fsa->NumElements();
  const Arc *arcs_data = closure_fsa->values.Data();
  ContextPtr &c = closure_fsa->Context();
  Renumbering arc_renumbering(c, num_arcs);
  char *arc_keep_data = arc_renumbering.Keep().Data();
  K2_EVAL(
      c, num_arcs, lambda_set_keep_arc_data, (int32_t arc_idx012)->void {
        const Arc &cur_arc = arcs_data[arc_idx012];
        arc_keep_data[arc_idx012] = (cur_arc.src_state != cur_arc.dest_state);
      });
  *closure_fsa = SubsampleRagged(*closure_fsa, arc_renumbering);
  *arc_map = Index(*arc_map, arc_renumbering.New2Old());
}

void ComputeEpsilonClosureOneIter(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                                  Ragged<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(closure_fsa != nullptr && arc_map != nullptr);
  FsaVec &src = epsilon_fsa;
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr &c = src.Context();
  int32_t num_fsas = src.Dim0(), num_states = src.TotSize(1),
          src_num_arcs = src.TotSize(2);
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data(),
                *src_row_ids1_data = src.RowIds(1).Data(),
                *src_row_splits2_data = src.RowSplits(2).Data(),
                *src_row_ids2_data = src.RowIds(2).Data();
  Arc *src_arcs_data = src.values.Data();
  // For any epsilon cycle, e.g. s1->s1, if its score is positive,
  // we'll abort the program as positive score means we'll get infinity weight
  // under tropical semiring.
  Array1<int32_t> check_cycle(c, 1, 0);
  int32_t *check_cycle_data = check_cycle.Data();
  K2_EVAL(
      c, src_num_arcs, lambda_check_self_cycle, (int32_t arc_idx012)->void {
        Arc &cur_arc = src_arcs_data[arc_idx012];
        if (cur_arc.src_state == cur_arc.dest_state && cur_arc.score > 0) {
          check_cycle_data[0] = 1;
        }
      });
  K2_CHECK_EQ(check_cycle[0], 0)
      << "Detected epsilon cycles with positive weight!";

  // Suppose we append another axis (axis 3) to `src` (so src.NumAxes() will be
  // 4), then src_row_splits3 is indexed by arc indexes in `src`. The number of
  // elements for row i on axis 3 is the number of arcs we will expand from
  // arc i in `src`. By saying `expand`, we mean for each arc i in `src`,
  // suppose it's src_state is `s` and dest_state is `d` and `d` has `n` leaving
  // arcs whose dest_state are `d1`, `d2`, ..., `dn`, then we will generate
  // `n+1` arcs from current arc i, their src_states are `s`, and dest_states
  // are `d`, `d1`, `d2`, ..., `dn`. Thus, the number of elements for row i will
  // be `n + 1`.
  Array1<int32_t> src_row_splits3(c, src_num_arcs + 1);
  int32_t *src_row_splits3_data = src_row_splits3.Data();
  K2_EVAL(
      c, src_num_arcs, lambda_set_row_splits3_data, (int32_t arc_idx012)->void {
        int32_t cur_arc_dest_state_idx1 = src_arcs_data[arc_idx012].dest_state;
        int32_t state_idx01 = src_row_ids2_data[arc_idx012],
                fsa_idx0 = src_row_ids1_data[state_idx01],
                start_state_idx0x = src_row_splits1_data[fsa_idx0],
                dest_state_idx01 = start_state_idx0x + cur_arc_dest_state_idx1;
        int32_t num_leaving_arcs_of_dest_state =
            src_row_splits2_data[dest_state_idx01 + 1] -
            src_row_splits2_data[dest_state_idx01];
        src_row_splits3_data[arc_idx012] = 1 + num_leaving_arcs_of_dest_state;
      });
  ExclusiveSum(src_row_splits3.Arange(0, src_num_arcs), &src_row_splits3);
  // note below code has a Device to Host memory copy.
  int32_t expand_arc_nums = src_row_splits3.Back();

  // Here we'll create an Ragged<Arc> `expand` with NumAxes() == 4, its shape
  // is ComposeRaggedShapes(src.shape, RaggedShape2(&src_row_splits3, null,
  // expand_arc_nums)), its value is `expand_arcs` below. For each row i in
  // `src_row_splits3`, the corresponding arcs in `expand_arcs` are those arcs
  // we generate from arc i in `src`. Then, we can get `closure_fsa` by just
  // doing RemoveAxis(expand, 2). Of course after that we still need to sort
  // those arcs leaving from each state in `closure_fsa` on (dest_state then
  // weight), and then for consecutive arcs who have the same dest_state, we
  // only keep the first arc (who has the largest weight, as we sort state's
  // score in ascending order) as one of arc of the returned  `closure_fsa`.
  Array1<Arc> expand_arcs(c, expand_arc_nums);
  Arc *expand_arcs_data = expand_arcs.Data();
  RaggedShape expand_shape = ComposeRaggedShapes(
      src.shape, RaggedShape2(&src_row_splits3, nullptr, expand_arc_nums));
  // just create an alias `expand_row_splits3_data` to use in below lambda.
  const int32_t *expand_row_splits3_data = src_row_splits3_data;
  // expand_row_ids3 is the row_ids corresponding to expand_row_splits3.
  const int32_t *expand_row_ids3_data = expand_shape.RowIds(3).Data();
  // Here we pretend we crate an Ragged<int32_t> `expand_arc_map` with NumAxes()
  // ==2, row i in it is the sequence of src_arc_idx012 that arc i in `expand`
  // corresponds to.
  Array1<int32_t> expand_arc_map_row_splits(c, expand_arc_nums + 1);
  int32_t *expand_arc_map_row_splits_data = expand_arc_map_row_splits.Data();
  K2_EVAL(
      c, expand_arc_nums, lambda_set_expand_arc_map_row_splits,
      (int32_t expand_arc_idx0123)->void {
        // src_arc_idx012 equals to expand_arc_idx012
        int32_t src_arc_idx012 = expand_row_ids3_data[expand_arc_idx0123],
                expand_arc_idx012x = expand_row_splits3_data[src_arc_idx012],
                expand_arc_idx3 = expand_arc_idx0123 - expand_arc_idx012x;
        // expand_arc_idx3 != 0 means this arc is an `expanded` arc, so the
        // length of the corresponding arc-map sequence is 2
        expand_arc_map_row_splits_data[expand_arc_idx0123] =
            (expand_arc_idx3 != 0 ? 2 : 1);
      });
  ExclusiveSum(expand_arc_map_row_splits.Arange(0, expand_arc_nums),
               &expand_arc_map_row_splits);
  // note below code has a Device to Host memory copy.
  int32_t expand_arc_map_num_values = expand_arc_map_row_splits.Back();
  Array1<int32_t> expand_arc_map_row_ids(c, expand_arc_map_num_values);
  Array1<int32_t> expand_arc_map_values(c, expand_arc_map_num_values);
  int32_t *expand_arc_map_row_ids_data = expand_arc_map_row_ids.Data(),
          *expand_arc_map_values_data = expand_arc_map_values.Data();

  K2_EVAL(
      c, expand_arc_nums, lambda_set_expand_arcs,
      (int32_t expand_arc_idx0123)->void {
        // src_arc_idx012 equals to expand_arc_idx012
        int32_t src_arc_idx012 = expand_row_ids3_data[expand_arc_idx0123],
                expand_arc_idx012x = expand_row_splits3_data[src_arc_idx012],
                expand_arc_idx3 = expand_arc_idx0123 - expand_arc_idx012x;
        const Arc &cur_src_arc = src_arcs_data[src_arc_idx012];
        // set arc map info, `expand_arc_map_idx0x` is the first index for arc
        // map of arc expand_arc_idx0123 in `expand`.
        int32_t expand_arc_map_idx0x =
            expand_arc_map_row_splits_data[expand_arc_idx0123];
        expand_arc_map_row_ids_data[expand_arc_map_idx0x] = expand_arc_idx0123;
        expand_arc_map_values_data[expand_arc_map_idx0x] = src_arc_idx012;
        if (expand_arc_idx3 != 0) {
          // it's an `expanded` arc, we need to create a new arc whose src_state
          // is cur_src_arc.src_state and dest_state is the dest state of the
          // corresponding arc leaving cur_src_arc.dest_state.
          int32_t cur_src_arc_dest_state_idx1 = cur_src_arc.dest_state;
          int32_t cur_state_idx01 = src_row_ids2_data[src_arc_idx012],
                  fsa_idx0 = src_row_ids1_data[cur_state_idx01],
                  start_state_idx0x = src_row_splits1_data[fsa_idx0],
                  dest_state_idx01 =
                      start_state_idx0x + cur_src_arc_dest_state_idx1;
          // leaving_arc_idx01x is the index of the first arc leaving
          // cur_src_arc.dest_state. Noticed that we minus `1` here when
          // computing `leaving_arc_idx012` as the first arc in current state of
          // `expand` is just copying from cur_src_arc (the else branch
          // below that has `expand_arc_idx3 == 0`), it's not an `expanded` arc.
          int32_t leaving_arc_idx01x = src_row_splits2_data[dest_state_idx01],
                  leaving_arc_idx012 = leaving_arc_idx01x + expand_arc_idx3 - 1;
          const Arc &cur_expanding_arc = src_arcs_data[leaving_arc_idx012];
          // the label of the expanded arc is always `0` as the input fsa is
          // `epsilon_fsa` (containing only epsilon arcs), the score is the sum
          // of cur_src_arc's score and the expanding-arc's score.
          expand_arcs_data[expand_arc_idx0123] =
              Arc(cur_src_arc.src_state, cur_expanding_arc.dest_state, 0,
                  cur_src_arc.score + cur_expanding_arc.score);
          // it's an expanded arc, so there are two elements in its arc map.
          expand_arc_map_row_ids_data[expand_arc_map_idx0x + 1] =
              expand_arc_idx0123;
          expand_arc_map_values_data[expand_arc_map_idx0x + 1] =
              leaving_arc_idx012;
        } else {
          // it's not the `expanded` arc, just copy current arc in `src`.
          expand_arcs_data[expand_arc_idx0123] = cur_src_arc;
        }
      });
  expand_shape = RemoveAxis(expand_shape, 2);
  Array1<int32_t> sort_arc_map(c, expand_arc_nums);
  FsaVec expand(expand_shape, expand_arcs);
  SortSublists<Arc, ArcComparer>(&expand, &sort_arc_map);

  // For consecutive arcs who have the same dest_state, we only keep the first
  // arc (it has the largest score, as we sort state's score in
  // ascending order).
  Renumbering arc_renumbering(c, expand_arc_nums);
  char *arc_keep_data = arc_renumbering.Keep().Data();
  const int32_t *cur_expand_row_ids2_data = expand.RowIds(2).Data(),
                *cur_expand_row_splits2_data = expand.RowSplits(2).Data();
  K2_EVAL(
      c, expand_arc_nums, lambda_set_keep, (int32_t arc_idx012)->void {
        int32_t expand_state_idx01 = cur_expand_row_ids2_data[arc_idx012],
                expand_arc_idx01x =
                    cur_expand_row_splits2_data[expand_state_idx01];
        if (arc_idx012 == expand_arc_idx01x) {
          // always keep the first leaving arc of this state
          arc_keep_data[arc_idx012] = 1;
        } else {
          // arc_idx012 >= expand_arc_idx01x, so there must be multiple leaving
          // arcs from current state (i.e. `expand_state_idx01`), it's safe to
          // -1 below.
          int32_t cur_arc_dest_state = expand_arcs_data[arc_idx012].dest_state,
                  pre_arc_dest_state =
                      expand_arcs_data[arc_idx012 - 1].dest_state;
          arc_keep_data[arc_idx012] =
              (cur_arc_dest_state != pre_arc_dest_state);
        }
      });

  Array1<int32_t> arc_old_to_new = arc_renumbering.Old2New();
  Array1<int32_t> arc_new_to_old = arc_renumbering.New2Old();
  Array1<int32_t> closure_fsa_row_splits2 = arc_old_to_new[expand.RowSplits(2)];
  Array1<int32_t> closure_fsa_row_ids2 = expand.RowIds(2)[arc_new_to_old];
  int32_t closure_fsa_num_arcs = arc_renumbering.NumNewElems();
  Array1<Arc> closure_fsa_arcs(c, closure_fsa_num_arcs);
  Arc *closure_fsa_arcs_data = closure_fsa_arcs.Data();
  Array1<int32_t> arc_map_indexes(c, closure_fsa_num_arcs);
  int32_t *arc_map_indexes_data = arc_map_indexes.Data();
  const int32_t *sort_arc_map_data = sort_arc_map.Data(),
                *arc_new_to_old_data = arc_new_to_old.Data();
  K2_EVAL(
      c, closure_fsa_num_arcs, lambda_set_arc_map_indexes_and_arcs,
      (int32_t closure_fsa_arc_idx012)->void {
        int32_t expand_arc_idx012 = arc_new_to_old_data[closure_fsa_arc_idx012];
        closure_fsa_arcs_data[closure_fsa_arc_idx012] =
            expand_arcs_data[expand_arc_idx012];
        // noted we have sort `expand` before, but `expand_arc_map.values` is
        // indexed by the indexes before sorting
        int32_t expand_arc_idx012_before_sort =
            sort_arc_map_data[expand_arc_idx012];
        arc_map_indexes_data[closure_fsa_arc_idx012] =
            expand_arc_idx012_before_sort;
      });
  RaggedShape closure_fsa_shape = ComposeRaggedShapes(
      GetLayer(src.shape, 0),
      RaggedShape2(&closure_fsa_row_splits2, &closure_fsa_row_ids2,
                   closure_fsa_num_arcs));
  *closure_fsa = FsaVec(closure_fsa_shape, closure_fsa_arcs);

  Ragged<int32_t> expand_arc_map(
      RaggedShape2(&expand_arc_map_row_splits, &expand_arc_map_row_ids, -1),
      expand_arc_map_values);
  *arc_map = Index(expand_arc_map, arc_map_indexes);
}

void RemoveEpsilonsIterativeTropical(FsaOrVec &src_fsa, FsaOrVec *dest_fsa,
                                     Ragged<int32_t> *arc_map_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(dest_fsa != nullptr && arc_map_out != nullptr);
  K2_CHECK_GE(src_fsa.NumAxes(), 2);
  K2_CHECK_LE(src_fsa.NumAxes(), 3);
  if (src_fsa.NumAxes() == 2) {
    // Turn single Fsa into FsaVec.
    Fsa *srcs = &src_fsa;
    FsaVec src_vec = CreateFsaVec(1, &srcs), dest_vec;
    // Recurse..
    RemoveEpsilonsIterativeTropical(src_vec, &dest_vec, arc_map_out);
    *dest_fsa = GetFsaVecElement(dest_vec, 0);
    return;
  }

  ContextPtr &c = src_fsa.Context();
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

  // epsilon_closure_mapped will have all arcs in `epsilon_fsa_closure` whose
  // src_state and dest_state are both kept in `non_epsilon_fsa`.
  FsaVec epsilon_closure_mapped;
  Ragged<int32_t> epsilon_closure_mapped_arc_map;
  // note below we pass `epsilons_state_map` as the state map of
  // `epsilon_fsa_closure` as `epsilon_fsa_closure` and `epsilon_fsa` have the
  // same states.
  GetEpsilonClosureMapped(
      epsilon_fsa_closure, epsilons_state_map, epsilon_closure_arc_map,
      non_epsilon_fsa, non_epsilon_state_renumbering, &epsilon_closure_mapped,
      &epsilon_closure_mapped_arc_map);

  Renumbering epsilon_prec_renumbering;
  // This shape just says, for each arc in epsilon_closure_mapped
  // that is to be combined with following arcs, how many following
  // arcs it is combined with (else 0).
  RaggedShape foll_shape;
  DecideCombineWithFollowingOrPreceding(epsilon_closure_mapped, non_epsilon_fsa,
                                        &epsilon_prec_renumbering, &foll_shape);

  // `combined_foll` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs following them.
  FsaVec combined_foll;
  Ragged<int32_t> combined_foll_arc_map;
  CombineWithFollowingNonEpsilonArcs(
      epsilon_closure_mapped, epsilon_closure_mapped_arc_map, non_epsilon_fsa,
      non_epsilon_arc_map, foll_shape, &combined_foll, &combined_foll_arc_map);

  FsaVec epsilon_closure_prec =
      SubsampleRagged(epsilon_closure_mapped, epsilon_prec_renumbering);
  Ragged<int32_t> epsilon_closure_prec_arc_map =
      Index(epsilon_closure_mapped_arc_map, epsilon_prec_renumbering.New2Old());
  // `combined_prec` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs preceding them.
  FsaVec combined_prec;
  Ragged<int32_t> combined_prec_arc_map;
  {
    // This shape just says, for each arc in non_epsilon_fsa
    // that is to be combined with following epsilon arcs, how many following
    // epsilon arcs it is combined with (else 0).
    RaggedShape foll_shape;
    // for each arc in `combined_prec`, it gives those arc map info from
    // `epsilon_closure_prec_arc_map`.
    Ragged<int32_t> epsilon_closure_prec_arc_map_prec;
    CombineWithPrecedingNonEpsilonArcs(
        epsilon_closure_prec, epsilon_closure_prec_arc_map, non_epsilon_fsa,
        &combined_prec, &epsilon_closure_prec_arc_map_prec, &foll_shape);
    combined_prec_arc_map =
        AddPrefixToRagged(epsilon_closure_prec_arc_map_prec,
                          non_epsilon_arc_map[foll_shape.RowIds(1)]);
  }

  // we also need combined those epsilon arcs which is supposed to combined
  // preceding non-epsilon arcs with arcs in `combined_foll`, in case
  // combined_foll makes some states non-connected (then if combined an epsilon
  // arc with a preceding non-epsilon arc whose src_state is non-connected, it
  // will generate a meaningless arc, i.e. non-connected.)
  FsaVec combined_prec_foll;
  Ragged<int32_t> combined_prec_foll_arc_map;
  {
    RaggedShape foll_shape;
    Ragged<int32_t> epsilon_closure_prec_arc_map_prec;
    CombineWithPrecedingNonEpsilonArcs(
        epsilon_closure_prec, epsilon_closure_prec_arc_map, combined_foll,
        &combined_prec_foll, &epsilon_closure_prec_arc_map_prec, &foll_shape);
    Ragged<int32_t> combined_foll_arc_map_prec =
        Index(combined_foll_arc_map, foll_shape.RowIds(1));
    Ragged<int32_t> *arc_maps[2] = {&combined_foll_arc_map_prec,
                                    &epsilon_closure_prec_arc_map_prec};
    int32_t axis = 1;
    combined_prec_foll_arc_map = Append(axis, 2, arc_maps);
  }

  // NOTE: important that order matches with `arc_maps` below.
  FsaVec *vecs[4] = {&combined_foll, &combined_prec, &combined_prec_foll,
                     &non_epsilon_fsa};
  int32_t axis = 2;
  Array1<uint32_t> arcs_merge_map;
  FsaVec dest_fsa_unsorted = Append(axis, 4, vecs, &arcs_merge_map);

  Ragged<int32_t> non_epsilon_arc_map_ragged(
      RegularRaggedShape(c, non_epsilon_fsa.NumElements(), 1),
      non_epsilon_arc_map);

  Ragged<int32_t> dest_unsorted_arc_map;
  if (arc_map_out != nullptr) {
    // This block creates 'dest_unsorted_arc_map' which combines
    // combined_foll_arc_map, combined_prec_arc_map and
    // non_epsilon_arc_map_ragged

    // NOTE: important that order matches with `vecs` above.
    Ragged<int32_t> *arc_maps_full[] = {
        &combined_foll_arc_map, &combined_prec_arc_map,
        &combined_prec_foll_arc_map, &non_epsilon_arc_map_ragged};

    dest_unsorted_arc_map = Merge(4, arc_maps_full, arcs_merge_map, nullptr);
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
