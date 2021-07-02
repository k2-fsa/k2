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

#include <limits>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"

namespace k2 {

// Caution: this is really a .cu file.  It contains mixed host and device code.

class TopSorter {
 public:
  /**
     Topological sorter object.  You should call TopSort() after
     constructing it.  Please see TopSort() declaration in header for
     high-level overview of the algorithm.

       @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  explicit TopSorter(FsaVec &fsas) : c_(fsas.Context()), fsas_(fsas) {
    K2_CHECK_EQ(fsas_.NumAxes(), 3);
  }

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
  std::unique_ptr<Ragged<int32_t>> GetInitialBatch() {
    NVTX_RANGE(K2_FUNC);
    // Initialize it with a list of all states that currently have zero
    // in-degree.
    int32_t num_states = state_in_degree_.Dim();
    Renumbering state_renumbering(c_, num_states);
    // NOTE: this is not very optimal given that we're keeping only a small
    // number of states, but at this point I don't want to optimize too heavily.
    char *keep_data = state_renumbering.Keep().Data();
    const int32_t *state_in_degree_data = state_in_degree_.Data(),
                  *fsas_row_ids1_data = fsas_.RowIds(1).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data();
    K2_EVAL(
        c_, num_states, lambda_set_keep, (int32_t fsas_idx01)->void {
          // Make this state a member of the initial batch if it has zero
          // in-degree (note: this won't include final states, as we incremented
          // their in-degree to avoid them appearing here.)
          keep_data[fsas_idx01] = state_in_degree_data[fsas_idx01] == 0;
        });

    Array1<int32_t> first_iter_values = state_renumbering.New2Old();
    Array1<int32_t> first_iter_row_ids = fsas_.RowIds(1)[first_iter_values];
    int32_t num_fsas = fsas_.Dim0();
    Array1<int32_t> first_iter_row_splits(c_, num_fsas + 1);
    RowIdsToRowSplits(first_iter_row_ids, &first_iter_row_splits);
    return std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&first_iter_row_splits, &first_iter_row_ids,
                     first_iter_row_ids.Dim()),
        first_iter_values);
  }

  /*
    Computes the next batch of states
         @param [in] cur_states  Ragged array with 2 axes, with the shape of
    `[fsas][states]`, containing state-indexes (idx01) into fsas_.
    These are states which already have in-degree 0
         @return   Returns the states which, after processing.
   */
  std::unique_ptr<Ragged<int32_t>> GetNextBatch(Ragged<int32_t> &cur_states) {
    NVTX_RANGE(K2_FUNC);
    // Process arcs leaving all states in `cur_states`

    // First figure out how many arcs leave each state.
    Array1<int32_t> num_arcs_per_state(c_, cur_states.NumElements() + 1);
    int32_t *num_arcs_per_state_data = num_arcs_per_state.Data();
    const int32_t *states_data = cur_states.values.Data(),
                  *fsas_row_splits2_data = fsas_.RowSplits(2).Data();
    K2_EVAL(
        c_, cur_states.NumElements(), lambda_set_arcs_per_state,
        (int32_t states_idx01)->void {
          int32_t idx01 = states_data[states_idx01],
                  num_arcs = fsas_row_splits2_data[idx01 + 1] -
                             fsas_row_splits2_data[idx01];
          num_arcs_per_state_data[states_idx01] = num_arcs;
        });
    ExclusiveSum(num_arcs_per_state, &num_arcs_per_state);

    // arcs_shape `[fsas][states in-degree 0][arcs]
    RaggedShape arcs_shape = ComposeRaggedShapes(
        cur_states.shape, RaggedShape2(&num_arcs_per_state, nullptr, -1));

    // Each arc that generates a new state (i.e. for which
    // arc_renumbering.Keep[i] == true) will write the state-id to here (as an
    // idx01 into fsas_).  Other elements will be undefined.
    // We will also write the row-id (which fsa the state belongs) for each
    // new state.
    int32_t num_arcs = arcs_shape.NumElements();
    Array1<int32_t> temp(c_, 2 * num_arcs);
    Array1<int32_t> next_iter_states = temp.Arange(0, num_arcs);
    Array1<int32_t> new_state_row_ids = temp.Arange(num_arcs, 2 * num_arcs);

    // We'll be figuring out which of these arcs leads to a state that now has
    // in-degree 0.  (If >1 arc goes to such a state, only one will 'win',
    // arbitrarily).
    Renumbering arc_renumbering(c_, num_arcs);

    const int32_t *arcs_row_ids1_data = arcs_shape.RowIds(1).Data(),
                  *arcs_row_ids2_data = arcs_shape.RowIds(2).Data(),
                  *arcs_row_splits2_data = arcs_shape.RowSplits(2).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data(),
                  *dest_states_data = dest_states_.values.Data();
    char *keep_arc_data = arc_renumbering.Keep().Data();
    int32_t *state_in_degree_data = state_in_degree_.Data(),
            *next_iter_states_data = next_iter_states.Data(),
            *new_state_row_ids_data = new_state_row_ids.Data();
    K2_EVAL(
        c_, num_arcs, lambda_set_arc_renumbering,
        (int32_t arcs_idx012)->void {
          // note: the prefix `arcs_` means it is an idxXXX w.r.t. `arcs_shape`.
          // the prefix `fsas_` means the variable is an idxXXX w.r.t. `fsas_`.
          int32_t arcs_idx01 = arcs_row_ids2_data[arcs_idx012],
                  arcs_idx0 = arcs_row_ids1_data[arcs_idx01],
                  arcs_idx01x = arcs_row_splits2_data[arcs_idx01],
                  arcs_idx2 = arcs_idx012 - arcs_idx01x,
                  fsas_idx01 = states_data[arcs_idx01],  // a state index
                  fsas_idx01x = fsas_row_splits2_data[fsas_idx01],
                  fsas_idx012 = fsas_idx01x + arcs_idx2,
                  fsas_dest_state_idx01 = dest_states_data[fsas_idx012];
          // if this arc is a self-loop, just ignore this arc as we have
          // processed the dest_state (==src_state)
          if (fsas_dest_state_idx01 == fsas_idx01) {
            keep_arc_data[arcs_idx012] = 0;
            return;
          }
          if ((keep_arc_data[arcs_idx012] = AtomicDecAndCompareZero(
                   state_in_degree_data + fsas_dest_state_idx01))) {
            next_iter_states_data[arcs_idx012] = fsas_dest_state_idx01;
            new_state_row_ids_data[arcs_idx012] = arcs_idx0;
          }
        });

    Array1<int32_t> new2old_map = arc_renumbering.New2Old();
    if (new2old_map.Dim() == 0) {
      // There are no new states.  This means we terminated.  We'll check from
      // calling code that we processed all arcs.
      return nullptr;
    }
    int32_t num_states = new2old_map.Dim();
    Array1<int32_t> temp2(c_, 2 * num_states);
    // `new_states` will contain state-ids which are idx01's into `fsas_`.
    Array1<int32_t> new_states = temp2.Arange(0, num_states);
    // `ans_row_ids` will map to FSA index
    Array1<int32_t> ans_row_ids = temp2.Arange(num_states, 2 * num_states);

    const int32_t *new2old_map_data = new2old_map.Data();
    int32_t *ans_row_ids_data = ans_row_ids.Data(),
            *new_states_data = new_states.Data();
    K2_EVAL(
        c_, num_states, lambda_set_new_states_and_row_ids,
        (int32_t new_state_idx)->void {
          int32_t arcs_idx012 = new2old_map_data[new_state_idx];
          new_states_data[new_state_idx] = next_iter_states_data[arcs_idx012];
          ans_row_ids_data[new_state_idx] = new_state_row_ids_data[arcs_idx012];
        });

    int32_t num_fsas = fsas_.Dim0();
    Array1<int32_t> ans_row_splits(c_, num_fsas + 1);
    RowIdsToRowSplits(ans_row_ids, &ans_row_splits);

    auto ans = std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&ans_row_splits, &ans_row_ids, num_states),
        new_states);
    // The following will ensure the answer has deterministic numbering
    SortSublists(ans.get());
    return ans;
  }

  /*
    Returns the final batch of states.  This will include all final-states that
    existed in the original FSAs, i.e. at most one per input.  We treat them
    specially because we can't afford the final-state to not be the last state
    (this is only an issue because we support input where not all states were
    reachable from the start state).
   */
  std::unique_ptr<Ragged<int32_t>> GetFinalBatch() {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = fsas_.Dim0();
    const int32_t *fsas_row_splits1_data = fsas_.RowSplits(1).Data();
    Array1<int32_t> has_final_state(c_, num_fsas + 1);
    int32_t *has_final_state_data = has_final_state.Data();
    K2_EVAL(
        c_, num_fsas, lambda_set_has_final_state, (int32_t i)->void {
          int32_t split = fsas_row_splits1_data[i],
                  next_split = fsas_row_splits1_data[i + 1];
          has_final_state_data[i] = (next_split > split);
        });
    ExclusiveSum(has_final_state, &has_final_state);

    int32_t n = has_final_state[num_fsas];
    auto ans = std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&has_final_state, nullptr, n), Array1<int32_t>(c_, n));
    int32_t *ans_data = ans->values.Data();
    const int32_t *ans_row_ids1_data = ans->RowIds(1).Data();
    K2_EVAL(
        c_, n, lambda_set_final_state, (int32_t i)->void {
          int32_t fsa_idx0 = ans_row_ids1_data[i],
                  final_state = fsas_row_splits1_data[fsa_idx0 + 1] - 1;
          // If the following fails, it likely means an input FSA was invalid
          // (e.g. had exactly one state, which is not allowed).  Either that,
          // or a code error.
          K2_DCHECK_GT(final_state, fsas_row_splits1_data[fsa_idx0]);
          ans_data[i] = final_state;
        });
    return ans;
  }

  void InitDestStatesAndInDegree() {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = fsas_.shape.TotSize(0),
            num_states = fsas_.shape.TotSize(1);

    // Get in-degrees of states.
    int32_t num_arcs = fsas_.NumElements();
    Array1<int32_t> dest_states_idx01 = GetDestStates(fsas_, true);

    dest_states_ = Ragged<int32_t>(fsas_.shape, dest_states_idx01);

    // remove those arcs which are self-loops, as we will not count them in
    // state_in_degree_
    Renumbering arc_renumbering(c_, num_arcs);
    char *keep_arc_data = arc_renumbering.Keep().Data();
    const int32_t *dest_states_data = dest_states_.values.Data(),
                  *fsas_row_ids2_data = fsas_.RowIds(2).Data();
    K2_EVAL(
        c_, num_arcs, lambda_set_keep_arc, (int32_t arc_idx012)->void {
          int32_t dest_state_idx01 = dest_states_data[arc_idx012],
                  src_state_idx01 = fsas_row_ids2_data[arc_idx012];
          keep_arc_data[arc_idx012] = dest_state_idx01 != src_state_idx01;
        });
    state_in_degree_ =
        GetCounts(dest_states_.values[arc_renumbering.New2Old()], num_states);

    int32_t *state_in_degree_data = state_in_degree_.Data();
    const int32_t *fsas_row_splits1_data = fsas_.RowSplits(1).Data();

    // Increment the in-degree of final-states
    K2_EVAL(
        c_, num_fsas, lambda_inc_final_state_in_degree,
        (int32_t fsa_idx0)->void {
          int32_t this_idx01 = fsas_row_splits1_data[fsa_idx0],
                  next_idx01 = fsas_row_splits1_data[fsa_idx0 + 1];
          if (next_idx01 > this_idx01) {
            int32_t final_state = next_idx01 - 1;
            state_in_degree_data[final_state] += 1;
          };
        });
  }

  /* Does the main work of top-sorting and returns the resulting FSAs.
        @param [out] arc_map  if non-NULL, the map from (arcs in output)
                     to (corresponding arcs in input) is written to here.
        @return   Returns the top-sorted FsaVec.  (Note: this may have
                 fewer states than the input if there were unreachable
                 states.)
   */
  FsaVec TopSort(Array1<int32_t> *arc_map) {
    NVTX_RANGE(K2_FUNC);
    InitDestStatesAndInDegree();

    std::vector<std::unique_ptr<Ragged<int32_t>>> iters;
    iters.push_back(GetInitialBatch());

    {  // This block just checks that all non-empty FSAs in the input have their
       // start state in their first batch.
      int32_t num_fsas = fsas_.Dim0();
      Ragged<int32_t> *first_batch = iters.back().get();
      const int32_t *first_batch_states_data = first_batch->values.Data(),
                    *first_batch_row_splits1_data =
                        first_batch->RowSplits(1).Data(),
                    *fsas_row_splits1_data = fsas_.RowSplits(1).Data();
      // Act as a flag
      Array1<int32_t> start_state_present(c_, 1, 1);
      int32_t *start_state_present_data = start_state_present.Data();
      K2_EVAL(
          c_, num_fsas, lambda_set_start_state_present,
          (int32_t fsa_idx0)->void {
            int32_t start_state_idx0x = fsas_row_splits1_data[fsa_idx0],
                    next_start_state_idx0x =
                        fsas_row_splits1_data[fsa_idx0 + 1];
            if (next_start_state_idx0x > start_state_idx0x) {  // non-empty Fsa
              // `first_state_idx01` is the 1st state in the first batch of this
              // fsa (it must be the start state of this Fsa according to our
              // implementation of `GetFirstBatch`
              int32_t first_state_idx01 = first_batch_states_data
                  [first_batch_row_splits1_data[fsa_idx0]];
              if (first_state_idx01 != start_state_idx0x)
                start_state_present_data[0] = 0;
            }
          });
      K2_CHECK_EQ(start_state_present[0], 1)
          << "Our current implementation requires that the start state in each "
             "Fsa must be present in the first batch";
    }

    while (iters.back() != nullptr)
      iters.push_back(GetNextBatch(*iters.back()));
    // note: below, we're overwriting nullptr.
    iters.back() = GetFinalBatch();

    // Need raw pointers for Stack().
    std::vector<Ragged<int32_t> *> iters_ptrs(iters.size());
    for (size_t i = 0; i < iters.size(); ++i) iters_ptrs[i] = iters[i].get();
    Ragged<int32_t> all_states =
        Cat(1, static_cast<int32_t>(iters.size()), iters_ptrs.data());
    K2_CHECK_EQ(all_states.NumElements(), fsas_.TotSize(1))
        << "Our current implementation requires that the input Fsa is acyclic, "
           "but it seems there are cycles other than self-loops.";
    return RenumberFsaVec(fsas_, all_states.values, arc_map);
  }

  ContextPtr c_;
  FsaVec &fsas_;

  // For each arc in fsas_ (with same structure as fsas_), dest-state
  // of that arc as an idx01.
  Ragged<int32_t> dest_states_;

  // The remaining in-degree of each state (state_in_degree_.Dim() ==
  // fsas_.TotSize(1)), i.e. number of incoming arcs (except those from
  // states that were already processed).
  Array1<int32_t> state_in_degree_;
};

void TopSort(FsaVec &src, FsaVec *dest, Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK_LE(src.NumAxes(), 3);
  if (src.NumAxes() == 2) {
    // Turn single Fsa into FsaVec.
    FsaVec src_vec = FsaToFsaVec(src), dest_vec;
    // Recurse..
    TopSort(src_vec, &dest_vec, arc_map);
    *dest = GetFsaVecElement(dest_vec, 0);
    return;
  }
  TopSorter sorter(src);
  *dest = sorter.TopSort(arc_map);
}

}  // namespace k2
