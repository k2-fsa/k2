/**
 * Copyright      2021  Xiaomi Corporation (authors: Wei Kang)
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

class Connecter {
 public:
  /**
     Connecter object.  You should call Connect() after
     constructing it.  Please see Connect() declaration in header for
     high-level overview of the algorithm.

       @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  explicit Connecter(FsaVec &fsas) : c_(fsas.Context()), fsas_(fsas) {
    K2_CHECK_EQ(fsas_.NumAxes(), 3);
    int32_t num_states = fsas_.shape.TotSize(1);
    accessible_ = Array1<char>(c_, num_states, 0);
    coaccessible_ = Array1<char>(c_, num_states, 0);
  }

  /*
    Computes the next batch of states
         @param [in] cur_states  Ragged array with 2 axes, containing
    state-indexes (idx01) into fsas_.
         @return   Returns the states which, after processing.
   */
  std::unique_ptr<Ragged<int32_t>> GetNextBatch(Ragged<int32_t> &cur_states) {
    NVTX_RANGE(K2_FUNC);
    // Process arcs leaving all states in `cur`

    // First figure out how many arcs leave each state.
    // And set accessible for each state
    Array1<int32_t> num_arcs_per_state(c_, cur_states.NumElements() + 1);
    int32_t *num_arcs_per_state_data = num_arcs_per_state.Data();
    const int32_t *fsas_row_splits2_data = fsas_.RowSplits(2).Data(),
                  *states_data = cur_states.values.Data();
    char *accessible_data = accessible_.Data();
    K2_EVAL(
        c_, cur_states.NumElements(), lambda_set_arcs_and_accessible_per_state,
        (int32_t states_idx01)->void {
          int32_t fsas_idx01 = states_data[states_idx01],
                  num_arcs = fsas_row_splits2_data[fsas_idx01 + 1] -
                             fsas_row_splits2_data[fsas_idx01];
          num_arcs_per_state_data[states_idx01] = num_arcs;
          // Accessible
          accessible_data[fsas_idx01] = 1;
        });
    ExclusiveSum(num_arcs_per_state, &num_arcs_per_state);

    // arcs_shape `[fsas][states cur][arcs]
    RaggedShape arcs_shape = ComposeRaggedShapes(
        cur_states.shape, RaggedShape2(&num_arcs_per_state, nullptr, -1));

    // Each arc that generates a new state (i.e. for which
    // arc_renumbering.Keep[i] == true) will write the state-id to here (as an
    // idx01 into fsas_).  Other elements will be undefined.
    Array1<int32_t> next_iter_states(c_, arcs_shape.NumElements());

    // We'll be figuring out which of these arcs lead to a state that is not
    // accessible yet.
    Renumbering arc_renumbering(c_, arcs_shape.NumElements());

    const int32_t *arcs_row_ids1_data = arcs_shape.RowIds(1).Data(),
                  *arcs_row_ids2_data = arcs_shape.RowIds(2).Data(),
                  *arcs_row_splits2_data = arcs_shape.RowSplits(2).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data(),
                  *dest_states_data = dest_states_.values.Data();
    char *keep_arc_data = arc_renumbering.Keep().Data();
    int32_t *next_iter_states_data = next_iter_states.Data();
    K2_EVAL(
        c_, arcs_shape.NumElements(), lambda_set_arc_renumbering,
        (int32_t arcs_idx012)->void {
          // note: the prefix `arcs_` means it is an idxXXX w.r.t. `arcs_shape`.
          // the prefix `fsas_` means the variable is an idxXXX w.r.t. `fsas_`.
          int32_t arcs_idx01 = arcs_row_ids2_data[arcs_idx012],
                  arcs_idx01x = arcs_row_splits2_data[arcs_idx01],
                  arcs_idx2 = arcs_idx012 - arcs_idx01x,
                  fsas_idx01 = states_data[arcs_idx01],  // a state index
                  fsas_idx01x = fsas_row_splits2_data[fsas_idx01],
                  fsas_idx012 = fsas_idx01x + arcs_idx2,
                  fsas_dest_state_idx01 = dest_states_data[fsas_idx012];
          // if this arc is a self-loop, just ignore this arc as we won't
          // processe the dest_state (current state) again
          if (fsas_dest_state_idx01 == fsas_idx01 ||
              accessible_data[fsas_dest_state_idx01]) {
            keep_arc_data[arcs_idx012] = 0;
            return;
          }
          keep_arc_data[arcs_idx012] = 1;
          next_iter_states_data[arcs_idx012] = fsas_dest_state_idx01;
        });

    Array1<int32_t> new2old_map = arc_renumbering.New2Old();
    if (new2old_map.Dim() == 0) {
      // There are no new states.  This means we terminated.  We'll check from
      // calling code that we processed all arcs.
      return nullptr;
    }
    // `new_states` will contain state-ids which are idx01's into `fsas_`.
    Array1<int32_t> new_states = next_iter_states[new2old_map];
    Array1<int32_t> new_states_row_ids(c_, new_states.Dim());  // will map to
                                                               // FSA index
    const int32_t *new2old_map_data = new2old_map.Data();
    int32_t *new_states_row_ids_data = new_states_row_ids.Data();
    K2_EVAL(
        c_, new_states.Dim(), lambda_set_row_ids,
        (int32_t new_state_idx)->void {
          int32_t arcs_idx012 = new2old_map_data[new_state_idx],
                  arcs_idx01 = arcs_row_ids2_data[arcs_idx012],  // state index
              arcs_idx0 = arcs_row_ids1_data[arcs_idx01];        // FSA index
          new_states_row_ids_data[new_state_idx] = arcs_idx0;
        });

    int32_t num_fsas = fsas_.Dim0();
    Array1<int32_t> new_states_row_splits(c_, num_fsas + 1);
    RowIdsToRowSplits(new_states_row_ids, &new_states_row_splits);

    auto ans = std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&new_states_row_splits, &new_states_row_ids, new_states.Dim()),
        new_states);
    // The following will ensure the answer has deterministic numbering
    SortSublists(ans.get());
    return ans;
  }

  /*
    Computes the next batch of states in reverse order
         @param [in] cur_states  Ragged array with 2 axes, containing
    state-indexes (idx01) into fsas_.
         @return   Returns the states which, after processing.
   */
  std::unique_ptr<Ragged<int32_t>> GetNextBatchBackward(
      Ragged<int32_t> &cur_states) {
    NVTX_RANGE(K2_FUNC);
    // Process arcs entering all states in `cur`

    // First figure out how many arcs enter each state.
    // And set coaccessible for each state
    Array1<int32_t> num_arcs_per_state(c_, cur_states.NumElements() + 1);
    int32_t *num_arcs_per_state_data = num_arcs_per_state.Data();
    const int32_t *incoming_arcs_row_splits2_data =
                    incoming_arcs_.RowSplits(2).Data(),
                  *states_data = cur_states.values.Data();
    char *coaccessible_data = coaccessible_.Data();
    K2_EVAL(
        c_, cur_states.NumElements(), lambda_set_arcs_and_coaccessible_per_state,
        (int32_t states_idx01)->void {
          int32_t fsas_idx01 = states_data[states_idx01],
                  num_arcs = incoming_arcs_row_splits2_data[fsas_idx01 + 1] -
                             incoming_arcs_row_splits2_data[fsas_idx01];
          num_arcs_per_state_data[states_idx01] = num_arcs;
          // Coaccessible
          coaccessible_data[states_data[states_idx01]] = 1;
        });
    ExclusiveSum(num_arcs_per_state, &num_arcs_per_state);

    // arcs_shape `[fsas][states cur][arcs]
    RaggedShape arcs_shape = ComposeRaggedShapes(
        cur_states.shape, RaggedShape2(&num_arcs_per_state, nullptr, -1));

    // Each arc that generates a new state (i.e. for which
    // arc_renumbering.Keep[i] == true) will write the state-id to here (as an
    // idx01 into fsas_).  Other elements will be undefined.
    Array1<int32_t> next_iter_states(c_, arcs_shape.NumElements());

    // We'll be figuring out which of these arcs comes from a state that are
    // not coaccessible yet.
    Renumbering arc_renumbering(c_, arcs_shape.NumElements());

    const int32_t *arcs_row_ids1_data = arcs_shape.RowIds(1).Data(),
                  *arcs_row_ids2_data = arcs_shape.RowIds(2).Data(),
                  *arcs_row_splits2_data = arcs_shape.RowSplits(2).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data(),
                  *fsas_row_splits2_data = fsas_.RowSplits(2).Data(),
                  *fsas_row_ids1_data = fsas_.RowIds(1).Data(),
                  *incoming_arcs_data = incoming_arcs_.values.Data();
    const Arc *fsas_data = fsas_.values.Data();
    char *keep_arc_data = arc_renumbering.Keep().Data();
    int32_t *next_iter_states_data = next_iter_states.Data();
    K2_EVAL(
        c_, arcs_shape.NumElements(), lambda_set_arc_renumbering,
        (int32_t arcs_idx012)->void {
          // note: the prefix `arcs_` means it is an idxXXX w.r.t. `arcs_shape`.
          // the prefix `fsas_` means the variable is an idxXXX w.r.t. `fsas_`.
          int32_t arcs_idx01 = arcs_row_ids2_data[arcs_idx012],
                  arcs_idx01x = arcs_row_splits2_data[arcs_idx01],
                  arcs_idx2 = arcs_idx012 - arcs_idx01x,
                  fsas_idx01 = states_data[arcs_idx01],  // a state index
                  fsas_idx0 = fsas_row_ids1_data[fsas_idx01],
                  fsas_idx01x = incoming_arcs_row_splits2_data[fsas_idx01],
                  fsas_idx012 = fsas_idx01x + arcs_idx2,
                  fsas_src_state_idx1 =
                    fsas_data[incoming_arcs_data[fsas_idx012]].src_state,
                  fsas_src_state_idx0x = fsas_row_splits1_data[fsas_idx0],
                  fsas_src_state_idx01 =
                    fsas_src_state_idx0x + fsas_src_state_idx1;
          // if this arc is a self-loop, just ignore this arc as we won't
          // processe the src_state (current state) again.
          if (fsas_src_state_idx01 == fsas_idx01 ||
              coaccessible_data[fsas_src_state_idx01]) {
            keep_arc_data[arcs_idx012] = 0;
            return;
          }
          keep_arc_data[arcs_idx012] = 1;
          next_iter_states_data[arcs_idx012] = fsas_src_state_idx01;
        });

    Array1<int32_t> new2old_map = arc_renumbering.New2Old();
    if (new2old_map.Dim() == 0) {
      // There are no new states.  This means we terminated.  We'll check from
      // calling code that we processed all arcs.
      return nullptr;
    }
    // `new_states` will contain state-ids which are idx01's into `fsas_`.
    Array1<int32_t> new_states = next_iter_states[new2old_map];
    Array1<int32_t> new_states_row_ids(c_, new_states.Dim());  // will map to
                                                               // FSA index
    const int32_t *new2old_map_data = new2old_map.Data();
    int32_t *new_states_row_ids_data = new_states_row_ids.Data();
    K2_EVAL(
        c_, new_states.Dim(), lambda_set_row_ids,
        (int32_t new_state_idx)->void {
          int32_t arcs_idx012 = new2old_map_data[new_state_idx],
                  arcs_idx01 = arcs_row_ids2_data[arcs_idx012],  // state index
              arcs_idx0 = arcs_row_ids1_data[arcs_idx01];        // FSA index
          new_states_row_ids_data[new_state_idx] = arcs_idx0;
        });

    int32_t num_fsas = fsas_.Dim0();
    Array1<int32_t> new_states_row_splits(c_, num_fsas + 1);
    RowIdsToRowSplits(new_states_row_ids, &new_states_row_splits);

    std::unique_ptr<Ragged<int32_t>> ans = std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&new_states_row_splits, &new_states_row_ids,
        new_states.Dim()), new_states);
    // The following will ensure the answer has deterministic numbering
    SortSublists(ans.get());
    return ans;
  }

  /*
    Returns the start batch of states.  This will include all start-states that
    existed in the original FSAs.
   */
  std::unique_ptr<Ragged<int32_t>> GetStartBatch() {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = fsas_.Dim0();
    const int32_t *fsas_row_splits1_data = fsas_.RowSplits(1).Data();
    Array1<int32_t> has_start_state(c_, num_fsas + 1);
    int32_t *has_start_state_data = has_start_state.Data();
    K2_EVAL(
        c_, num_fsas, lambda_set_has_start_state, (int32_t i)->void {
          int32_t split = fsas_row_splits1_data[i],
                  next_split = fsas_row_splits1_data[i + 1];
          has_start_state_data[i] = (next_split > split);
        });
    ExclusiveSum(has_start_state, &has_start_state);

    int32_t n = has_start_state[num_fsas];
    std::unique_ptr<Ragged<int32_t>> ans = std::make_unique<Ragged<int32_t>>(
        RaggedShape2(&has_start_state, nullptr, n), Array1<int32_t>(c_, n));
    int32_t *ans_data = ans->values.Data();
    const int32_t *ans_row_ids1_data = ans->RowIds(1).Data();
    K2_EVAL(
        c_, n, lambda_set_start_state, (int32_t i)->void {
          int32_t fsa_idx0 = ans_row_ids1_data[i],
                  start_state = fsas_row_splits1_data[fsa_idx0];
          // If the following fails, it likely means an input FSA was invalid
          // (e.g. had exactly one state, which is not allowed).  Either that,
          // or a code error.
          K2_DCHECK_LT(start_state, fsas_row_splits1_data[fsa_idx0 + 1]);
          ans_data[i] = start_state;
        });
    return ans;
  }


  /*
    Returns the final batch of states.  This will include all final-states that
    existed in the original FSAs.
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
    std::unique_ptr<Ragged<int32_t>> ans = std::make_unique<Ragged<int32_t>>(
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

  /* Does the main work of connecting and returns the resulting FSAs.
        @param [out] arc_map  if non-NULL, the map from (arcs in output)
                     to (corresponding arcs in input) is written to here.
        @return   Returns the connected FsaVec.
   */
  FsaVec Connect(Array1<int32_t> *arc_map) {
    NVTX_RANGE(K2_FUNC);
    Array1<int32_t> dest_states_idx01 = GetDestStates(fsas_, true);
    dest_states_ = Ragged<int32_t>(fsas_.shape, dest_states_idx01);
    incoming_arcs_ = GetIncomingArcs(fsas_, dest_states_idx01);

    // Mark accessible states
    std::unique_ptr<Ragged<int32_t>> iter = GetStartBatch();
    while (iter != nullptr)
      iter = GetNextBatch(*iter);

    // Mark coaccessible states
    std::unique_ptr<Ragged<int32_t>> riter = GetFinalBatch();
    while (riter != nullptr)
      riter = GetNextBatchBackward(*riter);

    // Get remaining states and construct row_ids1/row_splits1
    int32_t num_states = fsas_.shape.TotSize(1);
    const char *accessible_data = accessible_.Data(),
               *coaccessible_data = coaccessible_.Data();
    Renumbering states_renumbering(c_, num_states);
    char* states_renumbering_data = states_renumbering.Keep().Data();
    K2_EVAL(
        c_, num_states, lambda_set_states_renumbering,
        (int32_t state_idx01)->void {
          if (accessible_data[state_idx01] && coaccessible_data[state_idx01])
            states_renumbering_data[state_idx01] = 1;
          else
            states_renumbering_data[state_idx01] = 0;
        });
    Array1<int32_t> new2old_map_states = states_renumbering.New2Old();
    Array1<int32_t> old2new_map_states = states_renumbering.Old2New();
    Array1<int32_t> new_row_ids1 = fsas_.RowIds(1)[new2old_map_states];
    Array1<int32_t> new_row_splits1(c_, fsas_.Dim0() + 1);
    RowIdsToRowSplits(new_row_ids1, &new_row_splits1);

    // Get remaining arcs
    int32_t num_arcs = fsas_.NumElements();
    Renumbering arcs_renumbering(c_, num_arcs);
    char* arcs_renumbering_data = arcs_renumbering.Keep().Data();
    const Arc *fsas_data = fsas_.values.Data();
    const int32_t *fsas_row_ids1_data = fsas_.RowIds(1).Data(),
                  *fsas_row_ids2_data = fsas_.RowIds(2).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data();
    K2_EVAL(
        c_, num_arcs, lambda_set_arcs_renumbering,
        (int32_t arc_idx012)->void {
          Arc arc = fsas_data[arc_idx012];
          int32_t fsas_idx01 = fsas_row_ids2_data[arc_idx012],
                  src_state_idx01 = fsas_idx01,
                  dest_state_idx01 =
                    arc.dest_state - arc.src_state + src_state_idx01;
          if (accessible_data[src_state_idx01] &&
              coaccessible_data[src_state_idx01] &&
              accessible_data[dest_state_idx01] &&
              coaccessible_data[dest_state_idx01])
            arcs_renumbering_data[arc_idx012] = 1;
          else
            arcs_renumbering_data[arc_idx012] = 0;
        });
    Array1<int32_t> new2old_map_arcs = arcs_renumbering.New2Old();
    int32_t remaining_arcs_num = new2old_map_arcs.Dim();

    // Construct row_ids2/row_splits2
    Array1<int32_t> new_row_ids2(c_, remaining_arcs_num);
    int32_t *new_row_ids2_data = new_row_ids2.Data();
    const int32_t *new2old_map_arcs_data = new2old_map_arcs.Data(),
                  *old2new_map_states_data = old2new_map_states.Data();
    K2_EVAL(
        c_, remaining_arcs_num, lambda_set_new_row_ids2,
        (int32_t arc_idx012)->void {
          int32_t fsas_idx012 = new2old_map_arcs_data[arc_idx012],
                  fsas_idx01 = fsas_row_ids2_data[fsas_idx012];
          new_row_ids2_data[arc_idx012] = old2new_map_states_data[fsas_idx01];
        });

    Array1<int32_t> new_row_splits2(c_, new2old_map_states.Dim() + 1);
    RowIdsToRowSplits(new_row_ids2, &new_row_splits2);

    // Update arcs to the renumbered states
    const int32_t *new_row_ids1_data = new_row_ids1.Data(),
                  *new_row_splits1_data = new_row_splits1.Data();
    Array1<Arc> remaining_arcs(c_, remaining_arcs_num);
    Arc *remaining_arcs_data = remaining_arcs.Data();
    K2_EVAL(
        c_, remaining_arcs_num, lambda_set_arcs,
        (int32_t arc_idx012)->void {
          int32_t fsas_idx012 = new2old_map_arcs_data[arc_idx012],
                  fsas_idx01 = fsas_row_ids2_data[fsas_idx012],
                  fsas_idx0 = fsas_row_ids1_data[fsas_idx01],
                  fsas_idx0x = fsas_row_splits1_data[fsas_idx0];
          Arc arc = fsas_data[fsas_idx012];
          int32_t src_old_idx1 = arc.src_state,
                  src_old_idx01 = fsas_idx0x + src_old_idx1,
                  src_idx01 = old2new_map_states_data[src_old_idx01],
                  src_idx0 = new_row_ids1_data[src_idx01],
                  src_idx0x = new_row_splits1_data[src_idx0],
                  src_idx1 = src_idx01 - src_idx0x,
                  dest_old_idx1 = arc.dest_state,
                  dest_old_idx01 = fsas_idx0x + dest_old_idx1,
                  dest_idx01 = old2new_map_states_data[dest_old_idx01],
                  dest_idx0 = new_row_ids1_data[dest_idx01],
                  dest_idx0x = new_row_splits1_data[dest_idx0],
                  dest_idx1 = dest_idx01 - dest_idx0x;
          Arc new_arc;
          new_arc.src_state = src_idx1;
          new_arc.dest_state = dest_idx1;
          new_arc.score = arc.score;
          new_arc.label = arc.label;
          remaining_arcs_data[arc_idx012] = new_arc;
        });

    if (arc_map != nullptr)
      *arc_map = new2old_map_arcs.Clone();

    return Ragged<Arc>(RaggedShape3(&new_row_splits1, &new_row_ids1, -1,
                                    &new_row_splits2, &new_row_ids2, -1),
                       remaining_arcs);
  }

  ContextPtr c_;
  FsaVec &fsas_;

  // For each arc in fsas_ (with same structure as fsas_), dest-state
  // of that arc as an idx01.
  Ragged<int32_t> dest_states_;
  // For each state in fsas_ (with same structure as fsas_), incoming-arc
  // of that state as an idx012.
  Ragged<int32_t> incoming_arcs_;
  // With the Dim() the same as num-states, to mark the state (as an idx01) to
  // be accessible or not
  Array1<char> accessible_;
  // With the Dim() the same as num-states, to mark the state (as an idx01) to
  // be coaccessible or not
  Array1<char> coaccessible_;
};

void Connect(FsaVec &src, FsaVec *dest, Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK_LE(src.NumAxes(), 3);
  if (src.NumAxes() == 2) {
    // Turn single Fsa into FsaVec.
    Fsa *srcs = &src;
    FsaVec src_vec = CreateFsaVec(1, &srcs), dest_vec;
    // Recurse..
    Connect(src_vec, &dest_vec, arc_map);
    *dest = GetFsaVecElement(dest_vec, 0);
    return;
  }
  Connecter connecter(src);
  *dest = connecter.Connect(arc_map);
}

}  // namespace k2
