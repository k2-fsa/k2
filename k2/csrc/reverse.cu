/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
 *                2022  ASLP@NWPU          (authors: Hang Lyu)
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

class Reverser {
 public:
  /*
    Reverser object.  You should call Reverse() after constructing it. Please
    see Reverse() declaration in header for high-level overview of the
    algorithm.

      @param [in] fsas    A vector of FSAs; must have 3 axes.
   */
  explicit Reverser(FsaVec &fsas) : c_(fsas.Context()), fsas_(fsas) {
    K2_CHECK_EQ(fsas_.NumAxes(), 3);
  }


  /*
   Figure out the non-empty Fsas (i.e. the fsa has at least two
   states) and make sure the original fsas is legal (i.e. there are at
   least two states in each non-empty fsa).
   "1" means the corresponding FSA is non-emtpy and legel. "0" indicates the
   corresponding FSA is an empty FSA.
   */
  Array1<int32_t> CollectNonEmptyFsas() {
    NVTX_RANGE(K2_FUNC);
    int32_t num_fsas = fsas_.Dim0();
    const int32_t *fsas_row_splits1_data = fsas_.RowSplits(1).Data();

    // The "ans" records all the non-empty fsa.
    Array1<int32_t> ans(c_, num_fsas + 1);
    int32_t *ans_data = ans.Data();
    K2_EVAL(
        c_, num_fsas, lambda_set_has_final_state, (int32_t i)->void {
          int32_t idx0x = fsas_row_splits1_data[i],
                  idx0x_next = fsas_row_splits1_data[i + 1],
                  idx0x_final_state = idx0x_next - 1;

          // If idx0x_next == idx0x, the FSA is an empty FSA.
          if (idx0x_next > idx0x) {
            // If the following fails, it likely means an input FSA was invalid
            // (e.g. had exactly one state, which is not allowed).
            K2_DCHECK_GT(idx0x_final_state, idx0x);
            ans_data[i] = 1;
          } else {
            ans_data[i] = 0;
          }
        });
    return ans;
  }


  /*
   The returned array is a state-level map from original 'old_state_idx01' to
   reversed 'new_state_idx01'. The mapping rules are as follows:
   1) The start state of the original FSA is mapped to the penultimate state of
   the reversed FSA.
   2) The final state of the original FSA is mapped to the start state of the
   reversed FSA.
   3) The other (median) states of the original FSA are unchanged relatively.

   Note: For each non-empty FSA, we need to add an extra final state to the
   reversed FSA, so an "offset" array is maintained to compute the 'idx01'.
   Bear in mind, the additional final state for each reversed FSA doesn't have
   a corresponding state in original FSA.
   */
  Array1<int32_t> ConstructStateMapping() {
    int32_t num_states = fsas_.TotSize(1);
    Array1<int32_t> old2new_map(c_, num_states);
    int32_t *old2new_map_data = old2new_map.Data();

    Array1<int32_t> offset_idx01 = ExclusiveSum(non_empty_fsas_);
    const int32_t *offset_idx01_data = offset_idx01.Data(),
                  *fsas_row_ids1_data = fsas_.RowIds(1).Data(),
                  *fsas_row_splits1_data = fsas_.RowSplits(1).Data();

    K2_EVAL(
        c_, num_states, lambda_set_old2new_map, (int32_t idx01)->void {
          // The 'idx0' must correspond to a non-empty FSA as it is computed
          // from the row_ids.
          int32_t idx0 = fsas_row_ids1_data[idx01],
                  idx0x = fsas_row_splits1_data[idx0],
                  idx0x_final_state = fsas_row_splits1_data[idx0 + 1] - 1;
          // Initialize as the offset.
          int32_t new_idx01 = offset_idx01_data[idx0];
          if (idx01 == idx0x) {
            new_idx01 += idx0x_final_state;
          } else if (idx01 == idx0x_final_state) {
            new_idx01 += idx0x;
          } else {
            new_idx01 += idx01;
          }
          old2new_map_data[idx01] = new_idx01;
        });
    return old2new_map;
  }


  /*
   Return the RaggedShape of the state-level reversed FsaVec.
   Compared with the original FsaVec, we need to add an extra final state for
   each non-empty FSA.
   */
  RaggedShape CreateReversedStateShape() {
    Array1<int32_t> acc_non_empty_fsas = ExclusiveSum(non_empty_fsas_);
    Array1<int32_t> reversed_row_split1 =
      Plus(acc_non_empty_fsas, fsas_.RowSplits(1));
    return RaggedShape2(&reversed_row_split1, nullptr, -1);
  }


  /*
   Retrun the RaggedShape of the arc-level reversed FsaVec.
   We generate the RaggedShape by figuring out the number of out-going arcs for
   each reversed state. As the original source state will become destination
   state, we collect the information by 'incoming_arcs'.

   All cases will be one-to-one mapping, except that we need to add an extra
   'final_arc' to the penultimate state of the reversed FSA (i.e. the
   corresponding start state in original FSA).
   */
  RaggedShape CreateReversedArcSahpe(Ragged<int32_t> &incoming_arcs,
                                     RaggedShape &reversed_states_shape) {
    int32_t num_ori_states = fsas_.TotSize(1);
    const int32_t *fsas_row_splits1_data = fsas_.RowSplits(1).Data(),
                  *fsas_row_ids1_data = fsas_.RowIds(1).Data(),
                  *incoming_arcs_row_splits2_data =
                    incoming_arcs.RowSplits(2).Data();

    Array1<int32_t> num_arcs_per_reversed_state(
        c_, reversed_states_shape.NumElements() + 1, 0);
    int32_t *num_arcs_per_reversed_state_data =
        num_arcs_per_reversed_state.Data();

    // For each reversed FSA, we just need to add one final state which
    // doesn't have the out-going arcs. So we deal with the each original state.
    const int32_t *state_old2new_map_data = state_old2new_map_.Data();
    K2_EVAL(
        c_, num_ori_states, lambda_set_num_arcs_per_reversed_state,
        (int32_t idx01)->void {
          int32_t idx0 = fsas_row_ids1_data[idx01],
                  idx0x = fsas_row_splits1_data[idx0],
                  num_arcs = (idx01 == idx0x ? 1 : 0);
          num_arcs += incoming_arcs_row_splits2_data[idx01 + 1] -
                      incoming_arcs_row_splits2_data[idx01];
          num_arcs_per_reversed_state_data[state_old2new_map_data[idx01]] =
            num_arcs;
        });
    ExclusiveSum(num_arcs_per_reversed_state, &num_arcs_per_reversed_state);
    return RaggedShape2(&num_arcs_per_reversed_state, nullptr, -1);
  }


  /*
   Return the reversed arcs. Here are two cases.
   1) For each arc in original FSA, we reverse it and keep the 'idx2' position
      unchanged. [e.g. the (arc1, arc2, arc3) goes into a state in original
      FSA. After reversing, the corresponding state will launch the three arcs
      in the same order (reversed_arc1, reversed_arc2, reversed_arc3)].
   2) For each non-empty original FSA, we add an extra "final_arc" which from
      the 'penultimate' reversed state to the 'final' reversed state. It always
      be the last arc for corresponding FSA.
   */
  Array1<Arc> ReverseArcsAndCreateArcMap(Ragged<int32_t> &incoming_arcs,
                                         RaggedShape &reversed_shape,
                                         Array1<int32_t> *arc_map) {
    int32_t num_reversed_arcs = reversed_shape.TotSize(2);
    Array1<Arc> reversed_arcs(c_, num_reversed_arcs);
    Arc *reversed_arcs_data = reversed_arcs.Data();

    // A map from reversed arcs to original arcs. "-1" means no mapping.
    Array1<int32_t> new2old_arcs_map(c_, num_reversed_arcs);
    int32_t *new2old_arcs_map_data = new2old_arcs_map.Data();

    int32_t num_incoming_arcs = incoming_arcs.shape.TotSize(2);
    const int32_t *incoming_arcs_data = incoming_arcs.values.Data(),
                  *incoming_arcs_row_ids2_data = incoming_arcs.RowIds(2).Data(),
                  *incoming_arcs_row_splits2_data =
                    incoming_arcs.RowSplits(2).Data(),
                  *state_old2new_map_data = state_old2new_map_.Data(),
                  *reversed_row_ids2_data = reversed_shape.RowIds(2).Data(),
                  *reversed_row_splits2_data =
                    reversed_shape.RowSplits(2).Data(),
                  *reversed_row_ids1_data = reversed_shape.RowIds(1).Data(),
                  *reversed_row_splits1_data =
                    reversed_shape.RowSplits(1).Data();

    const Arc *fsas_data = fsas_.values.Data();
    const int32_t *fsas_row_ids2_data = fsas_.RowIds(2).Data();
    // Process all existing arcs in orginial FSAs.
    K2_EVAL(
        c_, num_incoming_arcs, lambda_map_original_arcs,
        (int32_t incoming_arcs_idx012)->void {
          int32_t arcs_idx012 = incoming_arcs_data[incoming_arcs_idx012],
                  // Compute the offset.
                  incoming_dest_states_idx01 =
                    incoming_arcs_row_ids2_data[incoming_arcs_idx012],
                  incoming_arcs_idx01x =
                    incoming_arcs_row_splits2_data[incoming_dest_states_idx01],
                  incoming_arcs_idx2 =
                    incoming_arcs_idx012 - incoming_arcs_idx01x,
                  // Get the source state in reversed FSAs according to the
                  // originial dest state.
                  reversed_idx01 =
                    state_old2new_map_data[incoming_dest_states_idx01],
                  // Get the corresponding arc position.
                  reversed_idx01x = reversed_row_splits2_data[reversed_idx01],
                  reversed_idx012 = reversed_idx01x + incoming_arcs_idx2;
          Arc arc = fsas_data[arcs_idx012];
          // Figure out the reversed dest state.
          int32_t arcs_src_idx01 = fsas_row_ids2_data[arcs_idx012],
                  new_arcs_dest_idx01 =
                    state_old2new_map_data[arcs_src_idx01],
                  new_arcs_dest_idx0 =
                    reversed_row_ids1_data[new_arcs_dest_idx01],
                  new_arcs_dest_idx0x =
                    reversed_row_splits1_data[new_arcs_dest_idx0],
                  new_arcs_dest_idx1 =
                    new_arcs_dest_idx01 - new_arcs_dest_idx0x;
          // Figure out the reversed source state.
          int32_t arcs_dest_idx01 = arc.dest_state +
                                    (fsas_row_ids2_data[arcs_idx012] -
                                     arc.src_state),
                  new_arcs_src_idx01 =
                    state_old2new_map_data[arcs_dest_idx01],
                  new_arcs_src_idx0 =
                    reversed_row_ids1_data[new_arcs_src_idx01],
                  new_arcs_src_idx0x =
                    reversed_row_splits1_data[new_arcs_src_idx0],
                  new_arcs_src_idx1 = new_arcs_src_idx01 - new_arcs_src_idx0x;
          Arc new_arc;
          new_arc.src_state = new_arcs_src_idx1;
          new_arc.dest_state = new_arcs_dest_idx1;
          // The original 'final_arcs' will be changed into 'epsilon_arcs'.
          new_arc.label = (arc.label == -1 ? 0 : arc.label);
          new_arc.score = arc.score;
          reversed_arcs_data[reversed_idx012] = new_arc;
          // Deal with arc mapping.
          new2old_arcs_map_data[reversed_idx012] = arcs_idx012;
        });


    int32_t num_fsas = fsas_.Dim0();
    const int32_t *non_empty_fsas_data = non_empty_fsas_.Data();
    // Add an 'final_arc' for each non-empty FSA.
    K2_EVAL(
        c_, num_fsas, lambda_add_external_final_arc, (int32_t i)->void{
          if (non_empty_fsas_data[i]) {
            int32_t new_arcs_dest_idx0x = reversed_row_splits1_data[i],
                    new_arcs_dest_idx01 = reversed_row_splits1_data[i + 1] - 1,
                    new_arcs_dest_idx1 = new_arcs_dest_idx01 -
                                         new_arcs_dest_idx0x;

            Arc new_arc;
            new_arc.src_state = new_arcs_dest_idx1 - 1;
            new_arc.dest_state = new_arcs_dest_idx1;
            new_arc.label = -1;
            new_arc.score = 0;  // TropicalWeight::0ne() and LogWeight::One()
                                // are both 0.

            // This is the last arc for a FSA.
            int32_t new_arcs_idx0x_next = reversed_row_splits1_data[i + 1],
                    new_arcs_idx0xx_next =
                      reversed_row_splits2_data[new_arcs_idx0x_next],
                    new_arcs_idx012 = new_arcs_idx0xx_next - 1;
            reversed_arcs_data[new_arcs_idx012] = new_arc;
            // Deal with arc mapping.
            new2old_arcs_map_data[new_arcs_idx012] = -1;
          }
        });

    if (arc_map != nullptr)
      *arc_map = std::move(new2old_arcs_map);
    return reversed_arcs;
  }

  /*
    Does the main work of reversing and returns the resulting FSAs.
      @param [out] arc_map    if non-NULL, the map from (arcs in output)
                              to (corresponding arcs in input) is written here.
      @return    Returns the reversed FsaVec.
   */
  FsaVec Reverse(Array1<int32_t> *arc_map) {
    NVTX_RANGE(K2_FUNC);
    // Records the dest-state of each arc in fsas_ as an idx01.
    Array1<int32_t> dest_states_idx01 = GetDestStates(fsas_, true);
    // The 'incoming_arcs' records the incoming arcs information for each
    // dest_state_idx01. The length of its value equals to the number of arcs,
    // but it isn't indexed by arc_idx012.
    // The structure is [fsa][state][list_of_arcs]. It shows the arc_idx012 for
    // each dest-state.
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsas_, dest_states_idx01);

    // Mark the non-empty FSAs
    non_empty_fsas_ = CollectNonEmptyFsas();
    // Construct a state map from old_state_idx01 to reversed new_state_idx01
    state_old2new_map_ = ConstructStateMapping();

    // Construct the reversed shape
    RaggedShape reversed_states_shape = CreateReversedStateShape();
    RaggedShape reversed_arcs_shape = CreateReversedArcSahpe(
        incoming_arcs, reversed_states_shape);
    RaggedShape reversed_shape = ComposeRaggedShapes(
        reversed_states_shape, reversed_arcs_shape);
    // Reverse the arcs
    Array1<Arc> reversed_arcs_value = ReverseArcsAndCreateArcMap(
        incoming_arcs, reversed_shape, arc_map);

    return Ragged<Arc>(reversed_shape, reversed_arcs_value);
  }

  ContextPtr c_;
  FsaVec &fsas_;

  // The map from old_state_idx01 to reversed new_state_idx01.
  Array1<int32_t> state_old2new_map_;

  // Act as a flag that indicates whether an FSA is non-empty and legal.
  // When reversing, we will add an extra final state and final arc for each
  // non-empty FSA.
  Array1<int32_t> non_empty_fsas_;
};

void Reverse(FsaVec &src, FsaVec *dest, Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK_LE(src.NumAxes(), 3);
  if (src.NumAxes() == 2) {
    // Turn single Fsa into FsaVec.
    FsaVec src_vec = FsaToFsaVec(src), dest_vec;
    // Recurse..
    Reverse(src_vec, &dest_vec, arc_map);
    *dest = GetFsaVecElement(dest_vec, 0);
    return;
  }
  Reverser reverser(src);
  *dest = reverser.Reverse(arc_map);
}

}  // namespace k2
