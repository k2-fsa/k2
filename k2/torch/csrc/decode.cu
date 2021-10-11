/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"

namespace k2 {

FsaVec GetLattice(torch::Tensor nnet_output, FsaVec decoding_graph,
                  torch::Tensor supervision_segments, float search_beam,
                  float output_beam, int32_t min_activate_states,
                  int32_t max_activate_states, int32_t subsampling_factor,
                  Array1<int32_t> &in_aux_labels,
                  Ragged<int32_t> &in_ragged_aux_labels,
                  Array1<int32_t> *out_aux_labels,
                  Ragged<int32_t> *out_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

  DenseFsaVec dense_fsa_vec = CreateDenseFsaVec(
      nnet_output, supervision_segments, subsampling_factor - 1);

  FsaVec lattice;
  Array1<int32_t> arc_map_a;
  Array1<int32_t> arc_map_b;
  IntersectDensePruned(decoding_graph, dense_fsa_vec, search_beam, output_beam,
                       min_activate_states, max_activate_states, &lattice,
                       &arc_map_a, &arc_map_b);
  if (in_aux_labels.Dim() > 0) {
    // see Index() in array_ops.h
    *out_aux_labels = Index(in_aux_labels, arc_map_a, /*allow_minus_one*/ false,
                            /*default_value*/ 0);
  } else {
    // See Index() in ragged_ops.h
    *out_ragged_aux_labels = Index(in_ragged_aux_labels, /*axis*/ 0, arc_map_a);
  }
  return lattice;
}

FsaVec OneBestDecoding(FsaVec &lattice, Array1<int32_t> &in_aux_labels,
                       Ragged<int32_t> &in_ragged_aux_labels,
                       Array1<int32_t> *out_aux_labels,
                       Ragged<int32_t> *out_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

  Ragged<int32_t> state_batches = GetStateBatches(lattice, true);
  Array1<int32_t> dest_states = GetDestStates(lattice, true);
  Ragged<int32_t> incoming_arcs = GetIncomingArcs(lattice, dest_states);
  Ragged<int32_t> entering_arc_batches =
      GetEnteringArcIndexBatches(lattice, incoming_arcs, state_batches);

  bool log_semiring = false;
  Array1<int32_t> entering_arcs;
  GetForwardScores<float>(lattice, state_batches, entering_arc_batches,
                          log_semiring, &entering_arcs);

  Ragged<int32_t> best_path_arc_indexes = ShortestPath(lattice, entering_arcs);

  if (in_aux_labels.Dim() > 0) {
    *out_aux_labels = Index(in_aux_labels, best_path_arc_indexes.values,
                            /*allow_minus_one*/ false,
                            /*default_value*/ 0);
  } else {
    *out_ragged_aux_labels =
        Index(in_ragged_aux_labels, /*axis*/ 0, best_path_arc_indexes.values);
  }

  FsaVec out = FsaVecFromArcIndexes(lattice, best_path_arc_indexes);
  return out;
}

Ragged<int32_t> GetTexts(FsaVec &lattice, Array1<int32_t> &in_aux_labels,
                         Ragged<int32_t> &in_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

  Ragged<int32_t> ragged_aux_labels;
  if (in_aux_labels.Dim() != 0) {
    // [utt][state][arc] -> [utt][arc]
    RaggedShape aux_labels_shape = RemoveAxis(lattice.shape, 1);
    ragged_aux_labels = Ragged<int32_t>(aux_labels_shape, in_aux_labels);
  } else {
    RaggedShape aux_labels_shape =
        ComposeRaggedShapes(lattice.shape, in_ragged_aux_labels.shape);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    ragged_aux_labels =
        Ragged<int32_t>(aux_labels_shape, in_ragged_aux_labels.values);
  }
  ragged_aux_labels = RemoveValuesLeq(ragged_aux_labels, 0);
  return ragged_aux_labels;
}

}  // namespace k2
