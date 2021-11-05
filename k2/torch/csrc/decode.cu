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
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

FsaClass GetLattice(torch::Tensor nnet_output, FsaClass &decoding_graph,
                    torch::Tensor supervision_segments, float search_beam,
                    float output_beam, int32_t min_activate_states,
                    int32_t max_activate_states, int32_t subsampling_factor) {
  DenseFsaVec dense_fsa_vec = CreateDenseFsaVec(
      nnet_output, supervision_segments, subsampling_factor - 1);
  return IntersectDensePruned(decoding_graph, dense_fsa_vec, search_beam,
                              output_beam, min_activate_states,
                              max_activate_states);
}

Ragged<int32_t> GetTexts(FsaClass &lattice) {
  K2_CHECK(lattice.HasAttr("aux_labels"));
  Ragged<int32_t> ragged_aux_labels;
  torch::IValue aux_labels = lattice.GetAttr("aux_labels");
  if (aux_labels.isTensor()) {
    Array1<int32_t> aux_labels_array =
        Array1FromTorch<int32_t>(aux_labels.toTensor());
    RaggedShape aux_labels_shape = RemoveAxis(lattice.fsa.shape, 1);
    ragged_aux_labels = Ragged<int32_t>(aux_labels_shape, aux_labels_array);
  } else {
    K2_CHECK(IsRaggedInt(aux_labels));
    Ragged<int32_t> in_ragged_aux_labels = ToRaggedInt(aux_labels);
    RaggedShape aux_labels_shape =
        ComposeRaggedShapes(lattice.fsa.shape, in_ragged_aux_labels.shape);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    ragged_aux_labels =
        Ragged<int32_t>(aux_labels_shape, in_ragged_aux_labels.values);
  }
  ragged_aux_labels = RemoveValuesLeq(ragged_aux_labels, 0);
  return ragged_aux_labels;
}

}  // namespace k2
