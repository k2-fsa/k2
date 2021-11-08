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

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

FsaClass CtcTopo(int32_t max_token, bool modified /*= false*/,
                 torch::Device device /*=torch::kCPU*/) {
  Array1<int32_t> aux_labels;
  auto ctx = ContextFromDevice(device);
  Fsa fsa = CtcTopo(ctx, max_token, modified, &aux_labels);
  FsaClass dest(fsa);
  dest.SetAttr("aux_labels", torch::IValue(Array1ToTorch<int32_t>(aux_labels)));
  return dest;
}

FsaClass IntersectDensePruned(FsaClass &graph, DenseFsaVec &dense,
                              float search_beam, float output_beam,
                              int32_t min_activate_states,
                              int32_t max_activate_states) {
  Array1<int32_t> graph_arc_map;
  Array1<int32_t> dense_arc_map;
  FsaVec fsa;
  IntersectDensePruned(graph.fsa, dense, search_beam, output_beam,
                       min_activate_states, max_activate_states, &fsa,
                       &graph_arc_map, &dense_arc_map);
  FsaClass dest(fsa);
  dest.CopyAttrs(graph, Array1ToTorch<int32_t>(graph_arc_map));
  return dest;
}

FsaClass ShortestPath(FsaClass &lattice) {
  Ragged<int32_t> state_batches = GetStateBatches(lattice.fsa, true);
  Array1<int32_t> dest_states = GetDestStates(lattice.fsa, true);
  Ragged<int32_t> incoming_arcs = GetIncomingArcs(lattice.fsa, dest_states);
  Ragged<int32_t> entering_arc_batches =
      GetEnteringArcIndexBatches(lattice.fsa, incoming_arcs, state_batches);

  bool log_semiring = false;
  Array1<int32_t> entering_arcs;
  GetForwardScores<float>(lattice.fsa, state_batches, entering_arc_batches,
                          log_semiring, &entering_arcs);

  Ragged<int32_t> best_path_arc_indexes =
      ShortestPath(lattice.fsa, entering_arcs);

  FsaVec out = FsaVecFromArcIndexes(lattice.fsa, best_path_arc_indexes);
  torch::Tensor arc_map = Array1ToTorch<int32_t>(best_path_arc_indexes.values);
  return FsaClass::FromUnaryFunctionTensor(lattice, out, arc_map);
}

void Invert(FsaClass *lattice) {
  K2_CHECK_NE(lattice, nullptr);
  K2_CHECK(lattice->HasAttr("aux_labels"));

  if (lattice->HasTensorAttr("aux_labels")) {
    // The invert is trivial, just swap the labels and aux_labels.
    // No new arcs are added.
    auto aux_labels = lattice->GetTensorAttr("aux_labels").clone();
    lattice->SetTensorAttr("aux_labels", lattice->Labels().clone());
    lattice->SetLabels(aux_labels);
    return;
  }

  K2_CHECK(lattice->HasRaggedTensorAttr("aux_labels"));
  Ragged<int32_t> src_aux_labels = lattice->GetRaggedTensorAttr("aux_labels");

  Fsa dest;
  Ragged<int32_t> dest_aux_labels;
  Array1<int32_t> arc_map;
  Invert(lattice->fsa, src_aux_labels, &dest, &dest_aux_labels, &arc_map);

  // `label` is the 3rd field of struct Arc.
  FixFinalLabels(dest, reinterpret_cast<int32_t *>(dest.values.Data()) + 2, 4);

  lattice->DeleteAttr("aux_labels");
  lattice->properties = 0;
  lattice->fsa = dest;
  lattice->CopyAttrs(*lattice, Array1ToTorch(arc_map));
  lattice->SetRaggedTensorAttr("aux_labels", dest_aux_labels);
}

void ArcSort(FsaClass *lattice) {
  Fsa dest;
  Array1<int32_t> arc_map;
  ArcSort(lattice->fsa, &dest, &arc_map);
  lattice->properties = 0;
  lattice->fsa = dest;
  lattice->CopyAttrs(*lattice, Array1ToTorch(arc_map));
}

void Connect(FsaClass *lattice) {
  Fsa dest;
  Array1<int32_t> arc_map;
  Connect(lattice->fsa, &dest, &arc_map);
  lattice->properties = 0;
  lattice->fsa = dest;
  lattice->CopyAttrs(*lattice, Array1ToTorch(arc_map));
}

void TopSort(FsaClass *lattice) {
  Fsa dest;
  Array1<int32_t> arc_map;
  TopSort(lattice->fsa, &dest, &arc_map);
  lattice->properties = 0;
  lattice->fsa = dest;
  lattice->CopyAttrs(*lattice, Array1ToTorch(arc_map));
}

}  // namespace k2
