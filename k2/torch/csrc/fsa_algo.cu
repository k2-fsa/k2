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
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

FsaClass CtcTopo(int32_t max_token, bool modified /*= false*/,
                 torch::Device device /*=torch::kCPU*/) {
  Array1<int32_t> aux_labels;
  auto ctx = ContextFromDevice(device);
  Fsa fsa = CtcTopo(ctx, max_token, modified, &aux_labels);
  FsaClass dest(fsa);
  dest.SetTensorAttr("aux_labels", Array1ToTorch(aux_labels));
  return dest;
}

FsaClass TrivialGraph(int32_t max_token,
                      torch::Device device /*=torch::kCPU*/) {
  Array1<int32_t> aux_labels;
  auto ctx = ContextFromDevice(device);
  Fsa fsa = TrivialGraph(ctx, max_token, &aux_labels);
  FsaClass dest(fsa);
  dest.SetTensorAttr("aux_labels", Array1ToTorch(aux_labels));
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
  dest.CopyAttrs(graph, Array1ToTorch(graph_arc_map));
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
  torch::Tensor arc_map = Array1ToTorch(best_path_arc_indexes.values);
  return FsaClass::FromUnaryFunctionTensor(lattice, out, arc_map);
}

void Invert(FsaClass *lattice) {
  K2_CHECK_NE(lattice, nullptr);

  if (lattice->HasTensorAttr("aux_labels")) {
    // The invert is trivial, just swap the labels and aux_labels.
    // No new arcs are added.
    auto aux_labels = lattice->GetTensorAttr("aux_labels").clone();
    auto labels = lattice->Labels().clone();

    // FixFinalLabels
    auto minus_one =
        torch::tensor(-1, torch::device(labels.device()).dtype(labels.dtype()));
    aux_labels = torch::where(labels == -1, minus_one, aux_labels);

    lattice->SetTensorAttr("aux_labels", labels);
    lattice->SetLabels(aux_labels);
  } else {
    K2_CHECK(lattice->HasRaggedTensorAttr("aux_labels"));
    Ragged<int32_t> src_aux_labels = lattice->GetRaggedTensorAttr("aux_labels");

    Fsa dest;
    Ragged<int32_t> dest_aux_labels;
    Array1<int32_t> arc_map;
    Invert(lattice->fsa, src_aux_labels, &dest, &dest_aux_labels, &arc_map);

    // `label` is the 3rd field of struct Arc.
    FixFinalLabels(dest, reinterpret_cast<int32_t *>(dest.values.Data()) + 2,
                   4);

    lattice->DeleteRaggedTensorAttr("aux_labels");
    lattice->properties = 0;
    lattice->fsa = dest;
    lattice->CopyAttrs(*lattice, Array1ToTorch(arc_map));
    lattice->SetRaggedTensorAttr("aux_labels", dest_aux_labels);
  }
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

Nbest RandomPaths(FsaClass &lattice, int32_t num_paths) {
  auto &fsas = lattice.fsa;
  Ragged<int32_t> state_batches = GetStateBatches(fsas, /*transpose*/ true);
  Array1<int32_t> dest_states = GetDestStates(fsas, /*as_idx01*/ true);

  Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsas, dest_states);

  Ragged<int32_t> entering_arc_batches =
      GetEnteringArcIndexBatches(fsas, incoming_arcs, state_batches);

  Ragged<int32_t> leaving_arc_batches =
      GetLeavingArcIndexBatches(fsas, state_batches);
  bool log_semiring = true;

  using FloatType = float;
  Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
      fsas, state_batches, entering_arc_batches, log_semiring, nullptr);

  Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
      fsas, state_batches, leaving_arc_batches, log_semiring);

  Array1<FloatType> arc_post =
      GetArcPost(fsas, forward_scores, backward_scores);

  Array1<FloatType> arc_cdf = GetArcCdf(fsas, arc_post);

  Array1<FloatType> tot_scores = GetTotScores(fsas, forward_scores);

  // paths has three axes [utt][path][arc_pos]
  Ragged<int32_t> paths =
      RandomPaths(fsas, arc_cdf, num_paths, tot_scores, state_batches);

  bool has_ragged_aux_labels = true;

  // word_seqs has three axes [utt][path[word_id]
  Ragged<int32_t> word_seqs;
  if (lattice.HasTensorAttr("aux_labels")) {
    has_ragged_aux_labels = false;
    // Index a tensor with a ragged index
    // see Index() in k2/csrc/ragged_ops.h
    auto &aux_labels = lattice.GetTensorAttr("aux_labels");
    Array1<int32_t> aux_labels_array = Array1FromTorch<int32_t>(aux_labels);
    word_seqs = Index(aux_labels_array, paths);
  } else {
    K2_CHECK(lattice.HasRaggedTensorAttr("aux_labels"));
    auto &aux_labels = lattice.GetRaggedTensorAttr("aux_labels");
    // Index a ragged tensor with a ragged index
    // see Index() in k2/csrc/ragged_ops.h
    bool remove_axis = true;
    word_seqs = Index(aux_labels, paths, remove_axis);
  }

  word_seqs = RemoveValuesLeq(word_seqs, 0);

  // Each utterance has `num_paths` paths but some of them transduces
  // to the same word sequence, so we need to remove repeated word
  // sequences within an utterance. After removing repeats, each utterance
  // contains different number of paths
  //
  // `new2old` maps from the output path index to the input path index.
  Array1<int32_t> new2old_indexes;
  (void)UniqueSequences(word_seqs, nullptr, &new2old_indexes);

  // Index a ragged tensor with a tensor
  // See Index() in k2/csrc/ragged_ops.h
  //
  // kept_paths has axes [utt][path][arc_pos]
  Ragged<int32_t> kept_paths = Index(paths, /*axis*/ 1, new2old_indexes);

  // utt_to_path_shape has axes [utt][path]
  RaggedShape utt_to_path_shape = GetLayer(kept_paths.shape, 0);

  // Remove the utterance axis.
  kept_paths = kept_paths.RemoveAxis(0);
  // Now kept_paths has only two axes [path][arc_pos]

  // labels has 2 axes [path][token_id]
  // Note that it contains -1s.
  //
  // Index a tensor with a ragged index
  // see Index() in k2/csrc/ragged_ops.h
  auto lattice_labels = lattice.Labels();
  auto lattice_labels_array =
      Array1FromTorch<int32_t>(lattice_labels.contiguous());
  Ragged<int32_t> labels = Index(lattice_labels_array, kept_paths);

  // Remove -1 from labels as we will use it to construct a linear FSA
  labels = RemoveValuesEq(labels, -1);
  Fsa dest = LinearFsas(labels);
  FsaClass ans_lattice(dest);
  if (has_ragged_aux_labels) {
    auto &aux_labels = lattice.GetRaggedTensorAttr("aux_labels");
    // Index a ragged tensor with a tensor
    // See Index() in k2/csrc/ragged_ops.h
    Ragged<int32_t> ans_aux_labels =
        Index(aux_labels, /*axis*/ 0, kept_paths.values);
    ans_lattice.SetRaggedTensorAttr("aux_labels", ans_aux_labels);
  } else {
    auto &aux_labels = lattice.GetTensorAttr("aux_labels");
    Array1<int32_t> aux_labels_array = Array1FromTorch<int32_t>(aux_labels);
    // Index a tensor with a tensor index
    // See Index() in k2/csrc/array_ops.h
    Array1<int32_t> ans_aux_labels = Index(aux_labels_array, kept_paths.values,
                                           false,  // allow_minus_one
                                           0);     // default value
    ans_lattice.SetTensorAttr("aux_labels", Array1ToTorch(ans_aux_labels));
  }

  return {ans_lattice, utt_to_path_shape};
}

FsaClass IntersectDevice(FsaClass &a_fsas, FsaClass &b_fsas,
                         const Array1<int32_t> &b_to_a_map,
                         bool sorted_match_a) {
  Array1<int32_t> arc_map_a, arc_map_b;

  Fsa c_fsas = IntersectDevice(a_fsas.fsa, a_fsas.Properties(), b_fsas.fsa,
                               b_fsas.Properties(), b_to_a_map, &arc_map_a,
                               &arc_map_b, sorted_match_a);

  FsaClass ans(c_fsas);
  ans.CopyAttrs(a_fsas, Array1ToTorch(arc_map_a));
  ans.CopyAttrs(b_fsas, Array1ToTorch(arc_map_b));
  return ans;
}

FsaClass LinearFsaWithSelfLoops(FsaClass &fsas) {
  RaggedShape shape;
  if (fsas.fsa.NumAxes() == 2) {
    // A single Fsa
    auto shape0 =
        RegularRaggedShape(fsas.fsa.Context(), 1, fsas.fsa.TotSize(0));
    shape = ComposeRaggedShapes(shape0, fsas.fsa.shape);
  } else {
    shape = fsas.fsa.shape;
  }

  shape = RemoveAxis(shape, 1);  // remove the state axis

  auto labels = Ragged<int32_t>(
      shape, Array1FromTorch<int32_t>(fsas.Labels().contiguous()));
  labels = RemoveValuesLeq(labels, 0);

  auto linear_fsa = LinearFsas(labels);
  FsaVec ans;
  AddEpsilonSelfLoops(linear_fsa, &ans);

  if (fsas.fsa.NumAxes() == 2) ans = ans.RemoveAxis(0);
  return FsaClass(ans);
}

}  // namespace k2
