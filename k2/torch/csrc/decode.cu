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
  if (lattice.HasTensorAttr("aux_labels")) {
    torch::Tensor aux_labels = lattice.GetTensorAttr("aux_labels");
    Array1<int32_t> aux_labels_array = Array1FromTorch<int32_t>(aux_labels);
    RaggedShape aux_labels_shape = RemoveAxis(lattice.fsa.shape, 1);
    auto ragged_aux_labels =
        Ragged<int32_t>(aux_labels_shape, aux_labels_array);
    return RemoveValuesLeq(ragged_aux_labels, 0);
  } else {
    K2_CHECK(lattice.HasRaggedTensorAttr("aux_labels"));

    auto aux_labels = lattice.GetRaggedTensorAttr("aux_labels");
    RaggedShape aux_labels_shape =
        ComposeRaggedShapes(lattice.fsa.shape, aux_labels.shape);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    aux_labels_shape = RemoveAxis(aux_labels_shape, 1);
    auto ragged_aux_labels =
        Ragged<int32_t>(aux_labels_shape, aux_labels.values);
    return RemoveValuesLeq(ragged_aux_labels, 0);
  }
}

void WholeLatticeRescoring(FsaClass &G, float ngram_lm_scale,
                           FsaClass *lattice) {
  K2_CHECK(lattice->HasTensorAttr("lm_scores"));

  torch::Tensor am_scores =
      lattice->Scores() - lattice->GetTensorAttr("lm_scores");
  lattice->SetScores(am_scores);

  // Now, lattice contains only acoustic scores, we will attach LM scores
  // from the given n-gram LM
  lattice->DeleteTensorAttr("lm_scores");

  K2_CHECK_EQ(G.NumAttrs(), 1)
      << "G is expected to contain only 1 attribute: lm_scores.";
  K2_CHECK_EQ(G.fsa.NumAxes(), 3);
  K2_CHECK_EQ(G.fsa.Dim0(), 1);

  k2::Invert(lattice);
  // Now lattice has word IDs as labels and token IDs as aux_labels.

  // TODO(fangjun): Use Intersect() when device is CPU
  auto b_to_a_map =
      k2::Array1<int32_t>(G.fsa.Context(), lattice->fsa.Dim0(), 0);
  k2::Array1<int32_t> arc_map_a, arc_map_b;

  k2::Fsa dest = k2::IntersectDevice(G.fsa, G.Properties(), lattice->fsa,
                                     lattice->Properties(), b_to_a_map,
                                     &arc_map_a, &arc_map_b, true);

  lattice->properties = 0;
  lattice->fsa = dest;
  lattice->CopyAttrs(*lattice, k2::Array1ToTorch(arc_map_b));
  lattice->CopyAttrs(G, k2::Array1ToTorch(arc_map_a));
  k2::Connect(lattice);
  k2::TopSort(lattice);
  k2::Invert(lattice);
  // Now lattice has token IDs as labels and word IDs as aux_labels

  if (ngram_lm_scale != 1) {
    torch::Tensor lm_scores = lattice->GetTensorAttr("lm_scores");
    am_scores = lattice->Scores() - lm_scores;
    torch::Tensor scores = am_scores / ngram_lm_scale + lm_scores;
    lattice->SetScores(scores);
  }
}

void DecodeOneChunk(rnnt_decoding::RnntDecodingStreams &streams,
                    torch::jit::script::Module module,
                    torch::Tensor encoder_outs) {
  K2_CHECK_EQ(encoder_outs.dim(), 3);
  K2_CHECK_EQ(streams.NumStreams(), encoder_outs.size(0));
  int32_t T = encoder_outs.size(1);
  for (int32_t t = 0; t < T; ++t) {
    RaggedShape shape;
    Array2<int32_t> contexts;
    streams.GetContexts(&shape, &contexts);
    auto contexts_tensor = Array2ToTorch<int32_t>(contexts);
    contexts_tensor = contexts_tensor.to(torch::kInt64);
    auto decoder_outs =
        module.run_method("decoder_forward", contexts_tensor).toTensor();
    auto current_encoder_outs = encoder_outs.index(
        {torch::indexing::Slice(), torch::indexing::Slice(t, t + 1),
         torch::indexing::Slice()});
    auto row_ids = Array1ToTorch<int32_t>(shape.RowIds(1));
    current_encoder_outs =
        torch::index_select(current_encoder_outs, 0, row_ids);
    auto logits =
        module.run_method("joiner_forward", current_encoder_outs, decoder_outs)
            .toTensor();
    logits = logits.squeeze(1).squeeze(1);
    auto logprobs = logits.log_softmax(-1);
    auto logprobs_array = Array2FromTorch<float>(logprobs);
    streams.Advance(logprobs_array);
  }
  streams.TerminateAndFlushToStreams();
}

}  // namespace k2
