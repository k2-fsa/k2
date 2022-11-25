/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <utility>

#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/rnnt_decode.h"
#include "k2/csrc/torch_util.h"
#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/torch_api.h"

namespace k2 {

void ExclusiveSum(torch::Tensor src, torch::Tensor *dst) {
  Array1<int32_t> src_arr = FromTorch<int32_t>(src);
  Array1<int32_t> dst_arr = FromTorch<int32_t>(*dst);

  ExclusiveSum(src_arr, &dst_arr);
}

RaggedShapePtr RaggedShape2(torch::Tensor row_splits, torch::Tensor row_ids,
                            int32_t cached_tot_size /*=-1*/) {
  if (!row_splits.defined()) {
    K2_CHECK(row_ids.defined())
        << "You have to provide row_ids if row_splits is empty";
  }

  Array1<int32_t> row_splits_arr, row_ids_arr;

  if (row_splits.defined()) {
    row_splits_arr = FromTorch<int32_t>(row_splits);
  }

  if (row_ids.defined()) {
    row_ids_arr = FromTorch<int32_t>(row_ids);
  }

  return std::make_shared<RaggedShape>(RaggedShape2(
      row_splits.defined() ? &row_splits_arr : nullptr,
      row_ids.defined() ? &row_ids_arr : nullptr, cached_tot_size));
}

int32_t TotSize(RaggedShapePtr shape, int32_t axis) {
  return shape->TotSize(axis);
}

torch::Tensor RowIds(RaggedShapePtr shape, int32_t axis) {
  return ToTorch(shape->RowIds(axis));
}

torch::Tensor RowSplits(RaggedShapePtr shape, int32_t axis) {
  return ToTorch(shape->RowSplits(axis));
}

FsaClassPtr GetCtcTopo(int32_t max_token, bool modified, torch::Device device) {
  return std::make_shared<FsaClass>(CtcTopo(max_token, modified, device));
}

FsaClassPtr GetTrivialGraph(int32_t max_token,
                            torch::Device device /*=torch::kCPU*/) {
  return std::make_shared<FsaClass>(TrivialGraph(max_token, device));
}

FsaClassPtr LoadFsaClass(const std::string &filename,
                         torch::Device map_location) {
  return std::make_shared<FsaClass>(LoadFsa(filename, map_location));
}

FsaClassPtr GetLattice(torch::Tensor log_softmax_out,
                       torch::Tensor log_softmax_out_lens,
                       FsaClassPtr decoding_graph, float search_beam,
                       float output_beam, int32_t min_activate_states,
                       int32_t max_activate_states,
                       int32_t subsampling_factor) {
  int32_t num_sequences = log_softmax_out.size(0);
  K2_CHECK_EQ(num_sequences, log_softmax_out_lens.size(0))
      << "The number of sequences should be equal, given " << num_sequences
      << " vs " << log_softmax_out_lens.size(0);

  torch::Tensor supervision_segments =
      torch::stack({torch::arange(num_sequences, torch::kInt),
                    torch::zeros({num_sequences}, torch::kInt),
                    log_softmax_out_lens.to(torch::kInt)},
                   1)
          .to(torch::kCPU);

  FsaClass lattice =
      GetLattice(log_softmax_out, *decoding_graph, supervision_segments,
                 search_beam, output_beam, min_activate_states,
                 max_activate_states, subsampling_factor);

  return std::make_shared<FsaClass>(lattice);
}

std::vector<std::vector<int32_t>> BestPath(const FsaClassPtr &lattice) {
  FsaClass paths = ShortestPath(*lattice);
  auto ragged_aux_labels = GetTexts(paths);
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();
  return aux_labels_vec;
}

FsaClassPtr ShortestPath(const FsaClassPtr &lattice) {
  FsaClass path = ShortestPath(*lattice);
  return std::make_shared<FsaClass>(path);
}

void ScaleTensorAttribute(FsaClassPtr &fsa, float scale,
                          const std::string &attribute) {
  if (attribute == "scores") {
    auto scores = fsa->Scores();
    scores = scores * scale;
    fsa->SetScores(scores);
    return;
  }
  K2_CHECK(fsa->HasTensorAttr(attribute))
      << "The given Fsa doesn't has the attribute : " << attribute;
  auto old_value = fsa->GetTensorAttr(attribute);
  K2_CHECK(old_value.scalar_type() == torch::kFloat ||
           old_value.scalar_type() == torch::kDouble)
      << "Only support scaling float type attributes, the type of given "
         "attribute : "
      << attribute << " is " << old_value.scalar_type();
  fsa->SetTensorAttr(attribute, old_value * scale);
}

torch::Tensor GetTensorAttr(FsaClassPtr &fsa, const std::string &attribute) {
  if (attribute == "labels") {
    return fsa->Labels();
  } else if (attribute == "scores") {
    return fsa->Scores();
  }

  K2_CHECK(fsa->HasTensorAttr(attribute))
      << "The given Fsa doesn't has the attribute : " << attribute;
  return fsa->GetTensorAttr(attribute);
}

void SetTensorAttr(FsaClassPtr &fsa, const std::string &attribute,
                   torch::Tensor value) {
  if (attribute == "labels") {
    fsa->SetLabels(value);
  } else if (attribute == "scores") {
    fsa->SetScores(value);
  } else {
    fsa->SetTensorAttr(attribute, value);
  }
}

// A wrapper for RnntDecodingStream which can connect the RnntDecodingStream
// with source Fsa graph, we need this graph to do attribute propagation.
struct RnntStream {
  std::shared_ptr<rnnt_decoding::RnntDecodingStream> stream;
  FsaClassPtr graph;
};

// A wrapper for RnntDecodingStreams which can connect the RnntDecodingStreams
// with source streams.
struct RnntStreams {
  std::shared_ptr<rnnt_decoding::RnntDecodingStreams> streams;
  std::vector<std::shared_ptr<RnntStream>> src_streams;
};

RnntStreamPtr CreateRnntStream(FsaClassPtr decoding_graph) {
  RnntStream rnnt_stream;
  rnnt_stream.graph = decoding_graph;
  rnnt_stream.stream =
      rnnt_decoding::CreateStream(std::make_shared<Fsa>(decoding_graph->fsa));
  return std::make_shared<RnntStream>(rnnt_stream);
}

RnntStreamsPtr CreateRnntStreams(
    const std::vector<RnntStreamPtr> &source_streams, int32_t vocab_size,
    int32_t context_size, float beam, int32_t max_contexts,
    int32_t max_states) {
  std::vector<std::shared_ptr<rnnt_decoding::RnntDecodingStream>> raw_streams(
      source_streams.size());
  for (size_t i = 0; i < raw_streams.size(); ++i) {
    raw_streams[i] = source_streams[i]->stream;
  }
  rnnt_decoding::RnntDecodingConfig config(vocab_size, context_size, beam,
                                           max_states, max_contexts);
  RnntStreams rnnt_streams;
  rnnt_streams.src_streams = source_streams;
  auto streams = rnnt_decoding::RnntDecodingStreams(raw_streams, config);
  rnnt_streams.streams =
      std::make_shared<rnnt_decoding::RnntDecodingStreams>(streams);
  return std::make_shared<RnntStreams>(rnnt_streams);
}

std::pair<RaggedShapePtr, torch::Tensor> GetRnntContexts(
    RnntStreamsPtr rnnt_streams) {
  RaggedShape shape;
  Array2<int32_t> contexts;
  rnnt_streams->streams->GetContexts(&shape, &contexts);
  torch::Tensor contexts_tensor = ToTorch<int32_t>(contexts);
  return std::make_pair(std::make_shared<RaggedShape>(shape), contexts_tensor);
}

void AdvanceRnntStreams(RnntStreamsPtr rnnt_streams, torch::Tensor log_probs) {
  log_probs = log_probs.to(torch::kFloat);
  Array2<float> log_probs_array = FromTorch<float>(log_probs, Array2Tag{});
  rnnt_streams->streams->Advance(log_probs_array);
}

void TerminateAndFlushRnntStreams(RnntStreamsPtr rnnt_streams) {
  rnnt_streams->streams->TerminateAndFlushToStreams();
}

FsaClassPtr FormatOutput(RnntStreamsPtr rnnt_streams,
                         std::vector<int32_t> num_frames, bool allow_partial) {
  FsaVec ofsa;
  Array1<int32_t> array_map;
  rnnt_streams->streams->FormatOutput(num_frames, allow_partial, &ofsa,
                                      &array_map);
  auto arc_map = Ragged<int32_t>(ofsa.shape, array_map).RemoveAxis(1);
  std::vector<FsaClass> graphs(rnnt_streams->src_streams.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    graphs[i] = *(rnnt_streams->src_streams[i]->graph);
  }
  FsaClass lattice(ofsa);
  lattice.CopyAttrs(graphs, arc_map);
  return std::make_shared<FsaClass>(lattice);
}

}  // namespace k2
