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
#if HAVE_K2_TORCH_API_H == 1
#include "k2/torch_api.h"  // for third party library
#else
#include "k2/torch/csrc/torch_api.h"
#endif

#include "gtest/gtest.h"
#include "torch/torch.h"

namespace k2 {

TEST(ExclusiveSum, SharedMemory) {
  auto src = torch::tensor({2, 3, 0}, torch::kInt);
  ExclusiveSum(src, &src);

  auto expected = torch::tensor({0, 2, 5}, torch::kInt);
  EXPECT_TRUE(torch::allclose(src, expected));
}

TEST(ExclusiveSum, NotSharedMemory) {
  auto src = torch::tensor({2, 3, 0}, torch::kInt);
  auto dst = torch::empty_like(src);
  ExclusiveSum(src, &dst);

  auto expected = torch::tensor({0, 2, 5}, torch::kInt);
  EXPECT_TRUE(torch::allclose(dst, expected));
}

TEST(FsaAttribute, ScaleTensorAttribute) {
  auto fsa = GetTrivialGraph(5);  // This graph has 6 arcs.
  auto value = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, torch::kFloat32);

  SetTensorAttr(fsa, "scores", value);
  SetTensorAttr(fsa, "lm_scores", value);
  ScaleTensorAttribute(fsa, 2.0, "scores");
  ScaleTensorAttribute(fsa, -1.0, "lm_scores");

  EXPECT_TRUE(torch::allclose(
      GetTensorAttr(fsa, "scores"),
      torch::tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, torch::kFloat32)));
  EXPECT_TRUE(torch::allclose(
      GetTensorAttr(fsa, "lm_scores"),
      torch::tensor({-1.0, -2.0, -3.0, -4.0, -5.0, -6.0}, torch::kFloat32)));
}

TEST(RaggedShape, TestAllWrappedMethods) {
  auto sizes = torch::tensor({1, 3, 2, 0}, torch::kInt);
  auto row_splits = torch::empty_like(sizes);
  ExclusiveSum(sizes, &row_splits);

  auto shape = RaggedShape2(row_splits, torch::Tensor(), 6);
  EXPECT_EQ(TotSize(shape, 0), 3);
  EXPECT_EQ(TotSize(shape, 1), 6);

  auto expected_row_ids = torch::tensor({0, 1, 1, 1, 2, 2}, torch::kInt);
  EXPECT_TRUE(torch::allclose(RowIds(shape, 1), expected_row_ids));

  EXPECT_TRUE(torch::allclose(RowSplits(shape, 1), row_splits));
}

// This test does not do any checking, just to confirm it runs normally.
TEST(CtcDecode, TestBasicCtcDecode) {
  namespace F = torch::nn::functional;

  auto logits = torch::randn({5, 20, 50}, torch::kFloat32);
  auto log_softmax_out = F::log_softmax(logits, /*dim*/ 2);

  auto log_softmax_out_lens = torch::randint(20, {5}, torch::kInt32);

  auto ctc_topo = GetCtcTopo(49);

  auto lattice = GetLattice(log_softmax_out, log_softmax_out_lens, ctc_topo);
  auto results = BestPath(lattice);

  std::ostringstream oss;
  for (auto result : results) {
    for (auto id : result) {
      oss << id << " ";
    }
    oss << "\n";
  }
  std::cout << "Decoding results : " << oss.str();
}

// This test does not do any checking, just to confirm it runs normally.
TEST(RnntDecode, TestBasicRnntDecode) {
  namespace F = torch::nn::functional;
  int32_t vocab_size = 50, frames = 5, num_streams = 3, context_size = 2;
  auto graph = GetTrivialGraph(vocab_size - 1);
  std::vector<RnntStreamPtr> raw_streams(num_streams);
  for (int32_t i = 0; i < num_streams; ++i) {
    raw_streams[i] = CreateRnntStream(graph);
  }
  RnntStreamsPtr streams =
      CreateRnntStreams(raw_streams, vocab_size, context_size);

  for (int32_t i = 0; i < frames; ++i) {
    auto contexts = GetRnntContexts(streams);
    int32_t num_contexts = std::get<1>(contexts).size(0);
    auto logits = torch::randn({num_contexts, vocab_size}, torch::kFloat32);
    auto log_probs = F::log_softmax(logits, /*dim*/ 1);
    AdvanceRnntStreams(streams, log_probs);
  }
  TerminateAndFlushRnntStreams(streams);

  std::vector<int32_t> num_frames(num_streams, frames);

  auto lattice = FormatOutput(streams, num_frames, true /*allow_partial*/);
  auto results = BestPath(lattice);

  std::ostringstream oss;
  for (auto result : results) {
    for (auto id : result) {
      oss << id << " ";
    }
    oss << "\n";
  }
  std::cout << "Decoding results : " << oss.str();
}

}  // namespace k2
