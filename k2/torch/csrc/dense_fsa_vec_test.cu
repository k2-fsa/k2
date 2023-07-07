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

#include <vector>

#ifdef K2_WITH_CUDA
#include "c10/cuda/CUDAFunctions.h"
#endif
#include "gtest/gtest.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

TEST(CreateDenseFsaVec, AllowTruncate_0) {
  std::vector<torch::DeviceType> device_types = {torch::kCPU};
#ifdef K2_WITH_CUDA
  if (torch::cuda::device_count()) {
    device_types.push_back(torch::kCUDA);
  }
#endif
  // clang-format off
  std::vector<float> v = {
    // utterance 0, 3 frames
    0,   1, 2, 3.5, 5,
    0.5, 2, 3, 10, -1,
    3,   6, 9, 1,   8,

    // utterance 1, 2 frames
    1,   3, 5, -2,  0,
    0,   2, 1,  3,  10,
    0,   1, 3,  8,  7,

    // utterance 2, 1 frame
    1,   5, 9, 10,  12,
    13,  6, 8,  7,   9,
    0,  -1, 3,  8,   7,
  };

  std::vector<int32_t> sup = {
    // utterance 0
    0, 0, 3,
    // utterance 2
    2, 0, 1,
    // utterance 1
    1, 1, 2,
  };
  // clang-format on
  for (auto device_type : device_types) {
    torch::Device device(device_type, 0);
    torch::Tensor log_probs =
        torch::from_blob(v.data(), {3, 3, 5},
                         torch::device(torch::kCPU).dtype(torch::kFloat32))
            .to(device);

    torch::Tensor supervision_segments = torch::from_blob(
        sup.data(), {3, 3}, torch::device(torch::kCPU).dtype(torch::kInt));

    DenseFsaVec dense_fsa_vec =
        CreateDenseFsaVec(log_probs, supervision_segments, 0);

    ContextPtr ctx = ContextFromDevice(device);
    RaggedShape expected_shape =
        RaggedShape("[[x x x x] [x x] [x x x]]").To(ctx);
    EXPECT_TRUE(Equal(dense_fsa_vec.shape, expected_shape));

    Array2<float> expected_scores(R"(
      [[-inf 0 1 2 3.5 5]
       [-inf 0.5 2 3 10 -1]
       [-inf 3 6 9 1 8]
       [0 -inf -inf -inf -inf -inf]

       [-inf 1 5 9 10 12]
       [0 -inf -inf -inf -inf -inf]

       [-inf 0 2 1 3 10]
       [-inf 0 1 3 8 7]
       [0 -inf -inf -inf -inf -inf]
       ])");
    expected_scores = expected_scores.To(ctx);

    EXPECT_TRUE(Equal(expected_scores, dense_fsa_vec.scores));
  }
}

TEST(CreateDenseFsaVec, AllowTruncate_1) {
  std::vector<torch::DeviceType> device_types = {torch::kCPU};
#ifdef K2_WITH_CUDA
  if (torch::cuda::device_count()) {
    device_types.push_back(torch::kCUDA);
  }
#endif
  // clang-format off
  std::vector<float> v = {
    // utterance 0, 3 frames
    -1, 2, 3, 4,
    8,  9, 6, 5.5,
    2,  3, 4, 5,
    // utterance 1, 1 frame
    -2, -1, 3, 4,
    2,   3, 0, 8,
    8,   9, 0, 9.8,
  };

  std::vector<int32_t> sup = {
    // utterance 1
    1, 2, 2,
    // utterance 0
    0, 0, 5,
  };

  // clang-format on
  for (auto device_type : device_types) {
    torch::Device device(device_type, 0);
    torch::Tensor log_probs =
        torch::from_blob(v.data(), {2, 3, 4},
                         torch::device(torch::kCPU).dtype(torch::kFloat32))
            .to(device);

    torch::Tensor supervision_segments = torch::from_blob(
        sup.data(), {2, 3}, torch::device(torch::kCPU).dtype(torch::kInt));

    DenseFsaVec dense_fsa_vec =
        CreateDenseFsaVec(log_probs, supervision_segments, 2);

    ContextPtr ctx = ContextFromDevice(device);
    RaggedShape expected_shape = RaggedShape("[[x x] [x x x x]]").To(ctx);
    EXPECT_TRUE(Equal(dense_fsa_vec.shape, expected_shape));

    Array2<float> expected_scores(R"(
      [[-inf 8 9 0 9.8]
       [0 -inf -inf -inf -inf]

       [-inf -1 2 3 4]
       [-inf 8 9 6 5.5]
       [-inf 2 3 4 5]
       [0 -inf -inf -inf -inf]
       ])");

    expected_scores = expected_scores.To(ctx);

    EXPECT_TRUE(Equal(expected_scores, dense_fsa_vec.scores));
  }
}

}  // namespace k2
