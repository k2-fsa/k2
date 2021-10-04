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

#include "gtest/gtest.h"
#include "k2/torch/csrc/dense_fsa_vec.h"

namespace k2 {

TEST(CreateDenseFsaVec, AllowTruncate_0) {
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

    // utterance 2, 1 frames
    1,   5, 9, 10,  12,
    13,  6, 8,  7,   9,
    0,  -1, 3,  8,   7,
  };
  // clang-format on
  torch::Tensor log_probs = torch::from_blob(
      v.data(), {3, 3, 5}, torch::device(torch::kCPU).dtype(torch::kFloat32));

  // clang-format off
  std::vector<int32_t> sup = {
    // utterance 0
    0, 0, 3,
    // utterance 2
    2, 0, 1,
    // utterance 1
    1, 1, 2,
  };
  // clang-format on
  torch::Tensor supervision_segments = torch::from_blob(
      sup.data(), {3, 3}, torch::device(torch::kCPU).dtype(torch::kInt));

  DenseFsaVec dense_fsa_vec =
      CreateDenseFsaVec(log_probs, supervision_segments, 0);
  K2_LOG(INFO) << dense_fsa_vec;
}

}  // namespace k2
