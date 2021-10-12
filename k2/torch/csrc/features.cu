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

#include <utility>

#include "k2/torch/csrc/features.h"
#include "kaldifeat/csrc/feature-fbank.h"

namespace k2 {

torch::Tensor ComputeFeatures(kaldifeat::Fbank &fbank,
                              torch::Tensor wave_data) {
  return fbank.ComputeFeatures(wave_data, /*vtln_warp*/ 1.0f);
}

std::vector<torch::Tensor> ComputeFeatures(
    kaldifeat::Fbank &fbank, const std::vector<torch::Tensor> &wave_data,
    std::vector<int64_t> *num_frames /*=nullptr*/) {
  const auto &frame_opts = fbank.GetOptions().frame_opts;

  std::vector<int64_t> num_frames_vec;
  num_frames_vec.reserve(wave_data.size());

  std::vector<torch::Tensor> strided_vec;
  strided_vec.reserve(wave_data.size());

  for (const auto &t : wave_data) {
    torch::Tensor strided = kaldifeat::GetStrided(t, frame_opts);
    num_frames_vec.push_back(strided.size(0));
    strided_vec.emplace_back(std::move(strided));
  }

  torch::Tensor strided = torch::cat(strided_vec, 0);
  torch::Tensor features = fbank.ComputeFeatures(strided, /*vtln_warp*/ 1.0f);

  auto ans = features.split_with_sizes(num_frames_vec, /*dim*/ 0);
  if (num_frames) *num_frames = std::move(num_frames_vec);
  return ans;
}

}  // namespace k2
