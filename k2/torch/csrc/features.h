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

#ifndef K2_TORCH_CSRC_FEATURES_H_
#define K2_TORCH_CSRC_FEATURES_H_

#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"

namespace k2 {

/** Compute fbank features of a 1-D tensor containing audio samples.

  @param fbank  The Fbank computer.
  @param wave_data  A 1-D tensor with dtype torch.float32. Its elements
               are expected in the range [-1, 1).
  @return Return a 2-D tensor containing the features. Number of
    rows equals to the number of frames.
 */
torch::Tensor ComputeFeatures(kaldifeat::Fbank &fbank, torch::Tensor wave_data);

/// See `ComputeFeatures` above. It computes fbank features for a list
/// of audio samples, in parallel.
///
/// @params num_frames If not null, it contains the number of feature frames of
///                    each wave.
std::vector<torch::Tensor> ComputeFeatures(
    kaldifeat::Fbank &fbank, const std::vector<torch::Tensor> &wave_data,
    std::vector<int64_t> *num_frames = nullptr);

}  // namespace k2

#endif  // K2_TORCH_CSRC_FEATURES_H_
