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

#ifndef K2_TORCH_CSRC_WAVE_READER_H_
#define K2_TORCH_CSRC_WAVE_READER_H_

#include <string>

#include "torch/script.h"

namespace k2 {

// It supports only mono, i.e., single channel, wave files, encoded
// in PCM format, i.e., raw format, without compression.
// Each sound sample shall be two bytes.
//
// If the above constraints are not satisfied, it throws an exception
// and shows you which constraint was violated.
class WaveReader {
 public:
  /** Construct a wave reader from a wave filename, encoded in PCM format.

      @param filename  Path to a wave file. Must be mono and PCM encoded.
                       Note: Samples are divided by 32768 so that they are
                       in the range [-1, 1)
   */
  explicit WaveReader(const std::string &filename);

  /// Return a 1-D tensor with dtype torch.float32
  const torch::Tensor &Data() const { return data_; }

  float SampleRate() const { return sample_rate_; }

 private:
  /// A 1-D tensor with dtype torch.float32
  torch::Tensor data_;

  float sample_rate_;
};

}  // namespace k2

#endif  // K2_TORCH_CSRC_WAVE_READER_H_
