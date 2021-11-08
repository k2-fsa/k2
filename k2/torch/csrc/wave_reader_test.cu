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
#include "k2/csrc/log.h"
#include "k2/torch/csrc/test_wave_data.h"
#include "k2/torch/csrc/wave_reader.h"

namespace k2 {

TEST(WaveReader, Mono) {
  std::stringstream ss;
  ss.write(reinterpret_cast<const char *>(kTestWav), sizeof(kTestWav));
  WaveReader reader(ss);
  torch::Tensor expected = torch::arange(16, torch::kShort);
  expected.data_ptr<int16_t>()[0] = 32767;
  expected = (expected / 32768.).to(torch::kFloat32);
  EXPECT_TRUE(reader.Data().allclose(expected, 1e-6));
  EXPECT_EQ(reader.SampleRate(), 16000);
}

}  // namespace k2
