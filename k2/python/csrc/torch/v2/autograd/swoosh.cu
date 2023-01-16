/**
 * @copyright
 * Copyright      2023  Xiaomi Corp.  (authors: Zengwei Yao, Fangjun Kuang)
 *
 * @copyright
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

#include "k2/python/csrc/torch/v2/autograd/swoosh.h"

namespace k2 {

struct SwooshLConstants {
  static constexpr float kShift = 4;
  static constexpr float kCoeff = 0.08;
  static constexpr float kOffset = 0.035;
};

struct SwooshRConstants {
  static constexpr float kShift = 1;
  static constexpr float kCoeff = 0.08;
  static constexpr float kOffset = 0.313261687;
};

template class SwooshFunction<SwooshLConstants>;
template class SwooshFunction<SwooshRConstants>;

using SwooshLFunction = SwooshFunction<SwooshLConstants>;
using SwooshRFunction = SwooshFunction<SwooshRConstants>;

void PybindSwoosh(py::module &m) {
  m.def(
      "swoosh_l",
      [](torch::Tensor x, float dropout_prob) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshLFunction::apply(x, dropout_prob);
      },
      py::arg("x"), py::arg("dropout_prob"));
  // please add swoosh_r
}

}  // namespace k2
