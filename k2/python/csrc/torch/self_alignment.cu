/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Liyong Guo)
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

#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/torch_util.h"
#include "k2/csrc/self_alignment.h"
#include "k2/python/csrc/torch/self_alignment.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"


void PybindSelfAlignment(py::module &m) {
  m.def(
      "self_alignment",
      [](torch::Tensor ranges, torch::Tensor x_lens,
         torch::Tensor y,
         torch::Tensor logits) -> std::pair<k2::FsaVec, torch::Tensor> {
        k2::DeviceGuard guard(k2::GetContext(ranges));
        k2::Array1<int32_t> label_map;
        k2::FsaVec ofsa = k2::SelfAlignment(ranges, x_lens, y, logits, &label_map);
        torch::Tensor tensor = ToTorch(label_map);
        return std::make_pair(ofsa, tensor);
      },
      py::arg("ranges"), py::arg("x_lens"), py::arg("y"), py::arg("logits"));
}
