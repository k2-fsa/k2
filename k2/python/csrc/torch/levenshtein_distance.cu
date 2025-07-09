/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/levenshtein_distance.h"

void PybindLevenshteinDistance(py::module &m) {
  m.def(
      "levenshtein_distance",
      [](torch::Tensor px, torch::Tensor py,
         torch::optional<torch::Tensor> boundary) -> torch::Tensor {
        k2::DeviceGuard guard(k2::GetContext(px));
        if (px.device().is_cpu()) {
          return k2::LevenshteinDistanceCpu(px, py, boundary);
        } else {
#ifdef K2_WITH_CUDA
          return k2::LevenshteinDistanceCuda(px, py, boundary);
#else
          K2_LOG(FATAL) << "Failed to find native CUDA module, make sure "
                        << "that you compiled the code with K2_WITH_CUDA.";
          return torch::Tensor();
#endif
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"));
}
