/**
 * @copyright
 * Copyright      2021  Xiaomi Corporation (authors: Wei Kang)
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

#include "k2/python/csrc/torch/mutual_information.h"

void PybindMutualInformation(py::module &m) {
  m.def(
      "mutual_information_forward",
      [](torch::Tensor px, torch::Tensor py, torch::Tensor boundary,
         torch::Tensor p) -> torch::Tensor {
        if (px.device().is_cpu()) {
          return k2::MutualInformationCpu(px, py, boundary, p);
        } else {
          // TODO: raise if not compiled with cuda
          return k2::MutualInformationCuda(px, py, boundary, p);
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"));

  m.def(
      "mutual_information_backward",
      [](torch::Tensor px, torch::Tensor py, torch::Tensor boundary,
         torch::Tensor p,
         torch::Tensor ans_grad) -> std::vector<torch::Tensor> {
        if (px.device().is_cpu()) {
          return k2::MutualInformationBackwardCpu(px, py, boundary, p,
                                                  ans_grad);
        } else {
          // TODO: raise if not compiled with cuda
          return k2::MutualInformationBackwardCuda(px, py, boundary, p,
                                                   ans_grad, true);
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"),
      py::arg("ans_grad"));
}
