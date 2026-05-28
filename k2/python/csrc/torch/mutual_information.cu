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

#include "k2/csrc/device_guard.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/mutual_information.h"

void PybindMutualInformation(py::module &m) {
  m.def(
      "mutual_information_forward",
      [](torch::Tensor px, torch::Tensor py,
         torch::optional<torch::Tensor> boundary,
         torch::Tensor p) -> torch::Tensor {
        k2::DeviceGuard guard(k2::GetContext(px));
        auto orig_device = px.device();
        if (px.device().is_cpu()) {
          return k2::MutualInformationCpu(px, py, boundary, p);
        } else if (px.device().type() == torch::kMPS) {
#ifdef K2_WITH_MPS
          // Only float32 is supported natively; fall back to CPU for double.
          if (px.scalar_type() == torch::kFloat) {
            return k2::MutualInformationMps(px, py, boundary, p);
          }
#endif
          // CPU fallback for MPS double (or no-MPS build)
          auto px_cpu = px.cpu(), py_cpu = py.cpu(), p_cpu = p.cpu();
          torch::optional<torch::Tensor> boundary_cpu;
          if (boundary.has_value()) boundary_cpu = boundary.value().cpu();
          auto result = k2::MutualInformationCpu(px_cpu, py_cpu, boundary_cpu,
                                                 p_cpu);
          p.copy_(p_cpu.to(orig_device));
          return result.to(orig_device);
        } else {
#ifdef K2_WITH_CUDA
          return k2::MutualInformationCuda(px, py, boundary, p);
#else
          K2_LOG(FATAL) << "Failed to find native CUDA module, make sure "
                        << "that you compiled the code with K2_WITH_CUDA.";
          return torch::Tensor();
#endif
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"));

  m.def(
      "mutual_information_backward",
      [](torch::Tensor px, torch::Tensor py,
         torch::optional<torch::Tensor> boundary, torch::Tensor p,
         torch::Tensor ans_grad) -> std::vector<torch::Tensor> {
        k2::DeviceGuard guard(k2::GetContext(px));
        auto orig_device = px.device();
        if (px.device().is_cpu()) {
          return k2::MutualInformationBackwardCpu(px, py, boundary, p,
                                                   ans_grad);
        } else if (px.device().type() == torch::kMPS) {
#ifdef K2_WITH_MPS
          if (px.scalar_type() == torch::kFloat) {
            return k2::MutualInformationBackwardMps(px, py, boundary, p,
                                                    ans_grad, false);
          }
#endif
          // CPU fallback for MPS double (or no-MPS build)
          auto px_cpu = px.cpu(), py_cpu = py.cpu(), p_cpu = p.cpu();
          auto ans_grad_cpu = ans_grad.cpu();
          torch::optional<torch::Tensor> boundary_cpu;
          if (boundary.has_value()) boundary_cpu = boundary.value().cpu();
          auto grads = k2::MutualInformationBackwardCpu(px_cpu, py_cpu,
                                                        boundary_cpu, p_cpu,
                                                        ans_grad_cpu);
          for (auto &g : grads) g = g.to(orig_device);
          return grads;
        } else {
#ifdef K2_WITH_CUDA
          return k2::MutualInformationBackwardCuda(px, py, boundary, p,
                                                   ans_grad, true);
#else
          K2_LOG(FATAL) << "Failed to find native CUDA module, make sure "
                        << "that you compiled the code with K2_WITH_CUDA.";
          return std::vector<torch::Tensor>();
#endif
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"),
      py::arg("ans_grad"));
}
