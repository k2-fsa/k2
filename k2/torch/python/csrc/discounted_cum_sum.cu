/**
 * @brief wraps discounted_cum_sum code.
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Daniel Povey)
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
#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/discounted_cum_sum.h"

namespace k2 {

static void DiscountedCumSumWrapper(torch::Tensor x, torch::Tensor gamma,
                                    torch::Tensor y, bool flip = false) {
  NVTX_RANGE(K2_FUNC);
  DeviceGuard guard(GetContext(x));
  Tensor x_k2 = FromTorch(x, TensorTag{});
  Tensor gamma_k2 = FromTorch(gamma, TensorTag{});
  Tensor y_k2 = FromTorch(y, TensorTag{});
  if (flip) {
    // We have to do this in C++ because Torch tensors don't support negative
    // strides.
    x_k2 = Flip(x_k2, 1);
    gamma_k2 = Flip(gamma_k2, 1);
    y_k2 = Flip(y_k2, 1);
  }
  DiscountedCumSum(x_k2, gamma_k2, &y_k2);
}

void PybindDiscountedCumSum(py::module &m) {
  // note it supports only 1-D and 2-D tensors.
  m.def("discounted_cum_sum", &DiscountedCumSumWrapper, py::arg("x"),
        py::arg("gamma"), py::arg("y"), py::arg("flip") = false,
        R"(
        Args:
          x:
            A 2-D tensor with dtype `torch.float` or `torch.double` and x.stride(1) == 1.
          gamma:
            A tensor with the same shape and dtype as x, and gamma.stride(1) == 1
          y:
            A tensor with the same shape and dtype as x, and y.stride(1) == 1.
            This function outputs to here.  It is allowed to be the same tensor
            as x and/or gamma.
            The shapes are interpreted as (N, T) with N as the batch size and T
            a sequence or time dimensions.  It implements:
                y(n, 0) = x(n, 0)
                y(n, t) = x(n, t) + y(n, t-1) * gamma(n, t)   (for 0<t<T)
         flip:
           If true, the time sequence is reversed..
        )");
}

}  // namespace k2
