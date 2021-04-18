/**
 * @brief wraps discounted_cum_sum code.
 *
 * @copyright
 * Copyright (c)  2010  Xiaomi Corp.  (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void DiscountedCumSumWrapper(torch::Tensor x, torch::Tensor gamma,
                                    torch::Tensor y, bool flip = false) {
  NVTX_RANGE(K2_FUNC);
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

}  // namespace k2

void PybindDiscountedCumSum(py::module &m) {
  // note it supports only 1-D and 2-D tensors.
  m.def("discounted_cum_sum", &k2::DiscountedCumSumWrapper, py::arg("x"), py::arg("gamma"),
        py::arg("y"), py::arg("flip") = false,
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
