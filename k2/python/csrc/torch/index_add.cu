/**
 * @brief index_add for k2.
 *
 * It has identical semantics as torch.Tensor.index_add_
 * except that it requires the dtype of the input index
 * to be torch.int32, whereas PyTorch expects the dtype to be
 * torch.int64. Furthermore, it ignores index[i] == -1.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

static void PybindIndexAdd(torch::Tensor index, torch::Tensor value,
                           torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);

  Array1<int32_t> indexes = FromTensor<int32_t>(index);
  Tensor src = FromTensor(value, TensorTag{});
  Tensor dest = FromTensor(*in_out, TensorTag{});
  IndexAdd(src, indexes, true, &dest);
}

}  // namespace k2

void PybindIndexAdd(py::module &m) {
  // note it supports only 1-D and 2-D tensors.
  m.def("index_add", &k2::PybindIndexAdd, py::arg("index"), py::arg("value"),
        py::arg("in_out"));
}
