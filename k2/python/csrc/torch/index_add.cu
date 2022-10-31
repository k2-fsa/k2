/**
 * @brief index_add for k2.
 *
 * It has identical semantics as torch.Tensor.index_add_
 * except that it requires the dtype of the input index
 * to be torch.int32, whereas PyTorch expects the dtype to be
 * torch.int64. Furthermore, it ignores index[i] == -1.
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/index_add.h"

namespace k2 {

static void PybindIndexAdd(torch::Tensor index, torch::Tensor value,
                           torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);
  DeviceGuard guard(GetContext(index));

  Array1<int32_t> indexes = FromTorch<int32_t>(index);
  Tensor src = FromTorch(value, TensorTag{});
  Tensor dest = FromTorch(*in_out, TensorTag{});
  IndexAdd(src, indexes, true, &dest);
}

}  // namespace k2

void PybindIndexAdd(py::module &m) {
  // note it supports only 1-D and 2-D tensors.
  m.def("index_add", &k2::PybindIndexAdd, py::arg("index"), py::arg("value"),
        py::arg("in_out"),
        R"(
        Args:
          index:
            A 1-D **contiguous** tensor with dtype `torch.int32`.
            Must satisfy `-1 <= index[i] < in_out.shape[0]` and
            `index.shape[0] == value.shape[0]`.
          value:
            A 1-D or a 2-D tensor. Supported dtypes are: `torch.int32`,
            `torch.float32`, and `torch.float64`.
          in_out:
            Its `ndim` equals to `value.ndim`. If it is a 2-D tensor, then
            `in_out.shape[1] == value.shape[1]`.
            Must satisfy `in_out.dtype == value.dtype`.

            On return: `in_out[index[i]] += value[i]` if `index[i] != -1`
        )");
}
