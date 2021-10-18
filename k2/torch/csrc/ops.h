/**
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                2021  Xiaomi Corp.       (author: Daniel Povey,
 *                                                  Haowen Qiu,
 *                                                  Wei Kang)
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

#ifndef K2_TORCH_CSRC_OPS_H_
#define K2_TORCH_CSRC_OPS_H_

#include <utility>

#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

/* Add the `value` to `in_out` on the positons in `index`, the `in_out` will be
modified by `in_out[indx] += value`.

Note:
  It has identical semantics as torch.Tensor.index_add_ except that it requires
  the dtype of the input index to be torch.int32, whereas PyTorch expects the
  dtype to be torch.int64. Furthermore, it ignores index[i] == -1.

Caution:
  It supports only 1-D and 2-D tensors.

Caution:
  `in_out` is modified **in-place**.

  @param [in] index A 1-D **contiguous** tensor with dtype `torch.int32`.
                    Must satisfy `-1 <= index[i] < in_out.shape[0]` and
                    `index.shape[0] == value.shape[0]`.
  @param [in] value A 1-D or a 2-D tensor. Supported dtypes are: `torch.int32`,
                    `torch.float32`, and `torch.float64`.
  @param [in,out] in_out Its `ndim` equals to `value.ndim`. If it is a
                         2-D tensor, then `in_out.shape[1] == value.shape[1]`.
                         Must satisfy `in_out.dtype == value.dtype`.
                         Will be modified in place, the modified tensor
                         satisfies `in_out[index[i]] += value[i]`
                         if `index[i] != -1`
  @return
    None.
 */
void IndexAdd(torch::Tensor index, torch::Tensor value, torch::Tensor *in_out);

/* Index a tensor with a 1-D tensor of indexes, along dimension 0.

Caution:
  The index MUST have the dtype equal to `torch.int32` and
  dimension equals to 1.

  @param [in] src The input tensor. Either 1-D or 2-D with dtype `torch.int32`,
                  `torch.int64`, `torch.float32`, or `torch.float64`.
  @param [in] index 1-D tensor of dtype `torch.int32` containing the indexes.
                    If an entry is -1, the corresponding entry in the returned
                    value is `default_value`. The elements of `index` should be
                    in the range `[-1..src.shape[0]-1]`.
  @param [in] default_value Used only when `src` is a 1-D tensor.
                            It sets ans[i] to default_value if index[i] is -1.

  @return
    Return A tensor with shape ``(index.numel(), *src.shape[1:])`` and dtype
    the same as `src`, e.g. if `src.ndim == 1`, `ans.shape` would be
    `(index.shape[0],)`; if `src.ndim == 2`, `ans.shape` would be
    `(index.shape[0], src.shape[1])`.
    Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
    or `ans[i, j] == src[index[i], j]` if `src.ndim == 2`, except for
    entries where `index[i] == -1` which will be `default_value`.
 */
torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index,
                          double default_value = 0);

/* TODO: Add docs here.
 */
torch::Tensor SimpleRaggedIndexSelect(torch::Tensor src, RaggedAny &ragged);

}  // namespace k2

#endif  // K2_TORCH_CSRC_OPS_H_
