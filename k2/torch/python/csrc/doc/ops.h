/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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

#ifndef K2_TORCH_PYTHON_CSRC_DOC_OPS_H_
#define K2_TORCH_PYTHON_CSRC_DOC_OPS_H_

namespace k2 {

static constexpr const char *kTensorIndexSelectDoc = R"doc(
Index a tensor with a 1-D tensor of indexes, along dimension 0.

Note:
  Autograd is supported.

Caution:
  The index MUST have the dtype equal to `torch.int32` and
  dimension equals to 1.

>>> import torch
>>> import k2
>>> src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
>>> index = torch.tensor([2, 4, -1, 6, 1], dtype=torch.int32)
>>> k2.index_select(src, index)
tensor([2, 4, 0, 6, 1])

>>> k2.index_select(src, index, 100)
tensor([  2,   4, 100,   6,   1])
>>> src = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]])
>>> k2.index_select(src, index)
tensor([[0, 3],
        [0, 5],
        [0, 0],
        [0, 7],
        [0, 2]])
>>> src = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
>>> index = torch.tensor([2, 4, -1, 1], dtype=torch.int32)
>>> r = k2.index_select(src, index)
>>> print(r)
tensor([2., 4., 0., 1.], grad_fn=<IndexSelectFunction>>)
>>> r.sum().backward()
>>> print(src.grad)
tensor([0., 1., 1., 0., 1., 0.])
>>> src = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
>>> index = torch.tensor([2, 4, -1, 1, 2, 4, 2, 2], dtype=torch.int32)
>>> r = k2.index_select(src, index)
>>> print(r)
tensor([2., 4., 0., 1., 2., 4., 2., 2.], grad_fn=<IndexSelectFunction>>)
>>> r.sum().backward()
>>> print(src.grad)
tensor([0., 1., 4., 0., 2., 0.])

Args:
  src:
    The input tensor. Either 1-D or 2-D with dtype `torch.int32`,
    `torch.int64`, `torch.float32`, or `torch.float64`.
  index:
    1-D tensor of dtype `torch.int32` containing the indexes.
    If an entry is -1, the corresponding entry in the returned value
    is `default_value`. The elements of `index` should be in the range
    `[-1..src.shape[0]-1]`.
  default_value:
    Used only when `src` is a 1-D tensor. It sets ans[i] to default_value
    if index[i] is -1.

Returns:
  A tensor with shape ``(index.numel(), *src.shape[1:])`` and dtype the
  same as `src`, e.g. if `src.ndim == 1`, `ans.shape` would be
  `(index.shape[0],)`; if `src.ndim == 2`, `ans.shape` would be
  `(index.shape[0], src.shape[1])`.
  Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
  or `ans[i, j] == src[index[i], j]` if `src.ndim == 2`, except for
  entries where `index[i] == -1` which will be `default_value`.
)doc";

static constexpr const char *kSimpleRaggedIndexSelectDoc = R"doc(
TODO: Add more docs here.
>> import torch
>> import k2
>>> src = torch.tensor([0, 2, 0, 10, 0, -1], dtype=torch.int32)
>>> index = k2.RaggedTensor([[1, 0, 4], [2, 3], [0], [], [4, 5, 2]], dtype=torch.int32)
>>> k2.simple_ragged_index_select(src, index)
tensor([ 2, 10,  0,  0, -1], dtype=torch.int32)

)doc";

static constexpr const char *kTensorIndexAddDoc = R"doc(
Add the `value` to `in_out` on the positons in `index`, the `in_out` will be
modified by `in_out[indx] += value`.

Note:
  It has identical semantics as torch.Tensor.index_add_ except that it requires
  the dtype of the input index to be torch.int32, whereas PyTorch expects the
  dtype to be torch.int64. Furthermore, it ignores index[i] == -1.

Caution:
  It supports only 1-D and 2-D tensors.

Caution:
  `in_out` is modified **in-place**.

Caution:
  This operation does not support autograd.

>>> import torch
>>> import k2
>>> index = torch.tensor([0, 1, 2, -1, 4, -1], dtype=torch.int32)
>>> value = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
>>> in_out = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
>>> k2.index_add(index=index, value=value, in_out=in_out)
>>> print(in_out)
tensor([10, 12, 14, 13, 18, 15], dtype=torch.int32)
>>> index = torch.tensor([0, 1, 1, -1, 4, 4], dtype=torch.int32)
>>> value = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
>>> in_out = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
>>> k2.index_add(index=index, value=value, in_out=in_out)
>>> print(in_out)
tensor([10, 14, 12, 13, 23, 15], dtype=torch.int32)
>>> index = torch.tensor([0, 1, 2, -1, 4, -1], dtype=torch.int32)
>>> value = torch.tensor([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.int32)
>>> in_out = torch.tensor([[0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15]], dtype=torch.int32)
>>> k2.index_add(index=index, value=value, in_out=in_out)
>>> print(in_out)
tensor([[ 1, 10],
        [ 1, 12],
        [ 1, 14],
        [ 0, 13],
        [ 1, 18],
        [ 0, 15]], dtype=torch.int32)

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
    Will be modified in place, the modified tensor satisfies
    `in_out[index[i]] += value[i]` if `index[i] != -1`

Returns:
  None.
)doc";

}  // namespace k2

#endif  // K2_TORCH_PYTHON_CSRC_DOC_OPS_H_
