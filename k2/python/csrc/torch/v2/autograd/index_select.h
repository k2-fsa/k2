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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_SELECT_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_SELECT_H_

#include <utility>

#include "k2/python/csrc/torch/v2/ops.h"

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class IndexSelectFunction
    : public torch::autograd::Function<IndexSelectFunction> {
 public:
  /** Returns a new tensor which indexes the input tensor along dimension 0
      using the entries in `index`.

      If the entry in `index` is -1, then the corresponding entry in the
      returned tensor is `default_value`.

      Caution:
        `index.dtype == torch.int32` and `index.ndim == 1`.

      @param src The input tensor. Either 1-D or 2-D with dtype torch.int32,
                 torch.int64, torch.float32 or torch.float64.
      @param index 1-D tensor of dtype torch.int32 containing the indexes.
                   If an entry is -1, the corresponding entry in the returned
                   value is `default_value`. The elements of `index` should be
                   in the range `[-1..src.shape[0]-1]`.
      @param default_value Used only when `src` is a 1-D tensor. It sets ans[i]
                           to default_value if index[i] is -1.

      @return Return a tensor with shape (index.numel(), *src.shape[1:]) and
              dtype the same as `src`, e.g. if `src.ndim == 1`, ans.shape would
              be (index.shape[0],); if `src.ndim == 2`, ans.shape would be
              (index.shape[0], src.shape[1]).
              Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
              or `ans[i,j] == src[index[i],j]` if `src.ndim == 2`, except for
              entries where `index[i] == -1` which will be `default_value`.
   */
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor src, torch::Tensor index,
                               float default_value) {
    ctx->save_for_backward({src, index});
    return IndexSelect(src, index, default_value);
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor src_tensor = saved[0];
    torch::Tensor indexes_tensor = saved[1];
    torch::Tensor grad_output_tensor = grad_outputs[0];
    torch::Tensor ans = torch::zeros_like(src_tensor);

    IndexAdd(indexes_tensor, grad_output_tensor, &ans);

    return {
        ans,              // src
        torch::Tensor(),  // index
        torch::Tensor(),  // default_value
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_SELECT_H_
