/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang)
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_AND_SUM_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_AND_SUM_H_

#include <utility>

#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class IndexAndSumFunction
    : public torch::autograd::Function<IndexAndSumFunction> {
 public:
  /** Index a 1-D tensor with a ragged tensor of indexes, perform
      a sum-per-sublist operation, and return the resulting 1-D tensor.

      The 1-D tensor can be a float tensor attribute of an FSA.

      The ragged tensor has two axes. For example, it can be
      the arc map from :func:`k2.Fsa.remove_epsilon`.

     @param src An 1-D tensor to be indexed.
     @param ragged The indexes. It MUST have two axes with dtype torch.int32.

     @return Return a 1-D tensor containing the sum of each indexed sublist,
     with the same dtype as the input ragged tensor.
   */
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor src, RaggedAny &ragged) {
    K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();

    int32_t num_axes = ragged.any.NumAxes();
    K2_CHECK_EQ(num_axes, 2);

    Dtype t = ragged.any.GetDtype();
    K2_CHECK_EQ(t, kInt32Dtype) << "Unsupported dtype: " << TraitsOf(t).Name();

    DeviceGuard guard(ragged.any.Context());

    torch::Tensor row_ids = ToTorch(ragged.any.RowIds(num_axes - 1));
    ctx->save_for_backward({src, row_ids, ragged.Data()});

    Dtype dtype = ScalarTypeToDtype(src.scalar_type());
    FOR_REAL_AND_INT32_TYPES(dtype, T, {
      Array1<T> src_array = FromTorch<T>(src);
      Ragged<T> s = k2::Index<T>(src_array, ragged.any.Specialize<int32_t>(),
                                 /*default_value*/ 0);

      Array1<T> ans_array(s.Context(), s.Dim0());
      SumPerSublist<T>(s, 0, &ans_array);
      return ToTorch(ans_array);
    });

    // Unreachable code
    return {};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor src_tensor = saved[0];
    torch::Tensor row_ids_tensor = saved[1];
    torch::Tensor indexes_tensor = saved[2];

    torch::Tensor grad_output_tensor = grad_outputs[0];

    torch::Tensor ans = torch::zeros_like(src_tensor);

    Array1<int32_t> row_ids = FromTorch<int32_t>(row_ids_tensor);
    Array1<int32_t> indexes = FromTorch<int32_t>(indexes_tensor);

    Tensor grad_output = FromTorch(grad_output_tensor, TensorTag{});

    Tensor expanded_grad_output =
        Index(grad_output, row_ids, /*allow_minus_one*/ true,
              /*default_value*/ 0);

    Tensor dest = FromTorch(ans, TensorTag{});
    IndexAdd(expanded_grad_output, indexes, /*allow_minus_one*/ true, &dest);

    return {
        ans,              // src
        torch::Tensor(),  // ragged
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_INDEX_AND_SUM_H_
