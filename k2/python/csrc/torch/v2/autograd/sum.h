/**
 * @brief A wrapper around SumPerSublist to support autograd.
 *
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SUM_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SUM_H_

#include <utility>

#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class SumFunction : public torch::autograd::Function<SumFunction> {
 public:
  /* Compute the Sum of a RaggedAny. It is a wrapper around
     "SumPerSublist" and supports autograd.

     The sum is done over the last axis.

     @param ragged The input RaggedAny
     @param dummy  Its purpose is to make autograd to track the operations on
                   the input `ragged`. It is the same as `ragged.data`.
     @param initial_value This value is added to the sum of each sublist,
                          so when a sublist is empty, its sum is this value.

     @return Return a 1-D tensor containing the sum of each sublist, with
             the same dtype as the input ragged tensor.
   */
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               const RaggedAny &ragged, torch::Tensor /*dummy*/,
                               float initial_value) {
    ctx->saved_data["n"] = ragged.any.values.Dim();

    int32_t num_axes = ragged.any.NumAxes();

    torch::Tensor row_ids =
        ToTorch(const_cast<RaggedAny &>(ragged).any.RowIds(num_axes - 1));

    ctx->save_for_backward({row_ids});

    Dtype t = ragged.any.GetDtype();

    FOR_REAL_AND_INT32_TYPES(t, T, {
      Array1<T> values(ragged.any.Context(), ragged.any.TotSize(num_axes - 2));
      SumPerSublist<T>(ragged.any.Specialize<T>(), initial_value, &values);
      return ToTorch(values);
    });

    // Unreachable code
    return {};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto n = ctx->saved_data["n"].toInt();
    auto saved = ctx->get_saved_variables();
    torch::Tensor row_ids = saved[0];

    auto grad_output = grad_outputs[0];
    // note: stride may be zero
    // see https://github.com/k2-fsa/k2/pull/1101
    int32_t stride = grad_output.stride(0);

    auto opts = torch::TensorOptions()
                    .dtype(grad_output.dtype())
                    .device(grad_output.device());

    torch::Tensor ans = torch::empty({n}, opts);
    ContextPtr c = GetContext(row_ids);

    const int32_t *row_ids_data = row_ids.data_ptr<int32_t>();
    Dtype t = ScalarTypeToDtype(grad_output.scalar_type());

    FOR_REAL_AND_INT32_TYPES(t, T, {
      const T *grad_output_data = grad_output.data_ptr<T>();
      T *ans_data = ans.data_ptr<T>();

      K2_EVAL(
          c, n, set_grad, (int32_t idx01)->void {
            int32_t idx0 = row_ids_data[idx01];
            ans_data[idx01] = grad_output_data[idx0 * stride];
          });
    });

    return {
        torch::Tensor(),  // ragged
        ans,              // dummy
        torch::Tensor()   // initial value
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SUM_H_
