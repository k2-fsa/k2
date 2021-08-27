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

#ifndef K2_PYTHON_CSRC_TORCH_AUTOGRAD_SUM_H
#define K2_PYTHON_CSRC_TORCH_AUTOGRAD_SUM_H

#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/ragged_any.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

using namespace torch::autograd;

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class SumFunction : public torch::autograd::Function<SumFunction> {
 public:
  /* Compute the Sum of a RaggedAny. It is a wrapper around
     "SumPerSublist" and supports autograd.

     The sum is done over the last axis.

     @param ragged The input RaggedAny
     @param dummy  Its purpose is to make autograd to track the operations on
                   the input `ragged`. It is the same as `ragged.data_`.

     @return Return a 1-D tensor containing the sum of each sublist.
   */
  static torch::Tensor forward(AutogradContext *ctx, const RaggedAny &ragged,
                               torch::Tensor /*dummy*/, float initial_value) {
    ctx->saved_data["n"] = ragged.any_.values.Dim();

    int32_t num_axes = ragged.any_.NumAxes();

    torch::Tensor row_ids =
        ToTorch(const_cast<RaggedAny &>(ragged).any_.RowIds(num_axes - 1));

    ctx->save_for_backward({row_ids});

    Dtype t = ragged.any_.GetDtype();

    FOR_REAL_AND_INT32_TYPES(t, T, {
      Array1<T> values(ragged.any_.Context(),
                       ragged.any_.TotSize(num_axes - 2));
      SumPerSublist<T>(ragged.any_.Specialize<T>(), initial_value, &values);
      return ToTorch(values);
    });

    // Unreachable code
    return {};
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto n = ctx->saved_data["n"].toInt();
    auto saved = ctx->get_saved_variables();
    torch::Tensor row_ids = saved[0];

    auto grad_output = grad_outputs[0];

    auto opts = torch::TensorOptions()
                    .dtype(grad_output.dtype())
                    .device(grad_output.device());

    // The gradient is not correct, for each sublist, we should multiply it with
    // the corresponding grad_outputs[0]
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
            ans_data[idx01] = grad_output_data[idx0];
          });
    });

    return {
        torch::Tensor(),  // any
        ans,              // dummy
        torch::Tensor()   // initial value
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_AUTOGRAD_SUM_H
