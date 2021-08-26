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
#include "torch/extension.h"

using namespace torch::autograd;

namespace k2 {

template <typename T>
class SumFunction : public torch::autograd::Function<SumFunction<T>> {
  static_assert(std::is_floating_point<T>::value);

 public:
  /* Compute the argmax of a RaggedAny. It is a wrapper around
     "SumPerSublist" and supports autograd.

     @param RaggedAny The input RaggedAny
     @param dummy  Its purpose is to make autograd to track the operation
                   of the input `any§. It is the same as any.data_
   */
  static torch::Tensor forward(AutogradContext *ctx, RaggedAny &any,
                               torch::Tensor /*dummy*/, float initial_value) {
    ctx->saved_data["n"] = any.any_.values.Dim();

    Array1<T> values(any.any_.Context(),
                     any.any_.TotSize(any.any_.NumAxes() - 2));

    SumPerSublist<T>(any.any_.Specialize<T>(), initial_value, &values);
    return ToTorch(values);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // see https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
    // https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html
    auto n = ctx->saved_data["n"].toInt();

    auto opts = torch::TensorOptions()
                    .dtype(grad_outputs[0].dtype())
                    .device(grad_outputs[0].device());

    // The gradient is not correct, for each sublist, we should multiply it with
    // the corresponding grad_outputs[0]
    torch::Tensor ans = torch::ones({n}, opts);

    return {
        torch::Tensor(),  // any
        ans,              // dummy
        torch::Tensor()   // initial value
    };
  }
};  // namespace k2

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_AUTOGRAD_SUM_H
