/**
 * @brief A wrapper around NormalizePerSublist to support autograd.
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_NORMALIZE_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_NORMALIZE_H_

#include <utility>

#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class NormalizeFunction : public torch::autograd::Function<NormalizeFunction> {
 public:
  /** Normalize each sublist. It is a wrapper around
    k2::NormalizePerSublist() to support autograd.

     Note: Backward propagation is implemented only for `use_log == true`.

     The normalization is done over the last axis.

     @param ragged The input RaggedAny
     @param use_log  True to normalize in log space.
     @param dummy  Its purpose is to make autograd to track the operations on
                   the input `ragged`. It is the same as `ragged.data`.
     @param out  The output of this function.

     @return Return a 1-D tensor containing the normalization of each sublist,
             which is `out->Data()`
   */
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               RaggedAny &ragged, bool use_log,
                               torch::Tensor /*dummy*/, RaggedAny *out) {
    int32_t num_axes = ragged.any.NumAxes();

    torch::Tensor row_splits = ToTorch(ragged.any.RowSplits(num_axes - 1));
    torch::Tensor row_ids = ToTorch(ragged.any.RowIds(num_axes - 1));

    ctx->saved_data["use_log"] = use_log;
    ctx->saved_data["num_sublists"] = ragged.any.TotSize(num_axes - 2);

    Dtype t = ragged.any.GetDtype();
    FOR_REAL_TYPES(t, T, {
      out->any =
          NormalizePerSublist(ragged.any.Specialize<T>(), use_log).Generic();
    });

    ctx->save_for_backward({row_splits, row_ids, out->Data().detach()});

    return out->Data();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    bool use_log = ctx->saved_data["use_log"].toBool();
    K2_CHECK_EQ(use_log, true)
        << "backprop is implemented only for use_log is true";

    int32_t num_sublists = ctx->saved_data["num_sublists"].toInt();

    auto saved = ctx->get_saved_variables();
    torch::Tensor row_splits_tensor = saved[0];
    torch::Tensor row_ids_tensor = saved[1];
    Array1<int32_t> row_splits = FromTorch<int32_t>(row_splits_tensor);
    Array1<int32_t> row_ids = FromTorch<int32_t>(row_ids_tensor);

    torch::Tensor out_data_tensor = saved[2];

    torch::Tensor out_grad = grad_outputs[0];
    ContextPtr context = GetContext(out_grad);

    RaggedShape shape = RaggedShape2(&row_splits, &row_ids, row_ids.Dim());

    // First, compute the gradient sum of each sublist
    Dtype t = ScalarTypeToDtype(out_grad.scalar_type());
    FOR_REAL_TYPES(t, T, {
      Array1<T> out_grad_sum(context, num_sublists);
      T *out_grad_sum_data = out_grad_sum.Data();

      const T *out_grad_data = out_grad.data_ptr<T>();

      int64_t stride = out_grad.strides()[0];

      if (stride != 0) {
        Array1<T> out_grad_array = FromTorch<T>(out_grad);
        Ragged<T> out_grad_ragged(shape, out_grad_array);
        SumPerSublist<T>(out_grad_ragged, 0, &out_grad_sum);
      } else {
        // stride is 0;
        // the sum is the number_of_elements_in_the_sublist * out_grad[0]
        const int32_t *row_splits_data = row_splits.Data();
        K2_EVAL(
            context, out_grad_sum.Dim(), lambda_compute_out_grad_sum,
            (int32_t i)->void {
              int32_t begin = row_splits_data[i];
              int32_t end = row_splits_data[i + 1];
              out_grad_sum_data[i] = (end - begin) * out_grad_data[0];
            });
      }

      Array1<T> ans_grad_array(context, row_ids.Dim());
      T *ans_grad_data = ans_grad_array.Data();

      const T *out_data = out_data_tensor.data_ptr<T>();
      const int32_t *row_ids_data = row_ids.Data();
      int32_t num_elements = ans_grad_array.Dim();

      K2_EVAL(
          context, num_elements, lambda_set_ans_grad, (int32_t i)->void {
            int32_t row = row_ids_data[i];
            T scale = out_grad_sum_data[row];
            ans_grad_data[i] =
                out_grad_data[i * stride] - exp(out_data[i]) * scale;
          });

      return {
          torch::Tensor(),          // ragged
          torch::Tensor(),          // use_log
          ToTorch(ans_grad_array),  // dummy
          torch::Tensor()           // out
      };
    });

    // Unreachable code
    return {
        torch::Tensor(),  // ragged
        torch::Tensor(),  // use_log
        torch::Tensor(),  // dummy
        torch::Tensor()   // out
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_NORMALIZE_H_
