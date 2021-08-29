/**
 * @brief A wrapper around k2::ArcSort to support autograd
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_ARC_SORT_H
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_ARC_SORT_H

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

using namespace torch::autograd;

namespace k2 {

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class ArcSortFunction : public torch::autograd::Function<ArcSortFunction> {
 public:
  /* ArcSort an Fsa. It is a wrapper around k2::ArcSort, supporting autograd.


     @param ragged The input Fsa.
     @param dummy  Its purpose is to make autograd to track the operations on
                   the input `ragged`. It is the same as `ragged.scores`.
     @param out  The output Fsa.

     @return Return a 1-D unused tensor, which is out->scores.
   */
  static torch::Tensor forward(AutogradContext *ctx,
                               /*const*/ RaggedArc &ragged,
                               torch::Tensor /*dummy*/, RaggedArc *out) {
    Array1<int32_t> arc_map;
    ArcSort(ragged.fsa, &out->fsa, &arc_map);

    ctx->save_for_backward({ToTorch(arc_map)});

    return out->Scores();
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor arc_map_tensor = saved[0];
    Array1<int32_t> arc_map = FromTorch<int32_t>(arc_map_tensor);
    torch::Tensor grad_output_tensor = grad_outputs[0];
    Tensor grad_output = FromTorch(grad_output_tensor, TensorTag{});

    Tensor ans = Index(grad_output, arc_map, /*allow_minus_one*/ false,
                       /*default_value*/ 0);

    return {
        torch::Tensor(),  // ragged
        ToTorch(ans),     // dummy
        torch::Tensor()   // out
    };
  }
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_ARC_SORT_H
