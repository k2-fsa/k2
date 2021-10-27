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

#ifndef K2_TORCH_CSRC_AUTOGRAD_UTILS_H_
#define K2_TORCH_CSRC_AUTOGRAD_UTILS_H_

#include <utility>

#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/ops.h"

namespace k2 {

class PhantomSetScoresFunction
    : public torch::autograd::Function<PhantomSetScoresFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               FsaClass &fsa,
                               torch::Tensor unused_in_fsa_scores) {
    return fsa.Scores();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    torch::Tensor grad_output_tensor = grad_outputs[0];
    return {
        torch::Tensor(),     // fsa
        grad_output_tensor,  // unused_in_fsa_scores
    };
  }
};

class PhantomIndexSelectScoresFunction
    : public torch::autograd::Function<PhantomIndexSelectScoresFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               FsaClass &fsa,
                               torch::Tensor unused_in_fsa_scores,
                               torch::Tensor arc_map) {
    ctx->save_for_backward({unused_in_fsa_scores, arc_map});
    return fsa.Scores();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor scores_tensor = saved[0];
    torch::Tensor arc_map_tensor = saved[1];
    torch::Tensor grad_output_tensor = grad_outputs[0];

    torch::Tensor ans = torch::zeros_like(scores_tensor);

    IndexAdd(arc_map_tensor, grad_output_tensor, &ans);

    return {
        torch::Tensor(),  // fsa
        ans,              // unused_in_fsa_scores
        torch::Tensor(),  // arc_map
    };
  }
};

class PhantomIndexAndSumScoresFunction
    : public torch::autograd::Function<PhantomIndexAndSumScoresFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               FsaClass &fsa,
                               torch::Tensor unused_in_fsa_scores,
                               Ragged<int32_t> &arc_map) {
    DeviceGuard guard(arc_map.Context());
    torch::Tensor row_ids1 = ToTorch(arc_map.RowIds(1));
    ctx->save_for_backward(
        {unused_in_fsa_scores, row_ids1, ToTorch(arc_map.values)});
    return fsa.Scores();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list scores_grad) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor unused_in_fsa_scores = saved[0];
    torch::Tensor row_ids1 = saved[1];
    torch::Tensor values = saved[2];
    torch::Tensor scores_grad_tensor = scores_grad[0];
    torch::Tensor expanded = IndexSelect(scores_grad_tensor, row_ids1, 0);
    torch::Tensor ans = torch::zeros_like(unused_in_fsa_scores);
    IndexAdd(values, expanded, &ans);
    return {
        torch::Tensor(),  // fsa
        ans,              // unused_in_fsa_scores
        torch::Tensor(),  // arc_map
    };
  }
};

}  // namespace k2

#endif  // K2_TORCH_CSRC_AUTOGRAD_UTILS_H_
