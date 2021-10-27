/**
 * @brief A wrapper around k2::GetForwardScores to support autograd.
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

#ifndef K2_TORCH_CSRC_AUTOGRAD_GET_FORWARD_SCORES_H_
#define K2_TORCH_CSRC_AUTOGRAD_GET_FORWARD_SCORES_H_

#include <utility>

#include "k2/csrc/fsa_utils.h"
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

template <typename T>
torch::Tensor BackpropGetForwardScores(FsaClass &fsas, bool use_double_scores,
                                       bool log_semiring,
                                       torch::Tensor forward_scores,
                                       torch::Tensor forward_scores_deriv) {
  Ragged<int32_t> state_batches = fsas.GetStateBatches(/*transpose*/ true);
  Ragged<int32_t> leaving_arc_batches = fsas.GetLeavingArcIndexBatches();
  Array1<int32_t> entering_arcs;
  const Array1<int32_t> *p_entering_arcs = nullptr;
  if (!log_semiring) {
    entering_arcs = fsas.GetEnteringArcs(use_double_scores);
    p_entering_arcs = &entering_arcs;
  }

  Array1<T> forward_scores_array = FromTorch<T>(forward_scores);
  Array1<T> forward_scores_deriv_array = FromTorch<T>(forward_scores_deriv);

  Array1<T> ans = BackpropGetForwardScores<T>(
      fsas.fsa, state_batches, leaving_arc_batches, log_semiring,
      p_entering_arcs, forward_scores_array, forward_scores_deriv_array);

  return ToTorch(ans);
}

// see https://pytorch.org/tutorials/advanced/cpp_autograd
class GetForwardScoresFunction
    : public torch::autograd::Function<GetForwardScoresFunction> {
 public:
  /* Compute the forward scores of an FsaVec. It is a wrapper around
     k2::GetForwardScores() to support autograd.


     @param fsas The input FsaVec.
     @param use_double_scores  True to use double precision.
                               False to use single precision.
     @param log_semiring  True to use log semiring while computing
                          forward scores. False to use tropical semiring.
     @param dummy  Its purpose is to make autograd to track the operations on
                   the input `fsas`. It is the same as `fsas.scores`.

     @return Return a 1-D tensor containing the forward scores of each state.
             Its dtype is torch.float64 if use_double_scores is True.
             The dtype is torch.float32 if use_double_scores is False.
   */
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               FsaClass &fsas, bool use_double_scores,
                               bool log_semiring, torch::Tensor /*dummy*/) {
    torch::Tensor forward_scores =
        fsas.GetForwardScoresImpl(use_double_scores, log_semiring).detach();

    ctx->saved_data["fsas"] =
        torch::make_custom_class<k2::FsaClassHolder>(&fsas);
    ctx->saved_data["use_double_scores"] = use_double_scores;
    ctx->saved_data["log_semiring"] = log_semiring;

    ctx->save_for_backward({forward_scores});

    return forward_scores;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
    torch::intrusive_ptr<FsaClassHolder> fsa_class_holder =
        ctx->saved_data["fsas"].toCustomClass<FsaClassHolder>();

    FsaClass *fsas = fsa_class_holder->fsas;
    bool use_double_scores = ctx->saved_data["use_double_scores"].toBool();
    bool log_semiring = ctx->saved_data["log_semiring"].toBool();

    auto saved = ctx->get_saved_variables();
    torch::Tensor forward_scores = saved[0];
    torch::Tensor forward_scores_deriv = grad_outputs[0];

    Dtype t = kFloatDtype;
    if (use_double_scores) t = kDoubleDtype;
    torch::Tensor scores_grad;
    FOR_REAL_TYPES(t, T, {
      scores_grad =
          BackpropGetForwardScores<T>(*fsas, use_double_scores, log_semiring,
                                      forward_scores, forward_scores_deriv);
    });

    return {
        torch::Tensor(),  // fsas
        torch::Tensor(),  // use_double_scores
        torch::Tensor(),  // log_semiring
        scores_grad       // dummy
    };
  }
};

}  // namespace k2

#endif  // K2_TORCH_CSRC_AUTOGRAD_GET_FORWARD_SCORES_H_
