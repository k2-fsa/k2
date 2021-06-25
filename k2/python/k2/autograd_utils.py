# Copyright      2020-2021  Xiaomi Corp.   (authors: Daniel Povey
#                                                    Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import torch

from .fsa import Fsa
import _k2


# This is a trick when we want to set the scores of an Fsa to a certain
# value but we know they *already have that value*.
#
# The 'forward' function "pretends" to set out_fsa.scores to
# 'unused_in_fsa_scores', and it returns out_fsa.scores.  This will
# attach a _grad_fn to out_fsa.scores, if unused_in_fsa_scores.requires_grad
# was true.
#
# The backprop is as if the function was just a copy (i.e. it copies
# the output gradient)
class _PhantomSetScoresFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, out_fsa: Fsa,
                unused_in_fsa_scores: torch.Tensor) -> torch.Tensor:
        if False:
            # TODO(dan): remove the following assertion at some point.
            assert torch.all(torch.eq(out_fsa.scores, unused_in_fsa_scores))
        return out_fsa.scores

    @staticmethod
    def backward(ctx, out_fsa_scores_grad: torch.Tensor
                ) -> Tuple[None, torch.Tensor]:  # noqa
        return None, out_fsa_scores_grad


class _PhantomIndexSelectScoresFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, out_fsa: Fsa, unused_in_fsa_scores: torch.Tensor,
                arc_map: torch.Tensor) -> torch.Tensor:
        if False:
            # TODO(fangjun): this is for debugging only. Can be removed.
            expected_scores = _k2.index_select(unused_in_fsa_scores, arc_map)
            assert torch.all(torch.eq(out_fsa.scores, expected_scores))

        ctx.save_for_backward(unused_in_fsa_scores, arc_map)
        return out_fsa.scores

    @staticmethod
    def backward(ctx, out_fsa_scores_grad: torch.Tensor
                ) -> Tuple[None, torch.Tensor, None]:  # noqa
        unused_in_fsa_scores, arc_map = ctx.saved_tensors

        ans = torch.zeros(unused_in_fsa_scores.shape,
                          dtype=torch.float32,
                          device=unused_in_fsa_scores.device,
                          requires_grad=False)
        _k2.index_add(arc_map, out_fsa_scores_grad, ans)
        return (
            None,  # out_fsa
            ans,  # unused_in_fsa_scores
            None  # arc_map
        )


class _PhantomIndexAndSumScoresFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, out_fsa: Fsa, unused_in_fsa_scores: torch.Tensor,
                arc_map: _k2.RaggedInt) -> torch.Tensor:
        if False:
            # TODO(fangjun): this is for debugging only. Can be removed.
            expected_scores = _k2.index_and_sum(
                unused_in_fsa_scores.contiguous(), arc_map)
            assert torch.all(torch.eq(out_fsa.scores, expected_scores))

        ctx.save_for_backward(unused_in_fsa_scores)
        ctx.arc_map = arc_map
        return out_fsa.scores

    @staticmethod
    def backward(ctx, out_fsa_scores_grad: torch.Tensor
                ) -> Tuple[None, torch.Tensor, None]:  # noqa
        unused_in_fsa_scores, = ctx.saved_tensors
        arc_map = ctx.arc_map

        expanded = _k2.index_select(out_fsa_scores_grad, arc_map.row_ids(1))
        ans = torch.zeros(unused_in_fsa_scores.shape,
                          dtype=torch.float32,
                          device=unused_in_fsa_scores.device,
                          requires_grad=False)
        _k2.index_add(arc_map.values(), expanded, ans)

        return (
            None,  # out_fsa
            ans,  # unused_in_fsa_scores
            None  # arc_map
        )


def phantom_set_scores_to(fsa: Fsa, scores_value: torch.Tensor) -> None:
    # we don't need the output value of the following call
    # (which is fsa.scores), since it is accessible through `fsa`.
    # The fact that it was returned from a torch.autograd.Function
    # gives it a grad_fn (assuming scores_value had requires_grad == True.)
    _PhantomSetScoresFunction.apply(fsa, scores_value)


def phantom_index_select_scores(fsa: Fsa, scores_value: torch.Tensor,
                                arc_map: torch.Tensor) -> None:
    _PhantomIndexSelectScoresFunction.apply(fsa, scores_value, arc_map)


def phantom_index_and_sum_scores(fsa: Fsa, scores_value: torch.Tensor,
                                 arc_map: _k2.RaggedInt) -> None:
    _PhantomIndexAndSumScoresFunction.apply(fsa, scores_value, arc_map)
