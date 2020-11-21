# Copyright (c)  2020  Xiaomi Corp.   (author: Daniel Povey)
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Tuple
import torch


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
    def forward(ctx, out_fsa,
                unused_in_fsa_scores: torch.Tensor) -> torch.Tensor:
        # TODO(dan): remove the following assertion at some point.
        assert torch.all(torch.eq(out_fsa.scores, unused_in_fsa_scores))
        return out_fsa.scores

    @staticmethod
    def backward(ctx, out_fsa_scores_grad: torch.Tensor
                ) -> Tuple[None, torch.Tensor]:  # noqa
        return None, out_fsa_scores_grad


def phantom_set_scores_to(fsa, scores_value) -> None:
    # we don't need the output value of the following call
    # (which is fsa.scores), since it is accessible through `fsa`.
    # The fact that it was returned from a torch.autograd.Function
    # gives it a grad_fn (assuming scores_value had requires_grad == True.)
    _PhantomSetScoresFunction.apply(fsa, scores_value)
