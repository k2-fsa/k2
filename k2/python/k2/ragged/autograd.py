# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
# See ../../../../LICENSE for clarification regarding multiple authors

from typing import List, Tuple

import torch
import _k2

from .tensor import RaggedFloat


class _NormalizeScores(torch.autograd.Function):

    @staticmethod
    def forward(ctx, src: RaggedFloat, out: List[RaggedFloat],
                unused_scores: torch.Tensor) -> torch.Tensor:
        '''Normalize a ragged tensor of scores.

        The normalization per sublist is done as follows:

            1. Compute the log sum per sublist
            2. Subtract the log sum computed above from the sublist and return
            it

        Note:
          If a sublist contains 3 elements `[a, b, c]`, then the log sum
          is defined as::

            s = log(exp(a) + exp(b) + exp(c))

          The resulting sublist looks like::

            [a - s, b - s, c - s]

        Args:
          src:
            The source ragged tensor.
          out:
            The output ragged tensor is put in this list. It is returned
            in the list since this function can only return values of type
            `torch.Tensor`. On input, we check that `len(out) == 1`.
          unused_scores:
            Its sole purpose is for autograd. It equals to `src.values`.
        Returns:
          Returns a tensor that equals to `out.values`. Callers should
          discard the return value.
        '''
        assert len(out) == 1
        ans_ragged = _k2.normalize_per_sublist(src.ragged)
        out[0] = RaggedFloat(ans_ragged)
        ctx.out = out[0]  # save for backward
        return out[0].values

    @staticmethod
    def backward(ctx,
                 out_grad: torch.Tensor) -> Tuple[None, None, torch.Tensor]:
        out = ctx.out
        ans_grad = _k2.normalize_per_sublist_backward(out.ragged, out_grad)
        return (
            None,  # src
            None,  # out
            ans_grad)


def normalize_scores(src: RaggedFloat) -> RaggedFloat:
    '''Normalize a ragged tensor of scores.

    The normalization per sublist is done as follows:

        1. Compute the log sum per sublist
        2. Subtract the log sum computed above from the sublist and return
        it

    Note:
      If a sublist contains 3 elements `[a, b, c]`, then the log sum
      is defined as::

        s = log(exp(a) + exp(b) + exp(c))

      The resulting sublist looks like::

        [a - s, b - s, c - s]

    Args:
      src:
        The source ragged tensor.
    Returns:
      The normalized ragged tensor.
    '''
    out = [None]  # placeholder

    # the return value is discarded for the following call
    # as it equals to out[0].values
    _NormalizeScores.apply(src, out, src.values)

    return out[0]
