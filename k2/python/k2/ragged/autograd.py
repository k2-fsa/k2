# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import List, Tuple

import torch
import _k2

from .tensor import RaggedFloat


class _NormalizeScores(torch.autograd.Function):

    @staticmethod
    def forward(ctx, src: RaggedFloat, use_log: bool, out: List[RaggedFloat],
                unused_scores: torch.Tensor) -> torch.Tensor:
        '''Normalize a ragged tensor of scores.

        If use_log is True, the normalization per sublist is done as follows:

            1. Compute the log sum per sublist
            2. Subtract the log sum computed above from the sublist and return
            it

        If use_log is False, the normalization per sublist is done as follows:

            1. Compute the sum per sublist
            2. Divide the sublist by the above sum and return the resulting
            sublist

        Note:
          If a sublist contains 3 elements `[a, b, c]`, then the log sum
          is defined as::

            s = log(exp(a) + exp(b) + exp(c))

          The resulting sublist looks like below if use_log is True::

            [a - s, b - s, c - s]

          If use_log is False, the resulting sublist looks like::

            [a/(a+b+c), b/(a+b+c), c/(a+b+c)]

        Caution:
          autograd is currently not implemented if use_log is False.

        Args:
          src:
            The source ragged tensor.
          use_log:
            It indicates which kind of normalization to be applied.
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
        ans_ragged = _k2.normalize_per_sublist(src.ragged, use_log)
        out[0] = RaggedFloat(ans_ragged)
        ctx.out_ragged = out[0].ragged  # save for backward
        ctx.use_log = use_log
        return out[0].values

    @staticmethod
    def backward(ctx,
                 out_grad: torch.Tensor) -> Tuple[None, None, torch.Tensor]:
        out_ragged = ctx.out_ragged
        use_log = ctx.use_log
        assert use_log is True
        ans_grad = _k2.normalize_per_sublist_backward(out_ragged, use_log,
                                                      out_grad)

        return (
            None,  # src
            None,  # use_log
            None,  # out
            ans_grad)  # unused_scores


def normalize_scores(src: RaggedFloat, use_log: bool) -> RaggedFloat:
    '''Normalize a ragged tensor of scores.

    If use_log is True, the normalization per sublist is done as follows:

        1. Compute the log sum per sublist
        2. Subtract the log sum computed above from the sublist and return
        it

    If use_log is False, the normalization per sublist is done as follows:

        1. Compute the sum per sublist
        2. Divide the sublist by the above sum and return the resulting
        sublist

    Note:
      If a sublist contains 3 elements `[a, b, c]`, then the log sum
      is defined as::

        s = log(exp(a) + exp(b) + exp(c))

      The resulting sublist looks like below if use_log is True::

        [a - s, b - s, c - s]

      If use_log is False, the resulting sublist looks like::

        [a/(a+b+c), b/(a+b+c), c/(a+b+c)]

    Caution:
      autograd is currently not implemented if use_log is False.

    Args:
      src:
        The source ragged tensor.
      use_log:
        It indicates which kind of normalization to be applied.

    Returns:
      The normalized ragged tensor.
    '''
    out = [None]  # placeholder

    # the return value is discarded for the following call
    # as it equals to out[0].values
    _NormalizeScores.apply(src, use_log, out, src.values)

    return out[0]
