# Copyright (c)  2021  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
#
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

import os
import torch
import _k2
from torch import Tensor
from typing import Tuple, Optional, Sequence, Union, List


class MutualInformationRecursionFunction(torch.autograd.Function):
    """A recursion that is useful in computing mutual information between two
    sequences of real vectors, but may be useful more generally in
    sequence-to-sequence tasks where monotonic alignment between pairs of
    sequences is desired.
    """

    @staticmethod
    def forward(
        ctx,
        px: torch.Tensor,
        py: torch.Tensor,
        pxy_grads: List[Optional[torch.Tensor]],
        boundary: Optional[torch.Tensor] = None,
        return_grad: bool = False,
    ) -> torch.Tensor:
        """
        Computing mutual information between two sequences of real vectors.
        Args:
          px:
            A torch.Tensor of some floating point type, with shape ``[B][S][T]``
            if modified, ``[B][S][T+1]`` if not modified.
            where ``B`` is the batch size, ``S`` is the
            length of the ``x`` sequence (including representations of
            ``EOS`` symbols but not ``BOS`` symbols), and ``T`` is the
            length of the ``y`` sequence (including representations of
            ``EOS`` symbols but not  ``BOS`` symbols).  In the mutual
            information application, ``px[b][s][t]`` would represent the
            following log odds ratio; ignoring the b index on the right
            to make the notation more
            compact::

              px[b][s][t] =  log [ p(x_s | x_{0..s-1}, y_{0..t-1}) / p(x_s) ]

            This expression also implicitly includes the log-probability of
            choosing to generate an ``x`` value as opposed to a ``y`` value.  In
            practice it might be computed as ``a + b``, where ``a`` is the log
            probability of choosing to extend the sequence of length ``(s,t)``
            with an ``x`` as opposed to a ``y`` value; and ``b`` might in
            practice be of the form::

                log(N exp f(x_s, y_{t-1}) / sum_t'  exp f(x_s, y_t'))

            where ``N`` is the number of terms that the sum over ``t'``
            included, which might include some or all of the other sequences as
            well as this one.

            Note:
              we don't require ``px`` and ``py`` to be contiguous, but the
              code assumes for optimization purposes that the ``T`` axis has
              stride 1.

          py:
            A torch.Tensor of the same dtype as ``px``, with shape
            ``[B][S+1][T]``, representing::

              py[b][s][t] =  log [ p(y_t | x_{0..s-1}, y_{0..t-1}) / p(y_t) ]

            This function does not treat ``x`` and ``y`` differently; the only
            difference is that for optimization purposes we assume the last axis
            (the ``t`` axis) has stride of 1; this is true if ``px`` and ``py``
            are contiguous.

          pxy_grads:
            A List to store the return grads of ``px`` and ``py``
            if return_grad == True.
            Remain unchanged if return_grad == False.

            See `this PR <https://github.com/k2-fsa/k2/pull/924>` for more
            information about why we add this parameter.

            Note:
              the length of the list must be 2, where the first element
              represents the grads of ``px`` and the second one represents
              the grads of ``py``.

          boundary:
            If supplied, a torch.LongTensor of shape ``[B][4]``, where each
            row contains ``[s_begin, t_begin, s_end, t_end]``,
            with ``0 <= s_begin <= s_end <= S`` and
            ``0 <= t_begin <= t_end < T``
            (this implies that empty sequences are allowed).
            If not supplied, the values ``[0, 0, S, T]`` will be assumed.
            These are the beginning and one-past-the-last positions in the
            ``x`` and ``y`` sequences respectively, and can be used if not
            all sequences are
            of the same length.

          return_grad:
            Whether to return grads of ``px`` and ``py``, this grad standing
            for the occupation probability is the output of the backward with a
            ``fake gradient`` the ``fake gradient`` is the same as the gradient
            you'd get if you did
            ``torch.autograd.grad((scores.sum()), [px, py])``.
            This is useful to implement the pruned version of rnnt loss.

        Returns:
          Returns a torch.Tensor of shape ``[B]``, containing the log of
          the mutual information between the b'th pair of sequences.  This is
          defined by the following recursion on ``p[b,s,t]`` (where ``p``
          is of shape ``[B,S+1,T+1]``), representing a mutual information
          between sub-sequences of lengths ``s`` and ``t``::

                 p[b,0,0] = 0.0
                 p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                    p[b,s,t-1] + py[b,s,t-1])
                           (if s > 0 or t > 0)

          where we handle edge cases by treating quantities with negative
          indexes as **-infinity**.  The extension to cases where the
          boundaries are specified should be obvious; it just works on
          shorter sequences with offsets into ``px`` and ``py``.
        """
        (B, S, T1) = px.shape
        T = py.shape[-1]
        assert T1 in [T, T + 1]
        assert py.shape == (B, S + 1, T)
        if boundary is not None:
            assert boundary.shape == (B, 4)

        # p is a tensor of shape (B, S + 1, T + 1) were p[s][t] is the
        # the mutual information of the pair of subsequences of x and y that
        # are of length s and t respectively.  p[0][0] will be 0.0 and p[S][T]
        # is the mutual information of the entire pair of sequences,
        # i.e. of lengths S and T respectively.
        # It is computed as follows (in C++ and CUDA):
        #       p[b,0,0] = 0.0
        #       p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
        #                          p[b,s,t-1] + py[b,s,t-1])
        #               if s > 0 or t > 0,
        #               treating values with any -1 index as -infinity.
        #      .. if `boundary` is set, we start fom p[b,s_begin,t_begin]=0.0.

        p = torch.empty(B, S + 1, T + 1, device=px.device, dtype=px.dtype)

        ans = _k2.mutual_information_forward(px, py, boundary, p)

        px_grad, py_grad = None, None
        if return_grad or px.requires_grad or py.requires_grad:
            ans_grad = torch.ones(B, device=px.device, dtype=px.dtype)
            (px_grad, py_grad) = _k2.mutual_information_backward(
                px, py, boundary, p, ans_grad)
            ctx.save_for_backward(px_grad, py_grad)
        assert len(pxy_grads) == 2
        pxy_grads[0] = px_grad
        pxy_grads[1] = py_grad

        return ans

    @staticmethod
    def backward(
        ctx, ans_grad: Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        (px_grad, py_grad) = ctx.saved_tensors
        (B,) = ans_grad.shape
        ans_grad = ans_grad.reshape(B, 1, 1)  # (B, 1, 1)
        px_grad *= ans_grad
        py_grad *= ans_grad
        return (px_grad, py_grad, None, None, None)


def mutual_information_recursion(
    px: Tensor,
    py: Tensor,
    boundary: Optional[Tensor] = None,
    return_grad: bool = False,
) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tensor]:
    """A recursion that is useful in computing mutual information between two
    sequences of real vectors, but may be useful more generally in
    sequence-to-sequence tasks where monotonic alignment between pairs of
    sequences is desired.  The definitions of the arguments are definitions that
    would be used when computing this type of mutual information, but you can
    also view them as arbitrary quantities and just make use of the formula
    computed by this function.

    Args:
      px:
        A torch.Tensor of some floating point type, with shape ``[B][S][T+1]``,
        where ``B`` is the batch size, ``S`` is the length of the ``x`` sequence
        (including representations of ``EOS`` symbols but not ``BOS`` symbols),
        and ``T`` is the length of the ``y`` sequence (including representations
        of ``EOS`` symbols but not ``BOS`` symbols).  In the mutual information
        application, ``px[b][s][t]`` would represent the following log odds
        ratio; ignoring the b index on the right to make the notation more
        compact::

          px[b][s][t] =  log [ p(x_s | x_{0..s-1}, y_{0..t-1}) / p(x_s) ]

        This expression also implicitly includes the log-probability of
        choosing to generate an ``x`` value as opposed to a ``y`` value.  In
        practice it might be computed as ``a + b``, where ``a`` is the log
        probability of choosing to extend the sequence of length ``(s,t)``
        with an ``x`` as opposed to a ``y`` value; and ``b`` might in practice
        be of the form::

            log(N exp f(x_s, y_{t-1}) / sum_t'  exp f(x_s, y_t'))

        where ``N`` is the number of terms that the sum over ``t'`` included,
        which might include some or all of the other sequences as well as this
        one.

        Note:
          we don't require ``px`` and ``py`` to be contiguous, but the
          code assumes for optimization purposes that the ``T`` axis has
          stride 1.

      py:
        A torch.Tensor of the same dtype as ``px``, with shape ``[B][S+1][T]``,
        representing::

          py[b][s][t] =  log [ p(y_t | x_{0..s-1}, y_{0..t-1}) / p(y_t) ]

        This function does not treat ``x`` and ``y`` differently; the only
        difference is that for optimization purposes we assume the last axis
        (the ``t`` axis) has stride of 1; this is true if ``px`` and ``py`` are
        contiguous.

      boundary:
        If supplied, a torch.LongTensor of shape ``[B][4]``, where each
        row contains ``[s_begin, t_begin, s_end, t_end]``,
        with ``0 <= s_begin <= s_end <= S`` and ``0 <= t_begin <= t_end < T``
        (this implies that empty sequences are allowed).
        If not supplied, the values ``[0, 0, S, T]`` will be assumed.
        These are the beginning and one-past-the-last positions in the ``x`` and
        ``y`` sequences respectively, and can be used if not all sequences are
        of the same length.

      return_grad:
        Whether to return grads of ``px`` and ``py``, this grad standing for the
        occupation probability is the output of the backward with a
        ``fake gradient`` the ``fake gradient`` is the same as the gradient
        you'd get if you did ``torch.autograd.grad((scores.sum()), [px, py])``.
        This is useful to implement the pruned version of rnnt loss.

    Returns:
      Returns a torch.Tensor of shape ``[B]``, containing the log of the mutual
      information between the b'th pair of sequences.  This is defined by
      the following recursion on ``p[b,s,t]`` (where ``p`` is of shape
      ``[B,S+1,T+1]``), representing a mutual information between sub-sequences
      of lengths ``s`` and ``t``::

             p[b,0,0] = 0.0
        if !modified:
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
        if modified:
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])

      where we handle edge cases by treating quantities with negative indexes
      as **-infinity**.  The extension to cases where the boundaries are
      specified should be obvious; it just works on shorter sequences with
      offsets into ``px`` and ``py``.
    """
    assert px.ndim == 3
    B, S, T1 = px.shape
    T = py.shape[-1]
    assert px.shape[-1] in [T, T + 1]  # if T, then "modified".
    assert py.shape == (B, S + 1, T)
    assert px.dtype == py.dtype
    if boundary is not None:
        assert boundary.dtype == torch.int64
        assert boundary.shape == (B, 4)
        for s_begin, t_begin, s_end, t_end in boundary.tolist():
            assert 0 <= s_begin <= s_end <= S
            assert 0 <= t_begin <= t_end <= T

    # The following statements are for efficiency
    px, py = px.contiguous(), py.contiguous()

    pxy_grads = [None, None]
    scores = MutualInformationRecursionFunction.apply(px, py, pxy_grads,
                                                      boundary, return_grad)
    px_grad, py_grad = pxy_grads
    return (scores, (px_grad, py_grad)) if return_grad else scores


def _inner_product(a: Tensor, b: Tensor) -> Tensor:
    """
    Does inner product on the last dimension, with expected broadcasting,
    i.e. equivalent to (a * b).sum(dim=-1)
    without creating a large temporary.
    """
    assert a.shape[-1] == b.shape[-1]  # The last dim must be equal
    a = a.unsqueeze(-2)  # (..., 1, K)
    b = b.unsqueeze(-1)  # (..., K, 1)
    c = torch.matmul(a, b)  # (..., 1, 1)
    return c.squeeze(-1).squeeze(-1)


def joint_mutual_information_recursion(
    px: Sequence[Tensor],
    py: Sequence[Tensor],
    boundary: Optional[Tensor] = None,
) -> Sequence[Tensor]:
    """A recursion that is useful for modifications of RNN-T and similar loss
    functions, where the recursion probabilities have a number of terms and you
    want them reported separately.  See mutual_information_recursion() for more
    documentation of the basic aspects of this.

    Args:
      px:
        a sequence of Tensors, each of the same shape [B][S][T+1]
      py:
        a sequence of Tensor, each of the same shape [B][S+1][T],
        the sequence must be the same length as px.
      boundary:
        optionally, a LongTensor of shape [B][4] containing rows
        [s_begin, t_begin, s_end, t_end], with 0 <= s_begin <= s_end <= S
        and 0 <= t_begin <= t_end < T, defaulting to [0, 0, S, T].
        These are the beginning and one-past-the-last positions in the x
        and y sequences respectively, and can be used if not all
        sequences are of the same length.
    Returns:
      a Tensor of shape (len(px), B),
      whose sum over dim 0 is the total log-prob of the recursion mentioned
      below, per sequence. The first element of the sequence of length len(px)
      is "special", in that it has an offset term reflecting the difference
      between sum-of-log and log-of-sum; for more interpretable loss values,
      the "main" part of your loss function should be first.

      The recursion below applies if boundary == None, when it defaults
      to (0, 0, S, T); where px_sum, py_sum are the sums of the elements of px
      and py::

          p = tensor of shape (B, S+1, T+1), containing -infinity
          p[b,0,0] = 0.0
          # do the following in loop over s and t:
          p[b,s,t] = log_add(p[b,s-1,t] + px_sum[b,s-1,t],
                              p[b,s,t-1] + py_sum[b,s,t-1])
                      (if s > 0 or t > 0)
          return b[:][S][T]

    This function lets you implement the above recursion efficiently, except
    that it gives you a breakdown of the contribution from all the elements of
    px and py separately.  As noted above, the first element of the
    sequence is "special".
    """
    N = len(px)
    assert len(py) == N and N > 0
    B, S, T1 = px[0].shape
    T = py[0].shape[2]
    assert T1 in [T, T + 1]  # T if modified...
    assert py[0].shape == (B, S + 1, T)
    assert px[0].dtype == py[0].dtype

    px_cat = torch.stack(
        px, dim=0
    )  # (N, B, S, T+1) if !modified,(N, B, S, T) if modified.
    py_cat = torch.stack(py, dim=0)  # (N, B, S+1, T)
    px_tot = px_cat.sum(dim=0)  # (B, S, T+1)
    py_tot = py_cat.sum(dim=0)  # (B, S+1, T)

    if boundary is not None:
        assert boundary.dtype == torch.int64
        assert boundary.shape == (B, 4)
        for s_begin, t_begin, s_end, t_end in boundary.tolist():
            assert 0 <= s_begin <= s_end <= S
            assert 0 <= t_begin <= t_end <= T

    # The following statements are for efficiency
    px_tot, py_tot = px_tot.contiguous(), py_tot.contiguous()

    assert px_tot.ndim == 3
    assert py_tot.ndim == 3

    p = torch.empty(B, S + 1, T + 1, device=px_tot.device, dtype=px_tot.dtype)

    # note, tot_probs is without grad.
    tot_probs = _k2.mutual_information_forward(px_tot, py_tot, boundary, p)

    # this is a kind of "fake gradient" that we use, in effect to compute
    # occupation probabilities.  The backprop will work regardless of the
    # actual derivative w.r.t. the total probs.
    ans_grad = torch.ones(B, device=px_tot.device, dtype=px_tot.dtype)

    (px_grad,
     py_grad) = _k2.mutual_information_backward(px_tot, py_tot, boundary, p,
                                                ans_grad)

    px_grad = px_grad.reshape(1, B, -1)
    py_grad = py_grad.reshape(1, B, -1)
    px_cat = px_cat.reshape(N, B, -1)
    py_cat = py_cat.reshape(N, B, -1)
    # get rid of -inf, would generate nan on product with 0
    px_cat = px_cat.clamp(min=torch.finfo(px_cat.dtype).min)
    py_cat = py_cat.clamp(min=torch.finfo(py_cat.dtype).min)

    x_prods = _inner_product(px_grad, px_cat)  # (N, B)
    y_prods = _inner_product(py_grad, py_cat)  # (N, B)

    # If all the occupation counts were exactly 1.0 (i.e. no partial counts),
    # "prods" should be equal to "tot_probs"; however, in general, "tot_probs"
    # will be more positive due to the difference between log-of-sum and
    # sum-of-log
    prods = x_prods + y_prods  # (N, B)
    with torch.no_grad():
        offset = tot_probs - prods.sum(dim=0)  # (B,)
    prods[0] += offset
    return prods  # (N, B)
