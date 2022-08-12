# Copyright      2021  Xiaomi Corp.       (author: Daniel Povey, Wei Kang)
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

import k2
import torch
from torch import Tensor
from typing import Optional, Tuple, Union
from .mutual_information import mutual_information_recursion


def fix_for_boundary(px: Tensor, boundary: Optional[Tensor] = None) -> Tensor:
    """
    Insert -inf's into `px` in appropriate places if `boundary` is not
    None.  If boundary == None and modified == False, px[:,:,-1] will
    be -infinity, but if boundary is specified, we need px[b,:,boundary[b,3]]
    to be -infinity.

     Args:
          px: a Tensor of of shape [B][S][T+1] (this function is only
              called if modified == False, see other docs for `modified`)
              px is modified in-place and returned.
           boundary: None, or a Tensor of shape [B][3] containing
              [s_begin, t_begin, s_end, t_end]; we need only t_end.
    """
    if boundary is None:
        return px
    B, S, T1 = px.shape
    boundary = boundary[:, 3].reshape(B, 1, 1).expand(B, S, T1)
    return px.scatter_(dim=2, index=boundary, value=float("-inf"))


def get_rnnt_logprobs(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just
    addition), to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This function is called from rnnt_loss_simple(), but may be useful for
    other purposes.

    Args:
      lm:
        Language model part of un-normalized logprobs of symbols, to be added to
        acoustic model part before normalizing.  Of shape::

           [B][S+1][C]

        where B is the batch size, S is the maximum sequence length of
        the symbol sequence, possibly including the EOS symbol; and
        C is size of the symbol vocabulary, including the termination/next-frame
        symbol.
        Conceptually, lm[b][s] is a vector of length [C] representing the
        "language model" part of the un-normalized logprobs of symbols,
        given all symbols *earlier than* s in the sequence.  The reason
        we still need this for position S is that we may still be emitting
        the termination/next-frame symbol at this point.
      am:
        Acoustic-model part of un-normalized logprobs of symbols, to be added
        to language-model part before normalizing.  Of shape::

           [B][T][C]

        where B is the batch size, T is the maximum sequence length of
        the acoustic sequences (in frames); and C is size of the symbol
        vocabulary, including the termination/next-frame symbol.  It reflects
        the "acoustic" part of the probability of any given symbol appearing
        next on this frame.
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
       modified: if True, each time a real symbol is consumed a frame will
           also be consumed, so at most 1 symbol can appear per frame.
    Returns:
        (px, py) (the names are quite arbitrary).
           px: logprobs, of shape [B][S][T+1] if !modified, [B][S][T] if modified.
           py: logprobs, of shape [B][S+1][T]

      in the recursion::

          p[b,0,0] = 0.0
          if !modified:
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
          if modified:
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])
          .. where p[b][s][t] is the "joint score" of the pair of subsequences
          of length s and t respectively.  px[b][s][t] represents the
          probability of extending the subsequences of length (s,t) by one in
          the s direction, given the particular symbol, and py[b][s][t]
          represents the probability of extending the subsequences of length
          (s,t) by one in the t direction,
          i.e. of emitting the termination/next-frame symbol.

          if !modified, px[:,:,T] equals -infinity, meaning on the
          "one-past-the-last" frame we cannot emit any symbols.
          This is simply a way of incorporating
          the probability of the termination symbol on the last frame.
    """
    assert lm.ndim == 3
    assert am.ndim == 3
    assert lm.shape[0] == am.shape[0]
    assert lm.shape[2] == am.shape[2]

    (B, T, C) = am.shape
    S = lm.shape[1] - 1
    assert symbols.shape == (B, S)
    assert S >= 1
    assert T >= S

    # subtracting am_max and lm_max is to ensure the probs are in a good range
    # to do exp() without causing underflow or overflow.
    am_max, _ = torch.max(am, dim=2, keepdim=True)  # am_max: [B][T][1]
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)  # lm_max: [B][S+1][1]
    am_probs = (am - am_max).exp()
    lm_probs = (lm - lm_max).exp()
    # normalizers: [B][S+1][T]
    normalizers = (
        torch.matmul(lm_probs, am_probs.transpose(1, 2))
        + torch.finfo(am_probs.dtype).tiny
    ).log()

    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)  # [B][S+1][T]

    # px is the probs of the actual symbols..
    px_am = torch.gather(
        am.unsqueeze(1).expand(B, S, T, C),
        dim=3,
        index=symbols.reshape(B, S, 1, 1).expand(B, S, T, 1),
    ).squeeze(
        -1
    )  # [B][S][T]

    if not modified:
        px_am = torch.cat(
            (
                px_am,
                torch.full(
                    (B, S, 1),
                    float("-inf"),
                    device=px_am.device,
                    dtype=px_am.dtype,
                ),
            ),
            dim=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    px_lm = torch.gather(
        lm[:, :S], dim=2, index=symbols.unsqueeze(-1)
    )  # [B][S][1]

    px = px_am + px_lm  # [B][S][T+1], last slice with indexes out of
    # boundary is  -inf
    px[:, :, :T] -= normalizers[:, :S, :]  # px: [B][S][T+1]

    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = am[:, :, termination_symbol].unsqueeze(1)  # [B][1][T]
    py_lm = lm[:, :, termination_symbol].unsqueeze(2)  # [B][S+1][1]
    py = py_am + py_lm - normalizers

    if not modified:
        px = fix_for_boundary(px, boundary)

    return (px, py)


def rnnt_loss_simple(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
    """A simple case of the RNN-T loss, where the 'joiner' network is just
    addition.

    Args:
      lm:
        language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes
      am:
        acoustic-model part of unnormalized log-probs of symbols, with shape
        (B, T, C), i.e. batch, frame, num_classes
      symbols:
        the symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
         also be consumed, so at most 1 symbol can appear per frame.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
      return_grad:
        Whether to return grads of px and py, this grad standing for the
        occupation probability is the output of the backward with a
        `fake gradient`, the `fake gradient` is the same as the gradient you'd
        get if you did `torch.autograd.grad((-loss.sum()), [px, py])`, note, the
        loss here is the loss with reduction "none".
        This is useful to implement the pruned version of rnnt loss.
    Returns:
       If return_grad is False, returns a tensor of shape (B,), containing the
       total RNN-T loss values for each element of the batch if reduction equals
       to "none", otherwise a scalar with the reduction applied.
       If return_grad is True, the grads of px and py, which is the output of
       backward with a `fake gradient`(see above), will be returned too. And the
       returned value will be a tuple like (loss, (px_grad, py_grad)).
    """
    px, py = get_rnnt_logprobs(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        modified=modified,
    )
    scores_and_grads = mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -torch.mean(negated_loss)
    elif reduction == "sum":
        loss = -torch.sum(negated_loss)
    else:
        assert (
            False
        ), f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
    return (loss, scores_and_grads[1]) if return_grad else loss


def get_rnnt_logprobs_joint(
    logits: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Reduces RNN-T problem to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This function is called from rnnt_loss().

    Args:
      logits:
        The output of joiner network, with shape (B, T, S + 1, C),
        i.e. batch, time_seq_len, symbol_seq_len+1, num_classes
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
          also be consumed, so at most 1 symbol can appear per frame.
    Returns:
      (px, py) (the names are quite arbitrary)::

          px: logprobs, of shape [B][S][T+1]
          py: logprobs, of shape [B][S+1][T]

      in the recursion::

         p[b,0,0] = 0.0
         if !modified:
            p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                               p[b,s,t-1] + py[b,s,t-1])
         if modified:
            p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                               p[b,s,t-1] + py[b,s,t-1])
      .. where p[b][s][t] is the "joint score" of the pair of subsequences of
      length s and t respectively.  px[b][s][t] represents the probability of
      extending the subsequences of length (s,t) by one in the s direction,
      given the particular symbol, and py[b][s][t] represents the probability
      of extending the subsequences of length (s,t) by one in the t direction,
      i.e. of emitting the termination/next-frame symbol.

      if !modified, px[:,:,T] equals -infinity, meaning on the
      "one-past-the-last" frame we cannot emit any symbols.
      This is simply a way of incorporating
      the probability of the termination symbol on the last frame.
    """
    assert logits.ndim == 4
    (B, T, S1, C) = logits.shape
    S = S1 - 1
    assert symbols.shape == (B, S)
    assert S >= 1
    assert T >= S

    normalizers = torch.logsumexp(logits, dim=3)
    normalizers = normalizers.permute((0, 2, 1))

    px = torch.gather(
        logits, dim=3, index=symbols.reshape(B, 1, S, 1).expand(B, T, S, 1)
    ).squeeze(-1)
    px = px.permute((0, 2, 1))

    if not modified:
        px = torch.cat(
            (
                px,
                torch.full(
                    (B, S, 1), float("-inf"), device=px.device, dtype=px.dtype
                ),
            ),
            dim=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    px[:, :, :T] -= normalizers[:, :S, :]

    py = (
        logits[:, :, :, termination_symbol].permute((0, 2, 1)).clone()
    )  # [B][S+1][T]
    py -= normalizers

    if not modified:
        px = fix_for_boundary(px, boundary)

    return (px, py)


def rnnt_loss(
    logits: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
    reduction: Optional[str] = "mean",
) -> Tensor:
    """A normal RNN-T loss, which uses a 'joiner' network output as input,
    i.e. a 4 dimensions tensor.

    Args:
      logits:
        The output of joiner network, with shape (B, T, S + 1, C),
        i.e. batch, time_seq_len, symbol_seq_len+1, num_classes
      symbols:
        The symbol sequences, a LongTensor of shape [B][S], and elements
        in {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T] if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
          also be consumed, so at most 1 symbol can appear per frame.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`

    Returns:
      If recursion is `none`, returns a tensor of shape (B,), containing the
      total RNN-T loss values for each element of the batch, otherwise a scalar
      with the reduction applied.
    """
    px, py = get_rnnt_logprobs_joint(
        logits=logits,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        modified=modified,
    )
    negated_loss = mutual_information_recursion(px=px, py=py, boundary=boundary)
    if reduction == "none":
        return -negated_loss
    elif reduction == "mean":
        return -torch.mean(negated_loss)
    elif reduction == "sum":
        return -torch.sum(negated_loss)
    else:
        assert (
            False
        ), f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"


def _adjust_pruning_lower_bound(
    s_begin: torch.Tensor, s_range: int
) -> torch.Tensor:
    """Adjust s_begin (pruning lower bound) to make it satisfied the following
    constrains

      - monotonic increasing, i.e. s_begin[i] <= s_begin[i + 1]
      - start with symbol 0 at first frame.
      - s_begin[i + 1] - s_begin[i] < s_range, whicn means that we can't skip
        any symbols.

    To make it monotonic increasing, we can use `monotonic_lower_bound` function
    in k2, which guarantee `s_begin[i] <= s_begin[i + 1]`. The main idea is:
    traverse the array in reverse order and update the elements by
    `min_value = min(a_begin[i], min_value)`, the initial `min_value` set to
    `inf`.

    The method we used to realize `s_begin[i + 1] - s_begin[i] < s_range`
    constrain is a little tricky. We first transform `s_begin` with
    `s_begin = -(s_begin - (s_range - 1) * torch.arange(0,T))`
    then we make the transformed `s_begin` monotonic increasing, after that,
    we transform back `s_begin` with the same formula as the previous
    transformation. The idea is: if we want to make
    `s_begin[i + 1] - s_begin[i] < s_range` we only need to make
    `-(s_begin[i] - i * (s_range - 1))` a non-decreasing array. Proof:

      -(s_begin[i] - i * (s_range - 1)) <= -(s_begin[i + 1] - (i + 1) * (s_range - 1))
                            -s_begin[i] <= -s_begin[i + 1] + (i + 1) * (s_range - 1) - i * (s_range - 1)
                            -s_begin[i] <= -s_begin[i + 1] + s_range - 1
            s_begin[i + 1] - s_begin[i] <= s_range - 1
            s_begin[i + 1] - s_begin[i] < s_range

    The above transformation can not guarantee the start symbol to be 0, so we
    have to make all the elements that less than 0 to be 0 before transforming
    back the `s_begin`.
    """
    # s_begin (B, T)
    (B, T) = s_begin.shape
    s_begin = k2.monotonic_lower_bound(s_begin)
    # do the magic transformation
    s_begin = -(
        s_begin - (s_range - 1) * torch.arange(0, T, device=s_begin.device)
    )
    # make the transformed tensor to be non-decreasing
    s_begin = k2.monotonic_lower_bound(s_begin)
    # make start symbol to be zero.
    s_begin = torch.clamp(s_begin, min=0)
    # do the magic transformation again to recover s_begin
    s_begin = -(
        s_begin - (s_range - 1) * torch.arange(0, T, device=s_begin.device)
    )
    return s_begin


# To get more insight of how we calculate pruning bounds, please read
# chapter 3.2 (Pruning bounds) of our Pruned RNN-T paper
# (https://arxiv.org/pdf/2206.13236.pdf)
def get_rnnt_prune_ranges(
    px_grad: torch.Tensor,
    py_grad: torch.Tensor,
    boundary: torch.Tensor,
    s_range: int,
) -> torch.Tensor:
    """Get the pruning ranges of normal rnnt loss according to the grads
    of px and py returned by mutual_information_recursion.

    For each sequence with T frames, we will generate a tensor with the shape of
    (T, s_range) containing the information that which symbols will be token
    into consideration for each frame. For example, here is a sequence with 10
    frames and the corresponding symbols are `[A B C D E F]`, if the s_range
    equals 3, one possible ranges tensor will be::

      [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3],
       [1, 2, 3], [1, 2, 3], [3, 4, 5], [3, 4, 5], [3, 4, 5]]

    which means we only consider `[A B C]` at frame 0, 1, 2, 3, and `[B C D]`
    at frame 4, 5, 6, `[D E F]` at frame 7, 8, 9.

    We can only consider limited number of symbols because frames and symbols
    are monotonic aligned, theoretically it can only generate particular range
    of symbols given a particular frame.

    Note:
      For the generated tensor ranges (assuming batch size is 1), ranges[:, 0]
      is a monotonic increasing tensor from 0 to `len(symbols)` and it satisfies
      `ranges[t+1, 0] - ranges[t, 0] < s_range` which means we won't skip any
      symbols.

    Args:
      px_grad:
        The gradient of px, see docs in `mutual_information_recursion` for more
        details of px.
      py_grad:
        The gradient of py, see docs in `mutual_information_recursion` for more
        details of py.
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame]
      s_range:
        How many symbols to keep for each frame.
    Returns:
      A tensor contains the kept symbols indexes for each frame, with shape
      (B, T, s_range).
    """
    (B, S, T1) = px_grad.shape
    T = py_grad.shape[-1]
    assert T1 in [T, T + 1]
    S1 = S + 1
    assert py_grad.shape == (B, S1, T)
    assert boundary.shape == (B, 4)
    assert S >= 1
    assert T >= S

    # s_range > S means we won't prune out any symbols. To make indexing with
    # ranges runs normally, s_range should be equal to or less than ``S + 1``.
    if s_range > S:
        s_range = S + 1

    if T1 == T:
        assert (
            s_range >= 1
        ), "Pruning range for modified RNN-T should be equal to or greater than 1, or no valid paths could survive pruning."

    else:
        assert (
            s_range >= 2
        ), "Pruning range for standard RNN-T should be equal to or greater than 2, or no valid paths could survive pruning."

    (B_stride, S_stride, T_stride) = py_grad.stride()
    blk_grad = torch.as_strided(
        py_grad,
        (B, S1 - s_range + 1, s_range, T),
        (B_stride, S_stride, S_stride, T_stride),
    )
    # (B, S1 - s_range + 1, T)
    blk_sum_grad = torch.sum(blk_grad, axis=2)

    px_pad = torch.zeros((B, 1, T1), dtype=px_grad.dtype, device=px_grad.device)
    # (B, S1, T)
    px_grad_pad = torch.cat((px_pad, px_grad), dim=1)

    # (B, S1 - s_range + 1, T)
    final_grad = blk_sum_grad - px_grad_pad[:, : S1 - s_range + 1, :T]

    # (B, T)
    s_begin = torch.argmax(final_grad, axis=1)

    # Handle the values of s_begin in padding positions.
    # -1 here means we fill the position of the last frame (before padding) with
    # padding value which is `len(symbols) - s_range + 1`.
    # This is to guarantee that we reach the last symbol at last frame (before
    # padding).
    # The shape of the mask is (B, T), for example, we have a batch containing
    # 3 sequences, their lengths are 3, 5, 6 (i.e. B = 3, T = 6), so the mask is
    # [[True, True, False, False, False, False],
    #  [True, True, True,  True,  False, False],
    #  [True, True, True,  True,  True,  False]]
    mask = torch.arange(0, T, device=px_grad.device).reshape(1, T).expand(B, T)
    mask = mask < boundary[:, 3].reshape(B, 1) - 1

    s_begin_padding = boundary[:, 2].reshape(B, 1) - s_range + 1
    # handle the cases when `len(symbols) < s_range`
    s_begin_padding = torch.clamp(s_begin_padding, min=0)

    s_begin = torch.where(mask, s_begin, s_begin_padding)

    # adjusting lower bound to make it satisfied some constrains, see docs in
    # `_adjust_pruning_lower_bound` for more details of these constrains.
    # T1 == T here means we are using the modified version of transducer,
    # the third constrain becomes `s_begin[i + 1] - s_begin[i] < 2`, because
    # it only emits one symbol per frame.
    s_begin = _adjust_pruning_lower_bound(s_begin, 2 if T1 == T else s_range)

    ranges = s_begin.reshape((B, T, 1)).expand((B, T, s_range)) + torch.arange(
        s_range, device=px_grad.device
    )

    return ranges


# This is a deprecated version of method to generate pruning bounds which is
# less exact than the one above (i.e. the one we publish in our paper).
# It will be deleted at some time, keeping it just for testing purpose.
def get_rnnt_prune_ranges_deprecated(
    px_grad: torch.Tensor,
    py_grad: torch.Tensor,
    boundary: torch.Tensor,
    s_range: int,
) -> torch.Tensor:
    """Get the pruning ranges of normal rnnt loss according to the grads
    of px and py returned by mutual_information_recursion.

    For each sequence with T frames, we will generate a tensor with the shape of
    (T, s_range) containing the information that which symbols will be token
    into consideration for each frame. For example, here is a sequence with 10
    frames and the corresponding symbols are `[A B C D E F]`, if the s_range
    equals 3, one possible ranges tensor will be::

      [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3],
       [1, 2, 3], [1, 2, 3], [3, 4, 5], [3, 4, 5], [3, 4, 5]]

    which means we only consider `[A B C]` at frame 0, 1, 2, 3, and `[B C D]`
    at frame 4, 5, 6, `[D E F]` at frame 7, 8, 9.

    We can only consider limited number of symbols because frames and symbols
    are monotonic aligned, theoretically it can only generate particular range
    of symbols given a particular frame.

    Note:
      For the generated tensor ranges (assuming batch size is 1), ranges[:, 0]
      is a monotonic increasing tensor from 0 to `len(symbols)` and it satisfies
      `ranges[t+1, 0] - ranges[t, 0] < s_range` which means we won't skip any
      symbols.

    Args:
      px_grad:
        The gradient of px, see docs in `mutual_information_recursion` for more
        details of px.
      py_grad:
        The gradient of py, see docs in `mutual_information_recursion` for more
        details of py.
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame]
      s_range:
        How many symbols to keep for each frame.
    Returns:
      A tensor contains the kept symbols indexes for each frame, with shape
      (B, T, s_range).
    """
    (B, S, T1) = px_grad.shape
    T = py_grad.shape[-1]
    assert T1 in [T, T + 1]
    assert py_grad.shape == (B, S + 1, T)
    assert boundary.shape == (B, 4)
    assert S >= 1
    assert T >= S

    # s_range > S means we won't prune out any symbols. To make indexing with
    # ranges runs normally, s_range should be equal to or less than ``S + 1``.
    if s_range > S:
        s_range = S + 1

    if T1 == T:
        assert (
            s_range >= 1
        ), "Pruning range for modified RNN-T should be equal to or greater than 1, or no valid paths could survive pruning."

    else:
        assert (
            s_range >= 2
        ), "Pruning range for standard RNN-T should be equal to or greater than 2, or no valid paths could survive pruning."

    px_pad = torch.zeros((B, 1, T1), dtype=px_grad.dtype, device=px_grad.device)
    py_pad = torch.zeros(
        (B, S + 1, 1), dtype=py_grad.dtype, device=py_grad.device
    )
    py_grad_padded = py_grad if T1 == T else torch.cat((py_grad, py_pad), dim=2)
    tot_grad = (
        torch.cat((px_grad, px_pad), dim=1) + py_grad_padded
    )  # (B, S + 1, T1)

    tot_grad = torch.cat(
        (
            torch.zeros(
                (B, 1, T1), dtype=tot_grad.dtype, device=tot_grad.device
            ),
            tot_grad,
        ),
        dim=1,
    )
    tot_grad = torch.cumsum(tot_grad, dim=1)
    diff_grad = tot_grad[:, s_range:, :] - tot_grad[:, 0:-s_range, :]
    s_begin = torch.argmax(diff_grad, dim=1)
    s_begin = s_begin[:, :T]

    # Handle the values of s_begin in padding positions.
    # -1 here means we fill the position of the last frame (before padding) with
    # padding value which is `len(symbols) - s_range + 1`.
    # This is to guarantee that we reach the last symbol at last frame (before
    # padding).
    # The shape of the mask is (B, T), for example, we have a batch containing
    # 3 sequences, their lengths are 3, 5, 6 (i.e. B = 3, T = 6), so the mask is
    # [[True, True, False, False, False, False],
    #  [True, True, True,  True,  False, False],
    #  [True, True, True,  True,  True,  False]]
    mask = torch.arange(0, T, device=px_grad.device).reshape(1, T).expand(B, T)
    mask = mask < boundary[:, 3].reshape(B, 1) - 1

    s_begin_padding = boundary[:, 2].reshape(B, 1) - s_range + 1
    # handle the cases when `len(symbols) < s_range`
    s_begin_padding = torch.clamp(s_begin_padding, min=0)

    s_begin = torch.where(mask, s_begin, s_begin_padding)

    # adjusting lower bound to make it satisfied some constrains, see docs in
    # `_adjust_pruning_lower_bound` for more details of these constrains.
    # T1 == T here means we are using the modified version of transducer,
    # the third constrain becomes `s_begin[i + 1] - s_begin[i] < 2`, because
    # it only emits one symbol per frame.
    s_begin = _adjust_pruning_lower_bound(s_begin, 2 if T1 == T else s_range)

    ranges = s_begin.reshape((B, T, 1)).expand((B, T, s_range)) + torch.arange(
        s_range, device=px_grad.device
    )
    return ranges


def do_rnnt_pruning(
    am: torch.Tensor, lm: torch.Tensor, ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prune the output of encoder(am) output and prediction network(lm)
    output of RNNT.

    Args:
      am:
        The encoder output, with shape (B, T, encoder_dim)
      lm:
        The prediction network output, with shape (B, S + 1, decoder_dim)
      ranges:
        A tensor containing the symbol indexes for each frame that we want to
        keep. Its shape is (B, T, s_range), see the docs in
        `get_rnnt_prune_ranges` for more details of this tensor.

    Returns:
      Return the pruned am and lm with shape (B, T, s_range, C)
    """
    # am (B, T, encoder_dm)
    # lm (B, S + 1, decoder_dim)
    # ranges (B, T, s_range)
    assert ranges.shape[0] == am.shape[0]
    assert ranges.shape[0] == lm.shape[0]
    assert am.shape[1] == ranges.shape[1]
    (B, T, s_range) = ranges.shape
    (B, S1, decoder_dim) = lm.shape
    encoder_dim = am.shape[-1]
    assert am.shape == (B, T, encoder_dim)
    S = S1 - 1

    # (B, T, s_range, encoder_dim)
    am_pruned = am.unsqueeze(2).expand((B, T, s_range, encoder_dim))

    # (B, T, s_range, decoder_dim)
    lm_pruned = torch.gather(
        lm.unsqueeze(1).expand((B, T, S + 1, decoder_dim)),
        dim=2,
        index=ranges.reshape((B, T, s_range, 1)).expand(
            (B, T, s_range, decoder_dim)
        ),
    )
    return am_pruned, lm_pruned


def _roll_by_shifts(src: torch.Tensor, shifts: torch.LongTensor):
    """Roll tensor with different shifts for each row.

    Note:
      We assume the src is a 3 dimensions tensor and roll the last dimension.

    Example:

      >>> src = torch.arange(15).reshape((1,3,5))
      >>> src
      tensor([[[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]]])
      >>> shift = torch.tensor([[1, 2, 3]])
      >>> shift
      tensor([[1, 2, 3]])
      >>> _roll_by_shifts(src, shift)
      tensor([[[ 4,  0,  1,  2,  3],
               [ 8,  9,  5,  6,  7],
               [12, 13, 14, 10, 11]]])
    """
    assert src.dim() == 3
    (B, T, S) = src.shape
    assert shifts.shape == (B, T)

    index = (
        torch.arange(S, device=src.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - shifts.reshape(B, T, 1)) % S
    return torch.gather(src, 2, index)


def get_rnnt_logprobs_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Tensor,
    modified: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Construct px, py for mutual_information_recursion with pruned output.

    Args:
      logits:
        The pruned output of joiner network, with shape (B, T, s_range, C)
      symbols:
        The symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      ranges:
        A tensor containing the symbol ids for each frame that we want to keep.
        It is a LongTensor of shape ``[B][T][s_range]``, where ``ranges[b,t,0]``
        contains the begin symbol ``0 <= s <= S - s_range +1``, such that
        ``logits[b,t,:,:]`` represents the logits with positions
        ``s, s + 1, ... s + s_range - 1``.
        See docs in :func:`get_rnnt_prune_ranges` for more details of what ranges
        contains.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
        also be consumed, so at most 1 symbol can appear per frame.
    Returns:
      Return the px (B, S, T) if modified else (B, S, T + 1) and
      py (B, S + 1, T) needed by mutual_information_recursion.
    """
    # logits (B, T, s_range, C)
    # symbols (B, S)
    # ranges (B, T, s_range)
    assert logits.ndim == 4
    (B, T, s_range, C) = logits.shape
    assert ranges.shape == (B, T, s_range)
    (B, S) = symbols.shape
    assert S >= 1
    assert T >= S

    normalizers = torch.logsumexp(logits, dim=3)

    symbols_with_terminal = torch.cat(
        (
            symbols,
            torch.tensor(
                [termination_symbol] * B,
                dtype=torch.int64,
                device=symbols.device,
            ).reshape((B, 1)),
        ),
        dim=1,
    )

    # (B, T, s_range)
    pruned_symbols = torch.gather(
        symbols_with_terminal.unsqueeze(1).expand((B, T, S + 1)),
        dim=2,
        index=ranges,
    )

    # (B, T, s_range)
    px = torch.gather(
        logits, dim=3, index=pruned_symbols.reshape(B, T, s_range, 1)
    ).squeeze(-1)
    px = px - normalizers

    # (B, T, S) with index larger than s_range in dim 2 fill with -inf
    px = torch.cat(
        (
            px,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=px.device,
                dtype=px.dtype,
            ),
        ),
        dim=2,
    )

    # (B, T, S) with index out of s_range in dim 2 fill with -inf
    px = _roll_by_shifts(px, ranges[:, :, 0])[:, :, :S]

    px = px.permute((0, 2, 1))

    if not modified:
        px = torch.cat(
            (
                px,
                torch.full(
                    (B, S, 1), float("-inf"), device=px.device, dtype=px.dtype
                ),
            ),
            dim=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    py = logits[:, :, :, termination_symbol].clone()  # (B, T, s_range)
    py = py - normalizers

    # (B, T, S + 1) with index larger than s_range in dim 2 filled with -inf
    py = torch.cat(
        (
            py,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=py.device,
                dtype=py.dtype,
            ),
        ),
        dim=2,
    )

    # (B, T, S + 1) with index out of s_range in dim 2 fill with -inf
    py = _roll_by_shifts(py, ranges[:, :, 0])
    # (B, S + 1, T)
    py = py.permute((0, 2, 1))

    if not modified:
        px = fix_for_boundary(px, boundary)

    return (px, py)


def rnnt_loss_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Tensor = None,
    modified: bool = False,
    reduction: Optional[str] = "mean",
) -> Tensor:
    """A RNN-T loss with pruning, which uses a pruned 'joiner' network output
    as input, i.e. a 4 dimensions tensor with shape (B, T, s_range, C),
    s_range means the symbols number kept for each frame.

    Args:
      logits:
        The pruned output of joiner network, with shape (B, T, s_range, C),
        i.e. batch, time_seq_len, prune_range, num_classes
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      ranges:
        A tensor containing the symbol ids for each frame that we want to keep.
        It is a LongTensor of shape ``[B][T][s_range]``, where ``ranges[b,t,0]``
        contains the begin symbol ``0 <= s <= S - s_range +1``, such that
        ``logits[b,t,:,:]`` represents the logits with positions
        ``s, s + 1, ... s + s_range - 1``.
        See docs in :func:`get_rnnt_prune_ranges` for more details of what ranges
        contains.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T] if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
        also be consumed, so at most 1 symbol can appear per frame.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
    Returns:
      If recursion is `none`, returns a tensor of shape (B,), containing the
      total RNN-T loss values for each element of the batch, otherwise a scalar
      with the reduction applied.
    """
    px, py = get_rnnt_logprobs_pruned(
        logits=logits,
        symbols=symbols,
        ranges=ranges,
        termination_symbol=termination_symbol,
        boundary=boundary,
        modified=modified,
    )
    negated_loss = mutual_information_recursion(px=px, py=py, boundary=boundary)
    if reduction == "none":
        return -negated_loss
    elif reduction == "mean":
        return -torch.mean(negated_loss)
    elif reduction == "sum":
        return -torch.sum(negated_loss)
    else:
        assert (
            False
        ), f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"


def get_rnnt_logprobs_smoothed(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just
    addition), to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This version allows you to make the loss-function one of the form::

          lm_only_scale * lm_probs +
          am_only_scale * am_probs +
          (1-lm_only_scale-am_only_scale) * combined_probs

    where lm_probs and am_probs are the probabilities given the lm and acoustic
    model independently.

    This function is called from
    :func:`rnnt_loss_smoothed`, but may be useful for other purposes.

    Args:
      lm:
        Language model part of un-normalized logprobs of symbols, to be added to
        acoustic model part before normalizing.  Of shape::

           [B][S+1][C]

        where B is the batch size, S is the maximum sequence length of
        the symbol sequence, possibly including the EOS symbol; and
        C is size of the symbol vocabulary, including the termination/next-frame
        symbol.
        Conceptually, lm[b][s] is a vector of length [C] representing the
        "language model" part of the un-normalized logprobs of symbols,
        given all symbols *earlier than* s in the sequence.  The reason
        we still need this for position S is that we may still be emitting
        the termination/next-frame symbol at this point.
      am:
        Acoustic-model part of un-normalized logprobs of symbols, to be added
        to language-model part before normalizing.  Of shape::

           [B][T][C]

        where B is the batch size, T is the maximum sequence length of
        the acoustic sequences (in frames); and C is size of the symbol
        vocabulary, including the termination/next-frame symbol.  It reflects
        the "acoustic" part of the probability of any given symbol appearing
        next on this frame.
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      lm_only_scale:
        the scale on the "LM-only" part of the loss.
      am_only_scale:
        the scale on the "AM-only" part of the loss, for which we use
        an "averaged" LM (averaged over all histories, so effectively unigram).
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
        also be consumed, so at most 1 symbol can appear per frame.
    Returns:
        (px, py) (the names are quite arbitrary).
           px: logprobs, of shape [B][S][T+1] if !modified, [B][S][T] if modified.
           py: logprobs, of shape [B][S+1][T]

        in the recursion::

          p[b,0,0] = 0.0
          if !modified:
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
          if modified:
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])
          .. where p[b][s][t] is the "joint score" of the pair of subsequences
          of length s and t respectively.  px[b][s][t] represents the
          probability of extending the subsequences of length (s,t) by one in
          the s direction, given the particular symbol, and py[b][s][t]
          represents the probability of extending the subsequences of length
          (s,t) by one in the t direction,
          i.e. of emitting the termination/next-frame symbol.

          px[:,:,T] equals -infinity, meaning on the "one-past-the-last" frame
          we cannot emit any symbols.  This is simply a way of incorporating
          the probability of the termination symbol on the last frame.
    """
    assert lm.ndim == 3
    assert am.ndim == 3
    assert lm.shape[0] == am.shape[0]
    assert lm.shape[2] == am.shape[2]
    (B, T, C) = am.shape
    S = lm.shape[1] - 1
    assert symbols.shape == (B, S)
    assert S >= 1
    assert T >= S

    # Caution: some parts of this code are a little less clear than they could
    # be due to optimizations.  In particular it may not be totally obvious that
    # all of the logprobs here are properly normalized.  We test that
    # this code is invariant to adding constants in the appropriate ways.

    # subtracting am_max and lm_max is to ensure the probs are in a good range
    # to do exp() without causing underflow or overflow.
    am_max, _ = torch.max(am, dim=2, keepdim=True)  # am_max: [B][T][1]
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)  # lm_max: [B][S+1][1]
    am_probs = (am - am_max).exp()  # [B][T][C]
    lm_probs = (lm - lm_max).exp()  # [B][S+1][C]
    # normalizers: [B][S+1][T]
    normalizers = (
        torch.matmul(lm_probs, am_probs.transpose(1, 2))
        + torch.finfo(lm_probs.dtype).tiny
    ).log()

    # normalizer per frame, if we take only the LM probs by themselves
    lmonly_normalizers = lm_probs.sum(
        dim=2, keepdim=True
    )  # lmonly_normalizers: [B][S+1][1]
    unigram_lm = (
        torch.mean(lm_probs / lmonly_normalizers, dim=(0, 1), keepdim=True)
        + torch.finfo(lm_probs.dtype).tiny
    )  # [1][1][C]
    amonly_normalizers = (
        torch.mv(am_probs.reshape(-1, C), unigram_lm.reshape(C))
        .reshape(B, T, 1)
        .log()
        + am_max
    )  # [B][T][1]
    amonly_normalizers = amonly_normalizers.transpose(1, 2)  # [B][1][T]
    unigram_lm = unigram_lm.log()
    lmonly_normalizers = (
        lmonly_normalizers.log() + lm_max
    )  # [B][S+1][1], log-normalizer, used for LM-only part of prob.

    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)  # [B][S+1][T]

    # px is the probs of the actual symbols (not yet normalized)..
    px_am = torch.gather(
        am.unsqueeze(1).expand(B, S, T, C),
        dim=3,
        index=symbols.reshape(B, S, 1, 1).expand(B, S, T, 1),
    ).squeeze(
        -1
    )  # [B][S][T]

    if not modified:
        px_am = torch.cat(
            (
                px_am,
                torch.full(
                    (B, S, 1),
                    float("-inf"),
                    device=px_am.device,
                    dtype=px_am.dtype,
                ),
            ),
            dim=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    px_lm = torch.gather(
        lm[:, :S], dim=2, index=symbols.unsqueeze(-1)
    )  # [B][S][1]
    px_lm_unigram = torch.gather(
        unigram_lm.expand(B, S, C), dim=2, index=symbols.unsqueeze(-1)
    )  # [B][S][1]

    px = px_am + px_lm  # [B][S][T+1] if not modified, [B][S][T] if modified
    px[:, :, :T] -= normalizers[:, :S, :]  # px: [B][S][T+1] or [B][S][T]

    px_amonly = (
        px_am + px_lm_unigram
    )  # [B][S][T+1] if !modified; [B][S][T] if modified.
    px_amonly[:, :, :T] -= amonly_normalizers
    px_lmonly = px_lm - lmonly_normalizers[:, :S, :]

    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = am[:, :, termination_symbol].unsqueeze(1)  # [B][1][T]
    py_lm = lm[:, :, termination_symbol].unsqueeze(2)  # [B][S+1][1]
    py = py_am + py_lm - normalizers

    py_lm_unigram = unigram_lm[0][0][termination_symbol]  # scalar, normalized..
    py_amonly = py_am + py_lm_unigram - amonly_normalizers  # [B][S+1][T]
    py_lmonly = py_lm - lmonly_normalizers  # [B][S+1][T]

    combined_scale = 1.0 - lm_only_scale - am_only_scale

    # We need to avoid exact zeros in the scales because otherwise multiplying
    # -inf by zero generates nan.
    if lm_only_scale == 0.0:
        lm_only_scale = 1.0e-20
    if am_only_scale == 0.0:
        am_only_scale = 1.0e-20

    px_interp = (
        px * combined_scale
        + px_lmonly * lm_only_scale
        + px_amonly * am_only_scale
    )
    py_interp = (
        py * combined_scale
        + py_lmonly * lm_only_scale
        + py_amonly * am_only_scale
    )

    if not modified:
        px_interp = fix_for_boundary(px_interp, boundary)

    return (px_interp, py_interp)


def rnnt_loss_smoothed(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[Tensor] = None,
    modified: bool = False,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tensor]:
    """A simple case of the RNN-T loss, where the 'joiner' network is just
    addition.

    Args:
      lm:
        language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes.
        These are assumed to be well-normalized, in the sense that we could
        use them as probabilities separately from the am scores
      am:
        acoustic-model part of unnormalized log-probs of symbols, with shape
        (B, T, C), i.e. batch, frame, num_classes
      symbols:
        the symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      lm_only_scale:
        the scale on the "LM-only" part of the loss.
      am_only_scale:
        the scale on the "AM-only" part of the loss, for which we use
        an "averaged" LM (averaged over all histories, so effectively unigram).
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
        also be consumed, so at most 1 symbol can appear per frame.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
      return_grad:
        Whether to return grads of px and py, this grad standing for the
        occupation probability is the output of the backward with a
        `fake gradient`, the `fake gradient` is the same as the gradient you'd
        get if you did `torch.autograd.grad((-loss.sum()), [px, py])`, note, the
        loss here is the loss with reduction "none".
        This is useful to implement the pruned version of rnnt loss.

    Returns:
       If return_grad is False, returns a tensor of shape (B,), containing the
       total RNN-T loss values for each element of the batch if reduction equals
       to "none", otherwise a scalar with the reduction applied.
       If return_grad is True, the grads of px and py, which is the output of
       backward with a `fake gradient`(see above), will be returned too. And the
       returned value will be a tuple like (loss, (px_grad, py_grad)).
    """
    px, py = get_rnnt_logprobs_smoothed(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        lm_only_scale=lm_only_scale,
        am_only_scale=am_only_scale,
        boundary=boundary,
        modified=modified,
    )
    scores_and_grads = mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -torch.mean(negated_loss)
    elif reduction == "sum":
        loss = -torch.sum(negated_loss)
    else:
        assert (
            False
        ), f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
    return (loss, scores_and_grads[1]) if return_grad else loss
