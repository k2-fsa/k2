# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

import torch

import _k2

from .fsa import Fsa


class _ForwardLogLikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fsas: Fsa, log_semiring: bool, use_float_scores: bool,
                unused_scores: torch.Tensor) -> torch.Tensor:
        '''Compute the forward loglikes of an FsaVec.

        Args:
          fsas:
            The input FsaVec.
          log_semiring:
            True to use log semiring.
            False to use tropical semiring.
          use_float_scores:
            True to use float, i.e., single precision floating point,
            to compute log likes. False to use double precision.
          unused_scores:
            It is used only for backward propagation purpose.
            It equals to `fsas.scores`.

        Returns:
          The forward loglike contained in a 1-D tensor.
          If `use_float_scores==True`, its dtype is `torch.float32`;
          it is `torch.float64` otherwise.
        '''
        if log_semiring is False:
            tot_scores = fsas.update_tot_scores_tropical(use_float_scores)
        else:
            tot_scores = fsas.update_tot_scores_log(use_float_scores)
            #  forward_scores = fsas.update_forward_scores_log(use_float_scores)
            #  backward_scores = fsas.update_backward_scores_log(use_float_scores)
        ctx.fsas = fsas
        ctx.log_semiring = log_semiring
        ctx.use_float_scores = use_float_scores
        ctx.save_for_backward(unused_scores)
        return tot_scores

    @staticmethod
    def backward(ctx, unused):
        fsas = ctx.fsas
        log_semiring = ctx.log_semiring
        use_float_scores = ctx.use_float_scores
        scores, = ctx.saved_tensors
        if log_semiring is False:
            entering_arcs = fsas.update_entering_arcs(use_float_scores)
            _, ragged_int = _k2.shortest_path(fsas.arcs, entering_arcs)
            best_path_arc_indexes = ragged_int.values().to(torch.int64)
            out_grad = torch.zeros_like(scores, requires_grad=False)
            out_grad[best_path_arc_indexes] = 1
            # We return four values since the `forward` method accepts four
            # arguments (excluding ctx).
            #      fsas, log_semiring, use_float_scores
            return None, None, None, out_grad


def get_forward_log_like(fsas: Fsa, log_semiring: bool,
                         use_float_scores: bool) -> torch.Tensor:
    '''Compute the forward loglikes of an FsaVec.
    Args:
      fsas:
        The input FsaVec.
      log_semiring:
        True to use log semiring.
        False to use tropical semiring.
      use_float_scores:
        True to use float, i.e., single precision floating point,
        to compute log likes. False to use double precision.
    Returns:
      The forward loglike contained in a 1-D tensor.
      If `use_float_scores==True`, its dtype is `torch.float32`;
      it is `torch.float64` otherwise.
    '''
    tot_scores = _ForwardLogLikeFunction.apply(fsas, log_semiring,
                                               use_float_scores, fsas.scores)
    return tot_scores
