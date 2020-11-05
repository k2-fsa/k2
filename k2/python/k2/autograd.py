# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

from typing import List, Tuple

import torch

import _k2

from .fsa import Fsa
from .dense_fsa_vec import DenseFsaVec


class _GetTotScoresFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fsas: Fsa, log_semiring: bool, use_float_scores: bool,
                unused_scores: torch.Tensor) -> torch.Tensor:
        '''Compute the total loglikes of an FsaVec.

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

        # NOTE: since `fsas`, `log_semiring` and `use_float_scores` are
        # not tensors, they are saved as attributes of `ctx`.
        ctx.fsas = fsas
        ctx.log_semiring = log_semiring
        ctx.use_float_scores = use_float_scores

        ctx.save_for_backward(unused_scores)

        return tot_scores

    @staticmethod
    def backward(ctx, unused: torch.Tensor
                ) -> Tuple[None, None, None, torch.Tensor]:  # noqa
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
            #      fsas, log_semiring, use_float_scores, unused_scores
            return None, None, None, out_grad
        else:
            forward_scores = fsas.update_forward_scores_log(use_float_scores)
            backward_scores = fsas.update_backward_scores_log(use_float_scores)
            if use_float_scores:
                func = _k2._get_arc_scores_float
            else:
                func = _k2._get_arc_scores_double
            arc_scores = func(fsas=fsas.arcs,
                              forward_scores=forward_scores,
                              backward_scores=backward_scores)
            return None, None, None, arc_scores.exp()


class _IntersectDensePrunedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_fsas: Fsa, b_fsas: DenseFsaVec, out_fsa: List[Fsa],
                beam: float, max_active_states: int, min_active_states: int,
                unused_scores_a: torch.Tensor,
                unused_scores_b: torch.Tensor) -> torch.Tensor:
        '''Intersect array of FSAs on CPU/GPU.

        Args:
          a_fsas:
            Input FsaVec, i.e., `decoding graphs`, one per sequence. It might
            just be a linear sequence of phones, or might be something more
            complicated. Must have either `a_fsas.shape[0] == b_fsas.dim0()`, or
            `a_fsas.shape[0] == 1` in which case the graph is shared.
          b_fsas:
            Input FSAs that correspond to neural network output.
          out_fsa:
            A list containing ONLY one entry which will be set to the
            generated FSA on return. We pass it as a list since the return
            value can only be types of torch.Tensor in the `forward` function.
          beam:
            Decoding beam, e.g. 10.  Smaller is faster, larger is more exact
            (less pruning). This is the default value; it may be modified by
            `min_active_states` and `max_active_states`.
          max_active_states:
            Maximum number of FSA states that are allowed to be active on any
            given frame for any given intersection/composition task. This is
            advisory, in that it will try not to exceed that but may not always
            succeed. You can use a very large number if no constraint is needed.
          min_active_states:
            Minimum number of FSA states that are allowed to be active on any
            given frame for any given intersection/composition task. This is
            advisory, in that it will try not to have fewer than this number
            active. Set it to zero if there is no constraint.
          unused_scores_a:
            It equals to `a_fsas.scores` and its sole purpose is for back
            propagation.
          unused_scores_b:
            It equals to `b_fsas.scores` and its sole purpose is for back
            propagation.
        Returns:
           Return `out_fsa[0].scores`.
        '''
        assert len(out_fsa) == 1

        ragged_arc, arc_map_a, arc_map_b = _k2.intersect_dense_pruned(
            a_fsas=a_fsas.arcs,
            b_fsas=b_fsas.dense_fsa_vec,
            beam=beam,
            max_active_states=max_active_states,
            min_active_states=min_active_states)

        # required by `index_select_` and `index_add_` (in backward)
        arc_map_a = arc_map_a.to(torch.int64)
        arc_map_b = arc_map_b.to(torch.int64)

        out_fsa[0] = Fsa.from_ragged_arc(ragged_arc)

        for name, a_value in a_fsas.named_tensor_attr():
            if name == 'scores':
                continue
            value = a_value.index_select(0, arc_map_a)
            setattr(out_fsa[0], name, value)

        for name, a_value in a_fsas.named_non_tensor_attr():
            setattr(out_fsa[0], name, a_value)

        if hasattr(out_fsa[0], 'properties'):
            del out_fsa[0].properties

        ctx.arc_map_a = arc_map_a
        ctx.arc_map_b = arc_map_b

        ctx.save_for_backward(unused_scores_a, unused_scores_b)

        return out_fsa[0].scores

    @staticmethod
    def backward(
            ctx, out_fsa_grad
    ) -> Tuple[None, None, None, None, None, None, torch.Tensor, torch.Tensor]:
        arc_map_a = ctx.arc_map_a
        arc_map_b = ctx.arc_map_b

        a_scores, b_scores = ctx.saved_tensors

        grad_a = torch.zeros(a_scores.size(0),
                             dtype=torch.float32,
                             device=a_scores.device,
                             requires_grad=False)

        grad_b = torch.zeros(
            *b_scores.shape,
            dtype=torch.float32,
            device=b_scores.device,
            requires_grad=False).contiguous()  # will use its `view()` later

        grad_a.index_add_(0, arc_map_a, out_fsa_grad)
        grad_b.view(-1).index_add_(0, arc_map_b, out_fsa_grad)

        return None, None, None, None, None, None, grad_a, grad_b


def get_tot_scores(fsas: Fsa, log_semiring: bool,
                   use_float_scores: bool) -> torch.Tensor:
    '''Compute the total loglikes of an FsaVec.
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
    tot_scores = _GetTotScoresFunction.apply(fsas, log_semiring,
                                             use_float_scores, fsas.scores)
    return tot_scores


def intersect_dense_pruned(a_fsas: Fsa, b_fsas: DenseFsaVec, beam: float,
                           max_active_states: int,
                           min_active_states: int) -> Fsa:
    '''Intersect array of FSAs on CPU/GPU.

    Caution:
      `a_fsas` MUST be arc sorted.

    Args:
      a_fsas:
        Input FsaVec, i.e., `decoding graphs`, one per sequence. It might just
        be a linear sequence of phones, or might be something more complicated.
        Must have either `a_fsas.shape[0] == b_fsas.dim0()`, or
        `a_fsas.shape[0] == 1` in which case the graph is shared.
      b_fsas:
        Input FSAs that correspond to neural network output.
      beam:
        Decoding beam, e.g. 10.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.

    Returns:
      The result of the intersection.
    '''
    out_fsa = [0]

    # the following return value is discarded since it is already contained
    # in `out_fsa[0].scores`
    _IntersectDensePrunedFunction.apply(a_fsas, b_fsas, out_fsa, beam,
                                        max_active_states, min_active_states,
                                        a_fsas.scores, b_fsas.scores)
    return out_fsa[0]
