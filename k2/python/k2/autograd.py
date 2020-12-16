# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

from typing import List, Tuple

import torch
import _k2

from .fsa import Fsa
from .ops import index_attr
from .dense_fsa_vec import DenseFsaVec


class _GetTotScoresFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fsas: Fsa, log_semiring: bool, use_double_scores: bool,
                unused_scores: torch.Tensor) -> torch.Tensor:
        '''Compute the total loglikes of an FsaVec.

        Args:
          fsas:
            The input FsaVec.
          log_semiring:
            True to use log semiring.
            False to use tropical semiring.
          use_double_scores:
            False to use float, i.e., single precision floating point,
            to compute log likes. True to use double precision.
          unused_scores:
            It is used only for backward propagation purpose.
            It equals to `fsas.scores`.

        Returns:
          The forward loglike contained in a 1-D tensor.
          If `use_double_scores==True`, its dtype is `torch.float64`;
          it is `torch.float32` otherwise.
        '''
        # the .detach() below avoids a reference cycle; if we didn't do that,
        # the backward_fn of tot_scores would be set to this object, giving
        # `fsas` a reference to this object, which also has a reference
        # to `fsas`.
        if log_semiring is False:
            tot_scores = fsas.get_tot_scores_tropical(
                use_double_scores).detach()
        else:
            tot_scores = fsas.get_tot_scores_log(use_double_scores).detach()

        # NOTE: since `fsas`, `log_semiring` and `use_double_scores` are
        # not tensors, they are saved as attributes of `ctx`.
        ctx.fsas = fsas
        ctx.log_semiring = log_semiring
        ctx.use_double_scores = use_double_scores

        ctx.save_for_backward(unused_scores)

        return tot_scores

    @staticmethod
    def backward(ctx, tot_scores_grad: torch.Tensor
                ) -> Tuple[None, None, None, torch.Tensor]:  # noqa
        fsas = ctx.fsas
        log_semiring = ctx.log_semiring
        use_double_scores = ctx.use_double_scores
        scores, = ctx.saved_tensors

        if log_semiring is False:
            entering_arcs = fsas.get_entering_arcs(use_double_scores)
            _, ragged_int = _k2.shortest_path(fsas.arcs, entering_arcs)
            if use_double_scores:
                out_grad = _k2._get_tot_scores_double_tropical_backward(
                    fsas.arcs, ragged_int, tot_scores_grad)
            else:
                out_grad = _k2._get_tot_scores_float_tropical_backward(
                    fsas.arcs, ragged_int, tot_scores_grad)
            # We return four values since the `forward` method accepts four
            # arguments (excluding ctx).
            #      fsas, log_semiring, use_double_scores, unused_scores
            return None, None, None, out_grad
        else:
            forward_scores = fsas.get_forward_scores_log(use_double_scores)
            backward_scores = fsas.get_backward_scores_log(use_double_scores)
            if use_double_scores:
                func = _k2._get_arc_scores_double
                bprop_func = _k2._get_tot_scores_double_log_backward
            else:
                func = _k2._get_arc_scores_float
                bprop_func = _k2._get_tot_scores_float_log_backward

            arc_scores = func(fsas=fsas.arcs,
                              forward_scores=forward_scores,
                              backward_scores=backward_scores)
            out_grad = bprop_func(fsas.arcs, arc_scores, tot_scores_grad)
            return None, None, None, out_grad


class _IntersectDensePrunedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_fsas: Fsa, b_fsas: DenseFsaVec, out_fsa: List[Fsa],
                search_beam: float, output_beam: float, min_active_states: int,
                max_active_states: int, unused_scores_a: torch.Tensor,
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
          search_beam:
            Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
            (less pruning). This is the default value; it may be modified by
            `min_active_states` and `max_active_states`.
          output_beam:
            Pruning beam for the output of intersection (vs. best path);
            equivalent to kaldi's lattice-beam.  E.g. 8.
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
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states)

        out_fsa[0] = Fsa(ragged_arc)

        for name, a_value in a_fsas.named_tensor_attr(include_scores=False):
            value = index_attr(a_value, arc_map_a)
            setattr(out_fsa[0], name, value)

        for name, a_value in a_fsas.named_non_tensor_attr():
            setattr(out_fsa[0], name, a_value)

        ctx.arc_map_a = arc_map_a
        ctx.arc_map_b = arc_map_b

        ctx.save_for_backward(unused_scores_a, unused_scores_b)

        return out_fsa[0].scores

    @staticmethod
    def backward(ctx, out_fsa_grad: torch.Tensor) \
            -> Tuple[None, None, None, None, None, None, None, torch.Tensor, torch.Tensor]: # noqa
        a_scores, b_scores = ctx.saved_tensors
        arc_map_a = ctx.arc_map_a
        arc_map_b = ctx.arc_map_b

        grad_a = torch.zeros(a_scores.size(0),
                             dtype=torch.float32,
                             device=a_scores.device,
                             requires_grad=False)

        grad_b = torch.zeros(
            *b_scores.shape,
            dtype=torch.float32,
            device=b_scores.device,
            requires_grad=False).contiguous()  # will use its `view()` later

        _k2.index_add(arc_map_a, out_fsa_grad, grad_a)
        _k2.index_add(arc_map_b, out_fsa_grad, grad_b.view(-1))

        return None, None, None, None, None, None, None, grad_a, grad_b


class _IntersectDenseFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_fsas: Fsa, b_fsas: DenseFsaVec, out_fsa: List[Fsa],
                output_beam: float, unused_scores_a: torch.Tensor,
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
          search_beam:
            Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
            (less pruning). This is the default value; it may be modified by
            `min_active_states` and `max_active_states`.
          output_beam:
            Pruning beam for the output of intersection (vs. best path);
            equivalent to kaldi's lattice-beam.  E.g. 8.
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

        ragged_arc, arc_map_a, arc_map_b = _k2.intersect_dense(
            a_fsas=a_fsas.arcs,
            b_fsas=b_fsas.dense_fsa_vec,
            output_beam=output_beam)

        out_fsa[0] = Fsa(ragged_arc)

        for name, a_value in a_fsas.named_tensor_attr(include_scores=False):
            value = index_attr(a_value, arc_map_a)
            setattr(out_fsa[0], name, value)

        for name, a_value in a_fsas.named_non_tensor_attr():
            setattr(out_fsa[0], name, a_value)

        ctx.arc_map_a = arc_map_a
        ctx.arc_map_b = arc_map_b

        ctx.save_for_backward(unused_scores_a, unused_scores_b)

        return out_fsa[0].scores

    @staticmethod
    def backward(ctx, out_fsa_grad: torch.Tensor) \
            -> Tuple[None, None, None, None, torch.Tensor, torch.Tensor]: # noqa
        a_scores, b_scores = ctx.saved_tensors
        arc_map_a = ctx.arc_map_a
        arc_map_b = ctx.arc_map_b

        grad_a = torch.zeros(a_scores.size(0),
                             dtype=torch.float32,
                             device=a_scores.device,
                             requires_grad=False)

        grad_b = torch.zeros(
            *b_scores.shape,
            dtype=torch.float32,
            device=b_scores.device,
            requires_grad=False).contiguous()  # will use its `view()` later

        _k2.index_add(arc_map_a, out_fsa_grad, grad_a)
        _k2.index_add(arc_map_b, out_fsa_grad, grad_b.view(-1))

        return None, None, None, None, grad_a, grad_b


class _UnionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fsas: Fsa, out_fsa: List[Fsa],
                unused_fsas_scores: torch.Tensor) -> torch.Tensor:
        '''Compute the union of all fsas in a FsaVec.

        Args:
          fsas:
            The input FsaVec. Caution: We require that each fsa in the FsaVec
            is non-empty (i.e., with at least two states).
          out_fsa:
            A list containing one entry. Since this function can only return
            values of type `torch.Tensor`, we return the union result in the
            list.
          unused_fsas_scores:
            It is the same as `fsas.scores`, whose sole purpose is for autograd.
            It is not used in this function.
        '''
        need_arc_map = True
        ragged_arc, arc_map = _k2.union(fsas.arcs, need_arc_map)
        out_fsa[0] = Fsa(ragged_arc)

        for name, value in fsas.named_tensor_attr(include_scores=False):
            value = index_attr(value, arc_map)
            setattr(out_fsa[0], name, value)

        for name, value in fsas.named_non_tensor_attr():
            setattr(out_fsa[0], name, value)
        ctx.arc_map = arc_map
        ctx.save_for_backward(unused_fsas_scores)

        return out_fsa[0].scores  # the return value will be discarded

    @staticmethod
    def backward(ctx, out_fsa_grad: torch.Tensor
                ) -> Tuple[None, None, torch.Tensor]:  # noqa
        arc_map = ctx.arc_map
        fsas_scores, = ctx.saved_tensors
        ans = torch.zeros(fsas_scores.size(0),
                          dtype=torch.float32,
                          device=fsas_scores.device,
                          requires_grad=False)
        _k2.index_add(arc_map, out_fsa_grad, ans)
        return None, None, ans


def get_tot_scores(fsas: Fsa, log_semiring: bool,
                   use_double_scores: bool) -> torch.Tensor:
    '''Compute the total loglikes of an FsaVec.

    Args:
      fsas:
        The input FsaVec.
      log_semiring:
        True to use log semiring.
        False to use tropical semiring.
      use_double_scores:
        False to use float, i.e., single precision floating point,
        to compute log likes. True to use double precision.

    Returns:
      The forward loglike contained in a 1-D tensor.
      If `use_double_scores==True`, its dtype is `torch.float64`;
      it is `torch.float32` otherwise.
    '''
    tot_scores = _GetTotScoresFunction.apply(fsas, log_semiring,
                                             use_double_scores, fsas.scores)
    return tot_scores


def intersect_dense_pruned(a_fsas: Fsa, b_fsas: DenseFsaVec,
                           search_beam: float, output_beam: float,
                           min_active_states: int,
                           max_active_states: int) -> Fsa:
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
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.

    Returns:
      The result of the intersection.
    '''
    out_fsa = [0]

    # the following return value is discarded since it is already contained
    # in `out_fsa[0].scores`
    _IntersectDensePrunedFunction.apply(a_fsas, b_fsas, out_fsa, search_beam,
                                        output_beam, min_active_states,
                                        max_active_states, a_fsas.scores,
                                        b_fsas.scores)
    return out_fsa[0]


def intersect_dense(a_fsas: Fsa, b_fsas: DenseFsaVec,
                    output_beam: float) -> Fsa:
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
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.

    Returns:
      The result of the intersection (pruned to `output_beam`; this pruning
      is exact, it uses forward and backward scores.
    '''
    out_fsa = [0]

    # the following return value is discarded since it is already contained
    # in `out_fsa[0].scores`
    _IntersectDenseFunction.apply(a_fsas, b_fsas, out_fsa, output_beam,
                                  a_fsas.scores, b_fsas.scores)
    return out_fsa[0]


def union(fsas: Fsa) -> Fsa:
    '''Compute the union of a FsaVec.

    Caution:
      We require that every fsa in fsas is non-empty, i.e.,
      contains at least two states

    Args:
      fsas:
        A FsaVec. That is, len(fsas.shape) == 3.

    Returns:
      A single Fsa that is the union of the input fsas.
    '''

    out_fsa = [0]  # as a placeholder
    _UnionFunction.apply(fsas, out_fsa, fsas.scores)
    return out_fsa[0]
