# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu, Wei Kang)
#                2021  Mobvoi Inc.        (authors: Yaguang Hu)
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

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import _k2
import k2

from . import fsa_properties
from .fsa import Fsa
from .ops import index
from .ops import index_select
from .ops import index_ragged

from torch import Tensor

# Note: look also in autograd.py, differentiable operations may be there.


def linear_fsa(labels: Union[List[int], List[List[int]], k2.RaggedInt],
               device: Optional[Union[torch.device, str]] = None) -> Fsa:
    '''Construct an linear FSA from labels.

    Note:
      The scores of arcs in the returned FSA are all 0.

    Args:
      labels:
        It can be one of the following types:

            - A list of integers, e.g., `[1, 2, 3]`
            - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
            - An instance of :class:`k2.RaggedInt`. Must have `num_axes() == 2`.
      device:
        Optional. It can be either a string (e.g., 'cpu',
        'cuda:0') or a torch.device.
        If it is None, then the returned FSA is on CPU. It has to be None
        if `labels` is an instance of :class:`k2.RaggedInt`.

    Returns:
      - If `labels` is a list of integers, return an FSA
      - If `labels` is a list of list-of-integers, return an FsaVec
      - If `labels` is an instance of :class:`k2.RaggedInt`, return an FsaVec
    '''
    if isinstance(labels, k2.RaggedInt):
        assert device is None
        assert labels.num_axes() == 2

    if device is not None:
        device = torch.device(device)
        if device.type == 'cpu':
            gpu_id = -1
        else:
            assert device.type == 'cuda'
            gpu_id = getattr(device, 'index', 0)
    else:
        gpu_id = -1
    ragged_arc = _k2.linear_fsa(labels, gpu_id)
    fsa = Fsa(ragged_arc)
    return fsa


def linear_fst(labels: Union[List[int], List[List[int]]],
               aux_labels: Union[List[int], List[List[int]]]) -> Fsa:
    '''Construct a linear FST from labels and its corresponding
    auxiliary labels.

    Note:
      The scores of arcs in the returned FST are all 0.

    Args:
      labels:
        A list of integers or a list of list of integers.
      aux_labels:
        A list of integers or a list of list of integers.

    Returns:
      An FST if the labels is a list of integers.
      A vector of FSTs (FsaVec) if the input is a list of list of integers.
    '''
    ragged_arc = _k2.linear_fsa(labels)
    if isinstance(labels[0], List):
        assert isinstance(aux_labels[0],
                          list), 'aux_labels and labels do not match.'
        buf = []
        for aux in aux_labels:
            buf.extend(aux + [-1])
        aux_labels_tmp = torch.tensor(buf, dtype=torch.int32)
    else:
        aux_labels_tmp = torch.tensor(aux_labels + [-1], dtype=torch.int32)
    fsa = Fsa(ragged_arc, aux_labels=aux_labels_tmp)
    return fsa


def top_sort(fsa: Fsa) -> Fsa:
    '''Sort an FSA topologically.

    Note:
      It returns a new FSA. The input FSA is NOT changed.

    Args:
      fsa:
        The input FSA to be sorted. It can be either a single FSA
        or a vector of FSAs.
    Returns:
      It returns a single FSA if the input is a single FSA; it returns
      a vector of FSAs if the input is a vector of FSAs.
    '''
    need_arc_map = True
    ragged_arc, arc_map = _k2.top_sort(fsa.arcs, need_arc_map=need_arc_map)

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    return out_fsa


def intersect_device(
        a_fsas: Fsa,
        b_fsas: Fsa,
        b_to_a_map: torch.Tensor,
        sorted_match_a: bool = False,
        ret_arc_maps: bool = False
) -> Union[Fsa, Tuple[Fsa, torch.Tensor, torch.Tensor]]:  # noqa
    '''Compute the intersection of two FsaVecs treating epsilons
    as real, normal symbols.

    This function supports both CPU and GPU. But it is very slow on CPU.
    That's why this function name ends with `_device`. It is intended for GPU.
    See :func:`k2.intersect` which is a more general interface
    (it will call the same underlying code, IntersectDevice(), if
    the inputs are on GPU and a_fsas is arc-sorted).

    Caution:
      Epsilons are treated as real, normal symbols.

    Hint:
      The two inputs do not need to be arc-sorted.

    Refer to :func:`k2.intersect` for how we assign the attributes of the
    output FsaVec.

    Args:
      a_fsas:
        An FsaVec (must have 3 axes, i.e., `len(a_fsas.shape) == 3`.
      b_fsas:
        An FsaVec (must have 3 axes) on the same device as `a_fsas`.
      b_to_a_map:
        A 1-D torch.Tensor with dtype torch.int32 on the same device
        as `a_fsas`. Map from FSA-id in `b_fsas` to the corresponding
        FSA-id in `a_fsas` that we want to compose it with.
        E.g. might be an identity map, or all-to-zero, or something the
        user chooses.

        Requires
            - `b_to_a_map.shape[0] == b_fsas.shape[0]`
            - `0 <= b_to_a_map[i] < a_fsas.shape[0]`
      ret_arc_maps:
        If False, return the resulting Fsa. If True, return a tuple
        containing three entries:

            - the resulting Fsa

            - a_arc_map, a 1-D torch.Tensor with dtype torch.int32.
              a_arc_map[i] is the arc index in a_fsas that corresponds
              to the i-th arc in the resulting Fsa. a_arc_map[i] is -1
              if the i-th arc in the resulting Fsa has no corresponding
              arc in a_fsas.

            - b_arc_map, a 1-D torch.Tensor with dtype torch.int32.
              b_arc_map[i] is the arc index in b_fsas that corresponds
              to the i-th arc in the resulting Fsa. b_arc_map[i] is -1
              if the i-th arc in the resulting Fsa has no corresponding
              arc in b_fsas.

    Returns:
      If ret_arc_maps is False, return intersected FsaVec;
      will satisfy `ans.shape == b_fsas.shape`.
      If ret_arc_maps is True, it returns additionally two arc maps:
      a_arc_map and b_arc_map.
    '''
    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect_device(
        a_fsas.arcs, a_fsas.properties, b_fsas.arcs, b_fsas.properties,
        b_to_a_map, need_arc_map, sorted_match_a)

    out_fsas = k2.utils.fsa_from_binary_function_tensor(a_fsas, b_fsas,
                                                        ragged_arc,
                                                        a_arc_map, b_arc_map)
    if ret_arc_maps:
        return out_fsas, a_arc_map, b_arc_map
    else:
        return out_fsas


def intersect(a_fsa: Fsa,
              b_fsa: Fsa,
              treat_epsilons_specially: bool = True,
              ret_arc_maps: bool = False
             ) -> Union[Fsa, Tuple[Fsa, torch.Tensor, torch.Tensor]]:  # noqa
    '''Compute the intersection of two FSAs.

    When `treat_epsilons_specially` is True, this function works only on CPU.
    When `treat_epsilons_specially` is False and both `a_fsa` and `b_fsa`
    are on GPU, then this function works on GPU; in this case, the two
    input FSAs do not need to be arc sorted.

    Args:
      a_fsa:
        The first input FSA. It can be either a single FSA or an FsaVec.
      b_fsa:
        The second input FSA. it can be either a single FSA or an FsaVec.
        If both a_fsa and b_fsa are FsaVec, they must contain the same
        number of FSAs.
      treat_epsilons_specially:
        If True, epsilons will be treated as epsilon, meaning epsilon arcs can
        match with an implicit epsilon self-loop.
        If False, epsilons will be treated as real, normal symbols (to have
        them treated as epsilons in this case you may have to add epsilon
        self-loops to whichever of the inputs is naturally epsilon-free).
      ret_arc_maps:
        If False, return the resulting Fsa. If True, return a tuple
        containing three entries:

            - the resulting Fsa

            - a_arc_map, a 1-D torch.Tensor with dtype torch.int32.
              a_arc_map[i] is the arc index in a_fsa that corresponds
              to the i-th arc in the resulting Fsa. a_arc_map[i] is -1
              if the i-th arc in the resulting Fsa has no corresponding
              arc in a_fsa.

            - b_arc_map, a 1-D torch.Tensor with dtype torch.int32.
              b_arc_map[i] is the arc index in b_fsa that corresponds
              to the i-th arc in the resulting Fsa. b_arc_map[i] is -1
              if the i-th arc in the resulting Fsa has no corresponding
              arc in b_fsa.

    Caution:
      The two input FSAs MUST be arc sorted if `treat_epsilons_specially`
      is True.

    Caution:
      The rules for assigning the attributes of the output Fsa are as follows:

      - (1) For attributes where only one source (a_fsa or b_fsa) has that
        attribute: Copy via arc_map, or use zero if arc_map has -1. This rule
        works for both floating point and integer attributes.

      - (2) For attributes where both sources (a_fsa and b_fsa) have that
        attribute: For floating point attributes: sum via arc_maps, or use zero
        if arc_map has -1. For integer attributes, it's not supported for now
        (the attributes will be discarded and will not be kept in the output
        FSA).

    Returns:
      If ret_arc_maps is False, return the result of intersecting a_fsa and
      b_fsa. len(out_fsa.shape) is 2 if and only if the two input FSAs are
      single FSAs; otherwise, len(out_fsa.shape) is 3.
      If ret_arc_maps is True, it returns additionally two arc_maps:
      a_arc_map and b_arc_map.
    '''
    if a_fsa.is_cpu() or b_fsa.is_cpu():
        assert a_fsa.properties & fsa_properties.ARC_SORTED != 0
        assert b_fsa.properties & fsa_properties.ARC_SORTED != 0

    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(
        a_fsa.arcs, a_fsa.properties, b_fsa.arcs, b_fsa.properties,
        treat_epsilons_specially, need_arc_map)

    out_fsa = k2.utils.fsa_from_binary_function_tensor(a_fsa, b_fsa, ragged_arc,
                                                       a_arc_map, b_arc_map)
    if ret_arc_maps:
        return out_fsa, a_arc_map, b_arc_map
    else:
        return out_fsa


def compose(a_fsa: Fsa,
            b_fsa: Fsa,
            treat_epsilons_specially: bool = True,
            inner_labels: Optional[str] = None) -> 'Fsa':
    '''Compute the composition of two FSAs.

    When `treat_epsilons_specially` is True, this function works only on CPU.
    When `treat_epsilons_specially` is False and both `a_fsa` and `b_fsa`
    are on GPU, then this function works on GPU; in this case, the two
    input FSAs do not need to be arc sorted.

    Note:
      `a_fsa.aux_labels` is required to be defined and it can be either
      a `torch.Tensor` or a ragged tensor of type `k2.RaggedInt`.
      If it is a ragged tensor, then it requires that a_fsa.requires_grad is
      False.

      For both FSAs, the `aux_labels` attribute is interpreted as output labels,
      (olabels), and the composition involves matching the olabels of a_fsa with
      the ilabels of b_fsa.  This is implemented by intersecting the inverse of
      a_fsa (a_fsa_inv) with b_fsa, then replacing the ilabels of the result
      with the original ilabels on a_fsa which are now the aux_labels of
      a_fsa_inv.  If `b_fsa.aux_labels` is not defined, `b_fsa` is treated as an
      acceptor (as in OpenFST), i.e. its olabels and ilabels are assumed to be
      the same.

    Refer to :func:`k2.intersect` for how we assign the attributes of the
    output FSA.

    Args:
      a_fsa:
        The first input FSA. It can be either a single FSA or an FsaVec.
      b_fsa:
        The second input FSA. it can be either a single FSA or an FsaVec.
      treat_epsilons_specially:
        If True, epsilons will be treated as epsilon, meaning epsilon arcs can
        match with an implicit epsilon self-loop.
        If False, epsilons will be treated as real, normal symbols (to have
        them treated as epsilons in this case you may have to add epsilon
        self-loops to whichever of the inputs is naturally epsilon-free).
     inner_labels:
        If specified (and if a_fsa has `aux_labels`), the labels that we matched
        on, which would normally be discarded, will instead be copied to
        this attribute name.

    Caution:
      `b_fsa` has to be arc sorted if the function runs on CPU.

    Returns:
      The result of composing a_fsa and b_fsa. `len(out_fsa.shape)` is 2
      if and only if the two input FSAs are single FSAs;
      otherwise, `len(out_fsa.shape)` is 3.
    '''
    try:
        assert isinstance(a_fsa.aux_labels, Tensor)
    except Exception as e:
        raise ValueError("Expected a_fsa to have aux_labels (not ragged): ",
                         str(e))
    a_fsa_inv = a_fsa.invert()
    a_fsa_inv = arc_sort(a_fsa_inv)
    if treat_epsilons_specially is True or b_fsa.is_cpu():
        # the GPU version does not need to sort b.
        assert b_fsa.properties & fsa_properties.ARC_SORTED != 0

    a_fsa_inv.rename_tensor_attribute_('aux_labels', 'left_labels')

    # Internally this will call host intersection, or Intersect(), or
    # IntersectDevice(), according to various criteria such as CPU vs. GPU
    # and treat_epsilons_specially == true or not (true not supported
    # on GPU).
    ans = intersect(a_fsa_inv, b_fsa,
                    treat_epsilons_specially=treat_epsilons_specially)

    if inner_labels is not None:
        ans.rename_tensor_attribute_('labels', inner_labels)

    if not hasattr(b_fsa, 'aux_labels'):
        assert inner_labels is None and 'it should not be necessary to set ' \
            'inner_labels if b has no aux_labels'
        ans.rename_tensor_attribute_('labels', 'aux_labels')

    ans.rename_tensor_attribute_('left_labels', 'labels')
    return ans


def connect(fsa: Fsa) -> Fsa:
    '''Connect an FSA.

    Removes states that are neither accessible nor co-accessible.

    Note:
      A state is not accessible if it is not reachable from the start state.
      A state is not co-accessible if it cannot reach the final state.

    Caution:
      If the input FSA is already connected, it is returned directly.
      Otherwise, a new connected FSA is returned.

    Args:
     fsa:
        The input FSA to be connected.

    Returns:
      An FSA that is connected.
    '''
    if fsa.properties & fsa_properties.ACCESSIBLE != 0 and \
            fsa.properties & fsa_properties.COACCESSIBLE != 0:
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.connect(fsa.arcs, need_arc_map=need_arc_map)
    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    return out_fsa


def arc_sort(fsa: Fsa, ret_arc_map: bool = False
            ) -> Union[Fsa, Tuple[Fsa, torch.Tensor]]:  # noqa
    '''Sort arcs of every state.

    Note:
      Arcs are sorted by labels first, and then by dest states.

    Caution:
      If the input `fsa` is already arc sorted, we return it directly.
      Otherwise, a new sorted fsa is returned.

    Args:
      fsa:
        The input FSA.
      ret_arc_map:
        True to return an extra arc_map (a 1-D tensor with dtype being
        torch.int32). arc_map[i] is the arc index in the input `fsa` that
        corresponds to the i-th arc in the output Fsa.
    Returns:
      If ret_arc_map is False, return the sorted FSA. It is the same as the
      input `fsa` if the input `fsa` is arc sorted. Otherwise, a new sorted
      fsa is returned and the input `fsa` is NOT modified.
      If ret_arc_map is True, an extra arc map is also returned.
    '''
    if fsa.properties & fsa_properties.ARC_SORTED != 0:
        if ret_arc_map:
            # in this case, arc_map is an identity map
            arc_map = torch.arange(fsa.num_arcs,
                                   dtype=torch.int32,
                                   device=fsa.device)
            return fsa, arc_map
        else:
            return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.arc_sort(fsa.arcs, need_arc_map=need_arc_map)

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    if ret_arc_map:
        return out_fsa, arc_map
    else:
        return out_fsa


def shortest_path(fsa: Fsa, use_double_scores: bool) -> Fsa:
    '''Return the shortest paths as linear FSAs from the start state
    to the final state in the tropical semiring.

    Note:
      It uses the opposite sign. That is, It uses `max` instead of `min`.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
      use_double_scores:
        False to use float, i.e., single precision floating point, for scores.
        True to use double.

    Returns:
      FsaVec, it contains the best paths as linear FSAs
    '''
    entering_arcs = fsa._get_entering_arcs(use_double_scores)
    ragged_arc, ragged_int = _k2.shortest_path(fsa.arcs, entering_arcs)
    arc_map = ragged_int.values()

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    return out_fsa


def add_epsilon_self_loops(fsa: Fsa, ret_arc_map: bool = False
                          ) -> Union[Fsa, Tuple[Fsa, torch.Tensor]]:  # noqa
    '''Add epsilon self-loops to an Fsa or FsaVec.

    This is required when composing using a composition method that does not
    treat epsilons specially, if the other FSA has epsilons in it.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
      ret_arc_map:
        If False, return the resulting Fsa.
        If True, return an extra arc map.

    Returns:
      If ret_arc_map is False, return an instance of :class:`Fsa` that has an
      epsilon self-loop on every non-final state.
      If ret_arc_map is True, it returns an extra arc_map. arc_map[i] is the
      arc index in the input `fsa` that corresponds to the i-th arc in the
      resulting Fsa. arc_map[i] is -1 if the i-th arc in the resulting Fsa
      has no counterpart in the input `fsa`.
    '''

    need_arc_map = True
    ragged_arc, arc_map = _k2.add_epsilon_self_loops(fsa.arcs,
                                                     need_arc_map=need_arc_map)

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    if ret_arc_map:
        return out_fsa, arc_map
    else:
        return out_fsa


def remove_epsilon_self_loops(fsa: Fsa) -> Fsa:
    '''Remove epsilon self-loops of an Fsa or an FsaVec.

    Caution:
      Unlike :func:`remove_epsilon`, this funciton removes only
      epsilon self-loops.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.

    Returns:
      An instance of :class:`Fsa` that has no epsilon self-loops on every
      non-final state.
    '''
    need_arc_map = True
    ragged_arc, arc_map = _k2.remove_epsilon_self_loops(fsa.arcs, need_arc_map)

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
    return out_fsa


def remove_epsilon(fsa: Fsa) -> Fsa:
    '''Remove epsilons (symbol zero) in the input Fsa.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        Works either for CPU or GPU, but the algorithm is different.
        We can only use the CPU algorithm if the input is top-sorted,
        and the GPU algorithm, while it works for CPU, may not be
        very fast.

        `fsa` must be free of epsilon loops that have score
        greater than 0.

    Returns:
      The resulting Fsa is equivalent to the input `fsa` under the
      tropical semiring but will be epsilon-free.  Any linear tensor
      attributes, such as 'aux_labels', will have been turned into
      ragged labels after removing fillers (i.e. labels whose
      value equals fsa.XXX_filler if the attribute name is XXX),
      counting -1's on final-arcs as fillers even if the filler
      value for that attribute is not -1.
    '''
    ragged_arc, arc_map = _k2.remove_epsilon(fsa.arcs, fsa.properties)

    out_fsa = k2.utils.fsa_from_unary_function_ragged(fsa, ragged_arc, arc_map,
                                                      remove_filler=True)

    if hasattr(out_fsa, 'aux_labels') and \
            isinstance(out_fsa.aux_labels, k2.RaggedInt):
        out_fsa.aux_labels = k2.ragged.remove_values_eq(out_fsa.aux_labels, 0)

    return out_fsa


def remove_epsilon_iterative_tropical(fsa: Fsa) -> Fsa:
    '''A wrapper for remove_epsilon(), deprecated but provided
    for back compatibility'''
    return remove_epsilon(fsa)


def remove_epsilon_and_add_self_loops(fsa: Fsa,
                                      remove_filler: bool = True) -> Fsa:
    '''Remove epsilons (symbol zero) in the input Fsa, and then add
    epsilon self-loops to all states in the input Fsa (usually as
    a preparation for intersection with treat_epsilons_specially=0).

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
      remove_filler:
        If true, we will remove any `filler values` of attributes when
        converting linear to ragged attributes.
    Returns:
      The resulting Fsa.   See :func:`remove_epsilon` for details.
      The only epsilons will be epsilon self-loops on all states.
    '''
    if fsa.properties & fsa_properties.EPSILON_FREE != 0:
        return add_epsilon_self_loops(fsa)

    ragged_arc, arc_map = _k2.remove_epsilon_and_add_self_loops(
        fsa.arcs, fsa.properties)

    out_fsa = k2.utils.fsa_from_unary_function_ragged(
        fsa, ragged_arc, arc_map, remove_filler=remove_filler)

    return out_fsa


def determinize(fsa: Fsa,
                weight_pushing_type: _k2.DeterminizeWeightPushingType =
                _k2.DeterminizeWeightPushingType.kNoWeightPushing) -> Fsa:
    '''Determinize the input Fsa.

    Caution:
      - It only works on for CPU.
      - Any weight_pushing_type value other than kNoWeightPushing causes
        the 'arc_derivs' to not accurately reflect the real derivatives,
        although this will not matter as long as the derivatives ultimately
        derive from FSA operations such as getting total scores or
        arc posteriors, which are insensitive to pushing.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        Must be connected. It's also expected to be epsilon-free,
        but this is not checked; in any case,
        epsilon will be treated as a normal symbol.
      weight_pushing_type:
        An enum value that determines what kind of weight pushing is desired,
        default kNoWeightPushing.

          kTropicalWeightPushing:
            use tropical semiring (actually, max on scores) for weight pushing.
          kLogWeightPushing:
            use log semiring (actually, log-sum on score) for weight pushing
          kNoWeightPushing:
            do no weight pushing; this will cause some delay in scores being
            emitted, and the weights created in this way will correspond
            exactly to those that would be produced by the arc_derivs.

        For decoding graph creation, we recommend kLogSumWeightPushing.
    Returns:
      The resulting Fsa, it's equivalent to the input `fsa` under
      tropical semiring but will be deterministic.
      It will be the same as the input `fsa` if the input
      `fsa` has property kFsaPropertiesArcSortedAndDeterministic.
      Otherwise, a new deterministic fsa is returned and the
      input `fsa` is NOT modified.
    '''
    assert fsa.is_cpu()
    assert fsa.requires_grad is False
    if fsa.properties & fsa_properties.ARC_SORTED_AND_DETERMINISTIC != 0:  # noqa
        return fsa

    ragged_arc, arc_map = _k2.determinize(fsa.arcs, weight_pushing_type)
    out_fsa = k2.utils.fsa_from_unary_function_ragged(fsa, ragged_arc, arc_map)
    return out_fsa


def closure(fsa: Fsa) -> Fsa:
    '''Compute the Kleene closure of the input FSA.

    Args:
      fsa:
        The input FSA. It has to be a single FSA. That is,
        len(fsa.shape) == 2.
    Returns:
      The resulting FSA which is the Kleene closure of the input FSA.
    '''

    def fix_aux_labels(src_aux_labels: torch.Tensor,
                       src_row_splits1: torch.Tensor,
                       arc_map: torch.Tensor) -> torch.Tensor:
        '''Fix the aux labels of the output FSA.

        Since :func:`_k2.closure` changes the labels of arcs entering
        the final state to 0, we need to change their corresponding
        aux labels to 0.

        Args:
          src_aux_labels:
            The aux labels of the input FSA.
          src_row_splits:
            The row splits1 of the input FSA.
          arc_map:
            The arc map produced by :func:`_k2.closure`.
        Returns:
          The aux_labels of the output fsa after converting -1 to 0.
        '''
        minus_one_index = torch.nonzero(src_aux_labels == -1, as_tuple=False)
        src_start_state_last_arc_index = src_row_splits1[1]

        minus_one_index[minus_one_index > src_start_state_last_arc_index] += 1

        # now minus one index contains arc indexes into the output FSA
        ans_aux_labels = index_select(src_aux_labels, arc_map)
        ans_aux_labels[minus_one_index] = 0
        ans_aux_labels[src_start_state_last_arc_index] = -1
        return ans_aux_labels

    need_arc_map = True
    ragged_arc, arc_map = _k2.closure(fsa.arcs, need_arc_map=need_arc_map)

    out_fsa = Fsa(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        if name == 'aux_labels':
            new_value = fix_aux_labels(value, fsa.arcs.row_splits(1), arc_map)
        else:
            new_value = index_select(value, arc_map)
        setattr(out_fsa, name, new_value)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def invert(fsa: Fsa,
           ret_arc_map: bool = False) -> Union[Fsa, Tuple[Fsa, torch.Tensor]]:
    '''Invert an FST, swapping the labels in the FSA with the auxiliary labels.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
      ret_arc_map:
        True to return an extra arc map, which is a 1-D tensor with dtype
        torch.int32. The returned arc_map[i] is the arc index in the input
        fsa that corresponds to the i-th arc in the returned fsa. arc_map[i]
        is -1 if the i-th arc in the returned fsa has no counterpart in the
        input fsa.
    Returns:
      If ret_arc_map is False, return the inverted Fsa, it's top-sorted if
      `fsa` is top-sorted.
      If ret_arc_map is True, return an extra arc map.
    '''
    if isinstance(fsa.aux_labels, torch.Tensor):
        if ret_arc_map is False:
            return fsa.invert()
        else:
            arc_map = torch.arange(fsa.num_arcs,
                                   dtype=torch.int32,
                                   device=fsa.device)
            return fsa.invert(), arc_map
    else:
        assert isinstance(fsa.aux_labels, k2.RaggedInt)
        fsa, arc_map = expand_ragged_attributes(
            fsa, ret_arc_map=True, ragged_attribute_names=['aux_labels'])
        fsa = fsa.invert_()
        if ret_arc_map:
            return fsa, arc_map
        else:
            return fsa


def random_paths(fsas: Fsa, use_double_scores: bool,
                 num_paths: int) -> k2.RaggedInt:
    '''Compute pseudo-random paths through the FSAs in this vector of FSAs
    (this object must have 3 axes, `self.arcs.num_axes() == 3`)

    Caution:
      It does not support autograd.

    Caution:
      Do not be confused by the function name. There is no
      randomness at all, thus no `seed`. It uses a deterministic algorithm
      internally, similar to arithmetic coding
      (see `<https://en.wikipedia.org/wiki/Arithmetic_coding>`_).

      Look into the C++ implementation code for more details.

    Args:
      fsas:
        A FsaVec, i.e., `len(fsas.shape) == 3`
      use_double_scores:
        If true, do computation with double-precision,
        else float (single-precision)
      num_paths:
        Number of paths requested through each FSA. FSAs that have no successful
        paths will have zero paths returned.
    Returns:
      Returns a k2.RaggedInt with 3 axes: [fsa][path][arc_pos]; the final
      sub-lists (indexed with arc_pos) are sequences of arcs starting from the
      start state and terminating in the final state. The values are arc_idx012,
      i.e. arc indexes.
    '''
    assert num_paths > 0, f'num_paths: {num_paths}'
    log_semiring = True
    arc_cdf = fsas._get_arc_cdf(use_double_scores=use_double_scores,
                                log_semiring=log_semiring)
    tot_scores = fsas._get_tot_scores(use_double_scores=use_double_scores,
                                      log_semiring=log_semiring)
    state_batches = fsas._get_state_batches()
    if use_double_scores:
        func = _k2.random_paths_double
    else:
        func = _k2.random_paths_float

    ans = func(fsas=fsas.arcs,
               arc_cdf=arc_cdf,
               num_paths=num_paths,
               tot_scores=tot_scores,
               state_batches=state_batches)
    return ans


def prune_on_arc_post(fsas: Fsa, threshold_prob: float,
                      use_double_scores: bool) -> Fsa:
    '''Remove arcs whose posteriors are less than the given threshold.

    Args:
      fsas:
        An FsaVec. Must have 3 axes.
      threshold_prob:
        Arcs whose posteriors are less than this value are removed.
        Note:
          0 < threshold_prob < 1
      use_double_scores:
        True to use double precision during computation; False to use
        single precision.
    Returns:
      Return a pruned FsaVec.
    '''
    arc_post = fsas.get_arc_post(use_double_scores=use_double_scores,
                                 log_semiring=True)
    need_arc_map = True
    if use_double_scores:
        func = _k2.prune_on_arc_post_double
    else:
        func = _k2.prune_on_arc_post_float

    ragged_arc, arc_map = func(fsas.arcs, arc_post, threshold_prob,
                               need_arc_map)

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsas, ragged_arc,
                                                      arc_map)
    return out_fsa


def expand_ragged_attributes(
        fsas: Fsa,
        ret_arc_map: bool = False,
        ragged_attribute_names: Optional[List[str]] = None
) -> Union[Fsa, Tuple[Fsa, torch.Tensor]]:  # noqa
    '''
    Turn ragged labels attached to this FSA into linear (Tensor) labels,
    expanding arcs into sequences of arcs as necessary to achieve this.
    Supports autograd.  If `fsas` had no ragged attributes, returns `fsas`
    itself.

    Caution:
      This function will ensure that for final-arcs in the returned
      fsa, the corresponding labels for all ragged attributes are -1; it will
      add an extra arc at the end is necessary to ensure this, if the
      original ragged attributes did not have -1 as their final element on
      final-arcs (note: our intention is that -1's on final arcs, like filler
      symbols, are removed when making attributes ragged; this is what
      fsa_from_unary_function_ragged() does if remove_filler==True (the
      default).

    Args:
      fsas:
        The source Fsa
      ret_arc_map:
        If true, will return a pair (new_fsas, arc_map)
        with `arc_map` a tensor of int32 that maps from arcs in the
        result to arcs in `fsas`, with -1's for newly created arcs.
        If false, just returns new_fsas.
      ragged_attribute_names:
        If specified, just this list of ragged
        attributes will be expanded to linear tensor attributes, and
        the rest will stay ragged.
    '''
    if ragged_attribute_names is None:
        ragged_attribute_tensors = []
        ragged_attribute_names = []
        for name, value in fsas.named_tensor_attr(include_scores=False):
            if isinstance(value, k2.RaggedInt):
                ragged_attribute_tensors.append(value)
                ragged_attribute_names.append(name)
    else:
        ragged_attribute_tensors = [getattr(fsas, name)
                                    for name in ragged_attribute_names]
        for t in ragged_attribute_tensors:
            assert isinstance(t, k2.RaggedInt)

    if len(ragged_attribute_tensors) == 0:
        if ret_arc_map:
            arc_map = torch.arange(fsas.num_arcs, dtype=torch.int32,
                                   device=fsas.device)
            return (fsas, arc_map)
        else:
            return fsas

    (dest_arcs, dest_labels, arc_map) = _k2.expand_arcs(
        fsas.arcs,
        ragged_attribute_tensors)

    # The rest of this function is a modified version of
    # `fsa_from_unary_function_tensor()`.
    dest = Fsa(dest_arcs)

    # Handle the non-ragged attributes, and ragged attributes that
    # we're not linearizing.
    for name, value in fsas.named_tensor_attr(include_scores=False):
        if isinstance(value, torch.Tensor):
            filler = float(fsas.get_filler(name))
            setattr(dest, name, index_select(value, arc_map,
                                             default_value=filler))
        elif name not in ragged_attribute_names:
            setattr(dest, name, index(value, arc_map))

    # Handle the attributes that were ragged but are now linear
    for name, value in zip(ragged_attribute_names, dest_labels):
        setattr(dest, name, value)

    # Copy non-tensor attributes
    for name, value in fsas.named_non_tensor_attr():
        setattr(dest, name, value)

    # make sure autograd works on the scores
    k2.autograd_utils.phantom_index_select_scores(dest, fsas.scores, arc_map)

    # Make sure -1's are only on final-arcs, and never on non-final arcs.
    if hasattr(dest, 'aux_labels'):
        _k2.fix_final_labels(dest.arcs, dest.aux_labels)

    if ret_arc_map:
        return dest, arc_map
    else:
        return dest


def replace_fsa(
        src: Fsa,
        index: Fsa,
        symbol_begin_range: int = 1,
        ret_arc_map: bool = False
) -> Union[Fsa, Tuple[Fsa, torch.Tensor, torch.Tensor]]:

    '''
    Replace arcs in index FSA with the corresponding fsas in a vector of
    FSAs(src). For arcs in `index` with label
    `symbol_range_begin <= label < symbol_range_begin + src.Dim0()` will be
    replaced with fsa indexed `label - symbol_begin_range` in `src`.
    The destination state of the arc in `index` is identified with the
    `final-state` of the corresponding FSA in `src`, and the arc in `index`
    will become an epsilon arc leading to a new state in the output that is
    a copy of the start-state of the corresponding FSA in `src`. Arcs with
    labels outside this range are just copied. Labels on final-arcs in `src`
    (Which will be -1) would be set to 0(epsilon) in the result fsa.

    Caution: Attributes of the result inherits from `index` and `src` via
             `arc_map_index` and `arc_map_src`, But if there are attributes
             with same name, only the attributes with dtype `torch.float32`
             are supported, the other kinds of attributes are discarded.
             See docs in `fsa_from_binary_function_tensor` for details.

    Args:
      src:
          Fsa that we'll be inserting into the result, MUST have 3 axes.
      index:
          The Fsa that is to be replaced, It can be a single FSA or a vector of
          FSAs.
      symbol_range_begin:
          Beginning of the range of symbols that are to be replaced with Fsas.
      ret_arc_map:  if true, will return a tuple
           (new_fsas, arc_map_index, arc_map_src) with `arc_map_index` and
           `arc_map_src` tensors of int32 that maps from arcs in the result to
           arcs in `index` and `src` , with -1's for the arcs not mapped.
           If false, just returns new_fsas.
    '''
    (dest_arc, arc_map_src, arc_map_index) = _k2.replace_fsa(
        src.arcs, index.arcs, symbol_begin_range)

    dest = k2.utils.fsa_from_binary_function_tensor(src, index, dest_arc,
                                                    arc_map_src, arc_map_index)
    if ret_arc_map:
        return dest, arc_map_index, arc_map_src
    else:
        return dest


def ctc_graph(symbols: Union[List[List[int]], k2.RaggedInt],
              modified: bool = False,
              device: Optional[Union[torch.device, str]] = None) -> Fsa:
    '''Construct ctc graphs from symbols.

    Note:
      The scores of arcs in the returned FSA are all 0.

    Args:
      symbols:
        It can be one of the following types:

            - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
            - An instance of :class:`k2.RaggedInt`. Must have `num_axes() == 2`.

      standard:
        Option to specify the type of CTC topology: "standard" or "simplified",
        where the "standard" one makes the blank mandatory between a pair of
        identical symbols. Default True.
      device:
        Optional. It can be either a string (e.g., 'cpu', 'cuda:0') or a
        torch.device.
        If it is None, then the returned FSA is on CPU. It has to be None
        if `symbols` is an instance of :class:`k2.RaggedInt`, the returned
        FSA will on the same device as `k2.RaggedInt`.

    Returns:
        An FsaVec containing the returned ctc graphs, with "Dim0()" the same as
        "len(symbols)"(List[List[int]]) or "Dim0()"(k2.RaggedInt)
    '''
    if device is not None:
        device = torch.device(device)
        if device.type == 'cpu':
            gpu_id = -1
        else:
            assert device.type == 'cuda'
            gpu_id = getattr(device, 'index', 0)
    else:
        gpu_id = -1

    symbol_values = None
    if isinstance(symbols, k2.RaggedInt):
        assert device is None
        assert symbols.num_axes() == 2
        symbol_values = symbols.values()
    else:
        symbol_values = torch.tensor(
            [it for symbol in symbols for it in symbol], dtype=torch.int32,
            device=device)

    need_arc_map = True
    ragged_arc, arc_map = _k2.ctc_graph(symbols, gpu_id,
                                        modified, need_arc_map)
    aux_labels = k2.index(symbol_values, arc_map)
    fsa = Fsa(ragged_arc, aux_labels=aux_labels)
    return fsa


def ctc_topo(max_token: int, modified: bool = False,
             device: Optional[Union[torch.device, str]] = None) -> k2.Fsa:
    '''Create a CTC topology.

    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.

    A standard CTC topology is the conventional one, where there
    is a mandatory blank between two repeated neighboring symbols.
    A non-standard, i.e., modified CTC topology, imposes no such constraint.

    See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
    and https://github.com/k2-fsa/snowfall/pull/209
    for more details.

    Args:
      max_token:
        The maximum token ID (inclusive). We assume that token IDs
        are contiguous (from 1 to `max_token`). 0 represents blank.
      modified:
        If False, create a standard CTC topology. Otherwise, create a
        modified CTC topology.
      device:
        Optional. It can be either a string (e.g., 'cpu',
        'cuda:0') or a torch.device.
        If it is None, then the returned FSA is on CPU.
    Returns:
      Return either a standard or a modified CTC topology as an FSA
      depending on whether `standard` is True or False.
    '''
    if device is not None:
        device = torch.device(device)
        if device.type == 'cpu':
            gpu_id = -1
        else:
            assert device.type == 'cuda'
            gpu_id = getattr(device, 'index', 0)
    else:
        gpu_id = -1
    ragged_arc, aux_labels = _k2.ctc_topo(max_token, gpu_id, modified)
    fsa = Fsa(ragged_arc, aux_labels=aux_labels)
    return fsa
