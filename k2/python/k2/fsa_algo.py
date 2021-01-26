# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
#                2021  Mobvoi Inc.        (authors: Yaguang Hu)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import List
from typing import Union

import torch
import _k2

from . import fsa_properties
from .fsa import Fsa
from .ops import index
from .ops import index_select

# Note: look also in autograd.py, differentiable operations may be there.


def linear_fsa(labels: Union[List[int], List[List[int]]]) -> Fsa:
    '''Construct an linear FSA from labels.

    Note:
      The scores of arcs in the returned FSA are all 0.

    Args:
      labels:
        A list of integers or a list of list of integers.

    Returns:
      An FSA if the labels is a list of integers.
      A vector of FSAs if the input is a list of list of integers.
    '''
    ragged_arc = _k2.linear_fsa(labels)
    fsa = Fsa(ragged_arc)
    return fsa


def linear_fst(labels: Union[List[int], List[List[int]]],
               aux_labels: Union[List[int], List[List[int]]]) -> Fsa:
    '''Construct an linear FST from labels and its corresponding
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
      A vector of FSTs if the input is a list of list of integers.
    '''
    ragged_arc = _k2.linear_fsa(labels)
    aux_labels_tmp = []
    if isinstance(labels[0], List):
        assert isinstance(aux_labels[0],
                          List), 'aux_labels and labels do not match.'
        for aux in aux_labels:
            aux_labels_tmp.extend(aux + [-1])  # -1 == kFinalSymbol
        aux_labels = torch.IntTensor(aux_labels_tmp)
    else:
        aux_labels_tmp = aux_labels + [-1]  # -1 == kFinalSymbol
    fsa = Fsa(ragged_arc, aux_labels=torch.IntTensor(aux_labels_tmp))
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
    sorted_fsa = Fsa(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(sorted_fsa, name, index(value, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(sorted_fsa, name, value)

    return sorted_fsa


def intersect_device(a_fsas: Fsa, b_fsas: Fsa,
                     b_to_a_map: torch.Tensor) -> Fsa:
    '''Compute the intersection of two FSAs treating epsilons
    as real, normal symbols.

    This function supports both CPU and GPU. But it is very slow on CPU.
    That's why this function name ends with `_device`. It is intended for GPU.
    See :func:`k2.intersect` for intersecting two FSAs on CPU.

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

    Returns:
      Returns composed FsaVec; will satisfy `ans.shape == b_fsas.shape`.
    '''
    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect_device(
        a_fsas.arcs, a_fsas.properties, b_fsas.arcs, b_fsas.properties,
        b_to_a_map, need_arc_map)
    out_fsas = Fsa(ragged_arc)

    for name, a_value in a_fsas.named_tensor_attr():
        if hasattr(b_fsas, name):
            # Both a_fsas and b_fsas have this attribute.
            # We only support attributes with dtype `torch.float32`.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                continue
            b_value = getattr(b_fsas, name)
            assert b_value.dtype == torch.float32

            value = index_select(a_value, a_arc_map) \
                    + index_select(b_value, b_arc_map)
            setattr(out_fsas, name, value)
        else:
            # only a_fsas has this attribute, copy it via arc_map
            value = index(a_value, a_arc_map)
            setattr(out_fsas, name, value)

    # now copy tensor attributes that are in b_fsas but are not in a_fsas
    for name, b_value in b_fsas.named_tensor_attr():
        if not hasattr(out_fsas, name):
            value = index(b_value, b_arc_map)
            setattr(out_fsas, name, value)

    for name, a_value in a_fsas.named_non_tensor_attr():
        setattr(out_fsas, name, a_value)

    for name, b_value in b_fsas.named_non_tensor_attr():
        if not hasattr(out_fsas, name):
            setattr(out_fsas, name, b_value)

    return out_fsas


def intersect(a_fsa: Fsa, b_fsa: Fsa,
              treat_epsilons_specially: bool = True) -> Fsa:
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
      treat_epsilons_specially:
        If True, epsilons will be treated as epsilon, meaning epsilon arcs can
        match with an implicit epsilon self-loop.
        If False, epsilons will be treated as real, normal symbols (to have
        them treated as epsilons in this case you may have to add epsilon
        self-loops to whichever of the inputs is naturally epsilon-free).

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
      The result of intersecting a_fsa and b_fsa. len(out_fsa.shape) is 2
      if and only if the two input FSAs are single FSAs;
      otherwise, len(out_fsa.shape) is 3.
    '''
    if a_fsa.is_cpu() or b_fsa.is_cpu():
        assert a_fsa.properties & fsa_properties.ARC_SORTED != 0
        assert b_fsa.properties & fsa_properties.ARC_SORTED != 0

    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(
        a_fsa.arcs, a_fsa.properties, b_fsa.arcs, b_fsa.properties,
        treat_epsilons_specially, need_arc_map)

    out_fsa = Fsa(ragged_arc)
    for name, a_value in a_fsa.named_tensor_attr():
        if hasattr(b_fsa, name):
            # Both a_fsa and b_fsa have this attribute.
            # We only support attributes with dtype `torch.float32`.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                continue
            b_value = getattr(b_fsa, name)
            assert b_value.dtype == torch.float32

            value = index_select(a_value, a_arc_map) \
                    + index_select(b_value, b_arc_map)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            value = index(a_value, a_arc_map)
            setattr(out_fsa, name, value)

    # now copy tensor attributes that are in b_fsa but are not in a_fsa
    for name, b_value in b_fsa.named_tensor_attr():
        if not hasattr(out_fsa, name):
            value = index(b_value, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

    return out_fsa


def compose(a_fsa: Fsa,
            b_fsa: Fsa,
            treat_epsilons_specially: bool = True,
            inner_labels: str = None) -> Fsa:
    '''Compute the composition of two FSAs (currently on CPU).

    When `treat_epsilons_specially` is True, this function works only on CPU.
    When `treat_epsilons_specially` is False and both `a_fsa` and `b_fsa`
    are on GPU, then this function works on GPU; in this case, the two
    input FSAs do not need to be arc sorted.

    Note:
      `a_fsa.aux_labels` is required to be defined.

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
    assert hasattr(a_fsa, 'aux_labels')

    assert isinstance(a_fsa.aux_labels, torch.Tensor)

    a_fsa_inv = a_fsa.invert()
    if treat_epsilons_specially is True or a_fsa_inv.is_cpu():
        a_fsa_inv = arc_sort(a_fsa_inv)

    if treat_epsilons_specially is True or b_fsa.is_cpu():
        assert b_fsa.properties & fsa_properties.ARC_SORTED != 0

    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(
        a_fsa_inv.arcs, a_fsa_inv.properties, b_fsa.arcs, b_fsa.properties,
        treat_epsilons_specially, need_arc_map)

    out_fsa = Fsa(ragged_arc)
    if inner_labels is not None:
        # out_fsa.`inner_labels` = out_fsa.labels
        setattr(out_fsa, inner_labels, out_fsa.labels)

    if hasattr(b_fsa, 'aux_labels'):
        out_fsa.aux_labels = index(b_fsa.aux_labels, b_arc_map)
    else:
        # need a clone here since `Fsa.labels` is a reference
        out_fsa.aux_labels = out_fsa.labels.clone()

    out_fsa.labels = index(a_fsa_inv.aux_labels, a_arc_map)

    for name, a_value in a_fsa_inv.named_tensor_attr():
        if name == 'aux_labels':
            continue
        if hasattr(b_fsa, name):
            # Both a_fsa and b_fsa have this attribute.
            # We only support attributes with dtype `torch.float32`.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                continue
            b_value = getattr(b_fsa, name)
            assert b_value.dtype == torch.float32

            # The following will actually overwrite `scores` with the same
            # value it had before; but this enables the autograd to work since
            # we do it using torch mechanisms.
            value = index_select(a_value, a_arc_map) + index_select(
                b_value, b_arc_map)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            value = index(a_value, a_arc_map)
            setattr(out_fsa, name, value)

    # now copy tensor attributes that are in b_fsa but are not in a_fsa
    for name, b_value in b_fsa.named_tensor_attr():
        if name == 'aux_labels':
            continue
        if not hasattr(out_fsa, name):
            value = index(b_value, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa_inv.named_non_tensor_attr():
        if name == 'symbols':
            continue

        if name == 'aux_symbols':
            setattr(out_fsa, 'symbols', a_value)
        else:
            setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if name == 'symbols' and not hasattr(b_fsa, 'aux_labels'):
            setattr(out_fsa, 'aux_symbols', b_value)
        elif not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

    return out_fsa


def connect(fsa: Fsa) -> Fsa:
    '''Connect an FSA.

    Removes states that are neither accessible nor co-accessible.

    It works only on CPU.

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

    assert fsa.is_cpu()

    need_arc_map = True
    ragged_arc, arc_map = _k2.connect(fsa.arcs, need_arc_map=need_arc_map)
    out_fsa = Fsa(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index(value, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def arc_sort(fsa: Fsa) -> Fsa:
    '''Sort arcs of every state.

    Note:
      Arcs are sorted by labels first, and then by dest states.

    Caution:
      If the input `fsa` is already arc sorted, we return it directly.
      Otherwise, a new sorted fsa is returned.

    Args:
      fsa:
        The input FSA.
    Returns:
      The sorted FSA. It is the same as the input `fsa` if the input
      `fsa` is arc sorted. Otherwise, a new sorted fsa is returned
      and the input `fsa` is NOT modified.
    '''
    if fsa.properties & fsa_properties.ARC_SORTED != 0:
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.arc_sort(fsa.arcs, need_arc_map=need_arc_map)
    out_fsa = Fsa(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index(value, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

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
    out_fsa = Fsa(ragged_arc)

    arc_map = ragged_int.values()
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index(value, arc_map))

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def add_epsilon_self_loops(fsa: Fsa) -> Fsa:
    '''Add epsilon self-loops to an Fsa or FsaVec.

    This is required when composing using a composition method that does not
    treat epsilons specially, if the other FSA has epsilons in it.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.

    Returns:
      An instance of :class:`Fsa` that has an epsilon self-loop on every
      non-final state.
    '''

    need_arc_map = True
    ragged_arc, arc_map = _k2.add_epsilon_self_loops(fsa.arcs,
                                                     need_arc_map=need_arc_map)

    out_fsa = Fsa(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        new_value = index(value, arc_map)
        setattr(out_fsa, name, new_value)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def remove_epsilon(fsa: Fsa) -> Fsa:
    '''Remove epsilons (symbol zero) in the input Fsa.

    Caution:
      It only works on for CPU and doesn't support autograd.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        Must be top-sorted.
    Returns:
        The result Fsa, it's equivalent to the input `fsa` under
        tropical semiring but will be epsilon-free.
        It will be the same as the input `fsa` if the input
        `fsa` is epsilon-free. Otherwise, a new epsilon-free fsa
        is returned and the input `fsa` is NOT modified.
    '''
    assert fsa.is_cpu()
    assert fsa.requires_grad is False
    if fsa.properties & fsa_properties.EPSILON_FREE != 0:
        return fsa

    ragged_arc, arc_map = _k2.remove_epsilon(fsa.arcs)
    aux_labels = None
    if hasattr(fsa, 'aux_labels'):
        aux_labels = index(fsa.aux_labels, arc_map)
    out_fsa = Fsa(ragged_arc, aux_labels)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def remove_epsilons_iterative_tropical(fsa: Fsa) -> Fsa:
    '''Remove epsilons (symbol zero) in the input Fsa.

    Caution:
      It doesn't support autograd for now.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        It can be either top-sorted or non-top-sorted.
    Returns:
        The result Fsa, it's equivalent to the input `fsa` under
        tropical semiring but will be epsilon-free.
        It will be the same as the input `fsa` if the input
        `fsa` is epsilon-free. Otherwise, a new epsilon-free fsa
        is returned and the input `fsa` is NOT modified.
    '''
    if fsa.properties & fsa_properties.EPSILON_FREE != 0:
        return fsa

    ragged_arc, arc_map = _k2.remove_epsilons_iterative_tropical(fsa.arcs)
    aux_labels = None
    if hasattr(fsa, 'aux_labels'):
        aux_labels = index(fsa.aux_labels, arc_map)
    out_fsa = Fsa(ragged_arc, aux_labels)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def determinize(fsa: Fsa) -> Fsa:
    '''Determinize the input Fsa.

    Caution:
      It only works on for CPU and doesn't support autograd (for now;
      this is not a fundamental limitation).

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        Must be connected. It's also expected to be epsilon-free,
        but this is not checked; in any case,
        epsilon will be treated as a normal symbol.
    Returns:
        The result Fsa, it's equivalent to the input `fsa` under
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

    ragged_arc, arc_map = _k2.determinize(fsa.arcs)
    aux_labels = None
    if hasattr(fsa, 'aux_labels'):
        aux_labels = index(fsa.aux_labels, arc_map)
    out_fsa = Fsa(ragged_arc, aux_labels)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def closure(fsa: Fsa) -> Fsa:
    '''Compute the Kleene closure of the input FSA.

    Args:
      fsa:
        The input FSA. It has to be a single FSA. That is,
        len(fsa.shape) == 2.
    Returns:
      The result FSA which is the Kleene closure of the input FSA.
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


def invert(fsa: Fsa) -> Fsa:
    '''Invert an FST, swapping the labels in the FSA with the auxiliary labels.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
    Returns:
      The inverted Fsa, it's top-sorted if `fsa` is top-sorted.
    '''
    assert fsa.is_cpu()
    assert fsa.requires_grad is False
    if isinstance(fsa.aux_labels, torch.Tensor):
        return fsa.invert()
    else:
        assert isinstance(fsa.aux_labels, _k2.RaggedInt)
        need_arc_map = False
        ragged_arc, aux_labels, _ = _k2.invert(fsa.arcs, fsa.aux_labels,
                                               need_arc_map)
        return Fsa(ragged_arc, aux_labels)
