# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
#                2021  Mobvoi Inc.        (authors: Yaguang Hu)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import List
from typing import Optional
from typing import Union

import torch
import _k2
import k2

from . import fsa_properties
from .fsa import Fsa
from .ops import index
from .ops import index_select

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
        # out_fsa.`inner_labels` = out_fsa.labels.clone()
        setattr(out_fsa, inner_labels, out_fsa.labels.clone())

    if hasattr(b_fsa, 'aux_labels'):
        out_fsa.aux_labels = index(b_fsa.aux_labels, b_arc_map)
    else:
        # need a clone here since `Fsa.labels` is a reference
        out_fsa.aux_labels = out_fsa.labels.clone()

    out_fsa.labels = index(a_fsa_inv.aux_labels, a_arc_map)

    for name, a_value in a_fsa_inv.named_tensor_attr():
        if name in ('aux_labels', inner_labels):
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
        if name in ('aux_labels', inner_labels):
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

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
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

    out_fsa = k2.utils.fsa_from_unary_function_tensor(fsa, ragged_arc, arc_map)
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
      The resulting Fsa, it's equivalent to the input `fsa` under
      tropical semiring but will be epsilon-free.
      It will be the same as the input `fsa` if the input
      `fsa` is epsilon-free. Otherwise, a new epsilon-free fsa
      is returned and the input `fsa` is NOT modified.
    '''
    if fsa.properties & fsa_properties.EPSILON_FREE != 0:
        return fsa

    ragged_arc, arc_map = _k2.remove_epsilon(fsa.arcs, fsa.properties)

    out_fsa = k2.utils.fsa_from_unary_function_ragged(fsa, ragged_arc, arc_map)
    return out_fsa


def remove_epsilon_iterative_tropical(fsa: Fsa) -> Fsa:
    '''A wrapper for remove_epsilon(), deprecated but provided
    for back compatibility'''
    return remove_epsilon(fsa)


def remove_epsilon_and_add_self_loops(fsa: Fsa) -> Fsa:
    '''Remove epsilons (symbol zero) in the input Fsa, and then add
    epsilon self-loops to all states in the input Fsa (usually as
    a preparation for intersection with treat_epsilons_specially=0).

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
    Returns:
      The resulting Fsa.   See :func:`remove_epsilon` for details.
      The only epsilons will be epsilon self-loops on all states.
    '''
    if fsa.properties & fsa_properties.EPSILON_FREE != 0:
        return add_epsilon_self_loops(fsa)

    ragged_arc, arc_map = _k2.remove_epsilon_and_add_self_loops(
        fsa.arcs, fsa.properties)

    out_fsa = k2.utils.fsa_from_unary_function_ragged(fsa, ragged_arc, arc_map)
    return out_fsa


def determinize(fsa: Fsa) -> Fsa:
    '''Determinize the input Fsa.

    Caution:
      It only works on for CPU.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or an FsaVec.
        Must be connected. It's also expected to be epsilon-free,
        but this is not checked; in any case,
        epsilon will be treated as a normal symbol.
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

    ragged_arc, arc_map = _k2.determinize(fsa.arcs)
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
        assert isinstance(fsa.aux_labels, k2.RaggedInt)
        need_arc_map = False
        ragged_arc, aux_labels, _ = _k2.invert(fsa.arcs, fsa.aux_labels,
                                               need_arc_map)
        return Fsa(ragged_arc, aux_labels)


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
