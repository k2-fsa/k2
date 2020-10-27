# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Union
from typing import List

from .fsa import Fsa
import _k2
import torch


def linear_fsa(symbols: Union[List[int], List[List[int]]]) -> Fsa:
    '''Construct an linear FSA from symbols.

    Note:
      The scores of arcs in the returned FSA are all 0.

    Args:
      symbols:
        A list of integers or a list of list of integers.

    Returns:
      An FSA if the input is a list of integers.
      A vector of FSAs if the input is a list of list of integers.
    '''
    ragged_arc = _k2.linear_fsa(symbols)
    fsa = Fsa.from_ragged_arc(ragged_arc)
    fsa.arc_labels_sorted = True
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
    arc_map = arc_map.to(torch.int64)  # required by index_select
    sorted_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(sorted_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(sorted_fsa, name, value)
    return sorted_fsa


def intersect(a_fsa: Fsa, b_fsa: Fsa) -> Fsa:
    '''Compute the intersection of two FSAs.

    Args:
      a_fsa:
        The first input FSA. It can be either a single FSA or a vector of FSAs.
      b_fsa:
        The second input FSA. It can be either a single FSA or a vector of FSAs.

    CAUTION:
      ``aux_labels`` are discarded since it is not clear how to set the
      aux_labels of the output FSA.

    Returns:
      The result of intersecting a_fsa and b_fsa.
    '''
    if not a_fsa.arc_labels_sorted:
        a_fsa = arc_sort(a_fsa)

    if not b_fsa.arc_labels_sorted:
        b_fsa = arc_sort(b_fsa)

    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(a_fsa.arcs, b_fsa.arcs,
                                                     need_arc_map)

    # Some of entries in a_arc_map and b_arc_map may be -1.
    # The arc_maps are incremented so that every entry is non-negative.
    a_arc_map = a_arc_map.to(torch.int64) + 1
    b_arc_map = b_arc_map.to(torch.int64) + 1

    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, a_value in a_fsa.named_tensor_attr():
        if name == 'aux_labels':
            # aux_labels are discarded!
            continue
        if hasattr(b_fsa, name):
            b_value = getattr(b_fsa, name)

            # a_arc_map and b_arc_map have been offset by 1
            # so we need a padding here
            padding = a_value.new_zeros((1, *a_value.shape[1:]))
            a_value = torch.cat((padding, a_value), dim=0)
            b_value = torch.cat((padding, b_value), dim=0)

            assert a_value.dtype == b_value.dtype == torch.float32
            value = a_value.index_select(0, a_arc_map) \
                    + b_value.index_select(0, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, b_value)

    out_fsa.arc_labels_sorted = True

    return out_fsa


def connect(fsa: Fsa) -> Fsa:
    '''Connect an FSA.

    Removes states that are neither accessible nor co-accessible.

    Note:
      A state is not accessible if it is not reachable from the start state.
      A state is not co-accessible if it cannot reach the final state.

    Args:
      fsa:
        The input FSA to be connected.

    Returns:
      An FSA that is connected.
    '''

    need_arc_map = True
    ragged_arc, arc_map = _k2.connect(fsa.arcs, need_arc_map=need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)
    return out_fsa


def arc_sort(fsa: Fsa) -> Fsa:
    '''Sort arcs of every state.

    Note:
      Arcs are sorted by labels first, and then by dest states.

    Caution:
      If the input ``fsa`` is already arc sorted, we return it directly.
      Otherwise, a new sorted fsa is returned.

    Args:
      fsa:
        The input FSA.
    Returns:
      The sorted FSA. It is the same as the input ``fsa`` if the input
      ``fsa`` is arc sorted. Otherwise, a new sorted fsa is returned
      and the input ``fsa`` is NOT modified.
    '''
    if fsa.arc_labels_sorted is True:
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.arc_sort(fsa.arcs, need_arc_map=need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    out_fsa.arc_labels_sorted = True
    if hasattr(out_fsa, 'arc_aux_labels_sorted'):
        out_fsa.arc_aux_labels_sorted = False

    return out_fsa


def compose(a_fsa: Fsa, b_fsa: Fsa) -> Fsa:
    '''Compose two FSAs.

    If ``a_fsa`` has ``aux_labels``, the compose is based on the ``aux_labels``
    of ``a_fsa`` and ``labels`` of ``b_fsa``.

    If ``a_fsa`` has no ``aux_labels``, the compose is based on the ``labels``
    of the two input fsas.

    Args:
      a_fsa:
        The first input FSA. It can be either a single FSA or a vector of FSAs.
      b_fsa:
        The second input FSA. It can be either a single FSA or a vector of FSAs.
    Returns:
      The result of composing a_fsa and b_fsa.
    '''
    # invert a_fsa so that we can intersect with labels
    saved_a_fsa = a_fsa.invert_()

    a_fsa = arc_sort(a_fsa)  # nop if a_fsa is already arc-sorted
    b_fsa = arc_sort(b_fsa)

    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(a_fsa.arcs, b_fsa.arcs,
                                                     need_arc_map)

    # Some of entries in a_arc_map and b_arc_map may be -1.
    # The arc_maps are incremented so that every entry is non-negative.
    a_arc_map = a_arc_map.to(torch.int64) + 1
    b_arc_map = b_arc_map.to(torch.int64) + 1

    out_fsa = Fsa.from_ragged_arc(ragged_arc)

    for name, a_value in a_fsa.named_tensor_attr():
        if name == 'aux_labels':
            # aux_label is handled specially
            continue
        if hasattr(b_fsa, name):
            b_value = getattr(b_fsa, name)

            # a_arc_map and b_arc_map have been offset by 1
            # so we need a padding here
            padding = a_value.new_zeros((1, *a_value.shape[1:]))
            a_value = torch.cat((padding, a_value), dim=0)
            b_value = torch.cat((padding, b_value), dim=0)

            assert a_value.dtype == b_value.dtype == torch.float32
            value = a_value.index_select(0, a_arc_map) \
                    + b_value.index_select(0, b_arc_map)
            setattr(out_fsa, name, value)

    # out_fsa.aux_labels = a_fsa.aux_labels if `a_fsa` is a transducer;
    # out_fsa.aux_labels = a.fsa.labels if `a_fsa` is an acceptor
    # we will invert `out_fsa` later.
    if hasattr(a_fsa, 'aux_labels'):
        a_aux_labels = a_fsa.aux_labels
    else:
        # this is an acceptor
        a_aux_labels = a_fsa.labels
    padding = a_aux_labels.new_zeros(1)

    a_aux_labels = torch.cat((padding, a_aux_labels), dim=0)
    out_fsa.aux_labels = a_aux_labels.index_select(0, a_arc_map)

    # out_fsa.labels = b_fsa.aux_labels if `b_fsa` is a transducer
    # out_fsa.labels = b_fsa.labels if `b_fsa` is an acceptor
    # we will invert `out_fsa` later
    if hasattr(b_fsa, 'aux_labels'):
        b_aux_labels = b_fsa.aux_labels
    else:
        # b_fsa is an acceptor
        b_aux_labels = b_fsa.labels
    b_aux_labels = torch.cat((padding, b_aux_labels), dim=0)
    out_fsa.labels = b_aux_labels.index_select(0, b_arc_map)

    # out_fsa.symbols = b_fsa.aux_symbols if b_fsa is a transducer
    # out_fsa.symbols = b_fsa.symbols if b_fsa is an acceptor
    # out_fsa.aux_symbols = a_fsa.aux_symbols if a_fsa is a transducer
    # out_fsa.aux_symbols = a_fsa.symbols if a_fsa is an acceptor
    for name, a_value in a_fsa.named_non_tensor_attr():
        if name not in ('symbols', 'aux_symbols'):
            setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name) and name not in ('symbols',
                                                       'aux_symbols'):
            setattr(out_fsa, name, b_value)

    if hasattr(a_fsa, 'aux_symbols'):
        out_fsa.aux_symbols = a_fsa.aux_symbols
    elif hasattr(a_fsa, 'symbols'):
        out_fsa.aux_symbols = a_fsa.symbols

    if hasattr(b_fsa, 'aux_symbols'):
        out_fsa.symbols = b_fsa.aux_symbols
    elif hasattr(b_fsa, 'symbols'):
        out_fsa.symbols = b_fsa.symbols

    saved_a_fsa.invert_()  # recover saved_a_fsa

    out_fsa.arc_labels_sorted = False
    out_fsa.arc_aux_labels_sorted = False
    out_fsa.invert_()

    return out_fsa


def shortest_distance(fsa: Fsa) -> torch.Tensor:
    '''Return the shortest distance from the start state to the final state
    in the tropical semiring.

    Note:
      It uses the opposite sign. That is, It uses `max` instead of `min`.

    Args:
      fsa:
        The input FSA.
    Returns:
      The sum of scores along the best path.
    '''
    need_arc_map = True
    _, arc_map = _k2.shortest_distance(fsa.arcs, need_arc_map)
    arc_map = arc_map.to(torch.int64)

    return fsa.score[arc_map].sum()


def shortest_path(fsa: Fsa) -> Fsa:
    '''Return the shortest path as a linear FSA from the start state
    to the final state in the tropical semiring.

    Note:
      It uses the opposite sign. That is, It uses `max` instead of `min`.

    Args:
      fsa:
        The input FSA.
    Returns:
      The best path as a linear FSA.
    '''
    need_arc_map = True
    ragged_arc, arc_map, _ = _k2.shortest_path(fsa.arcs, need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)
    out_fsa.arc_labels_sorted = True
    return out_fsa
