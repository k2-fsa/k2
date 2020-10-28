# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Union
from typing import List

from .fsa import Fsa
from .fsa_properties import is_arc_sorted
from .fsa_properties import is_accessible
from .fsa_properties import is_coaccessible
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

    # Delete the properties. Users should set it explicitly
    if hasattr(sorted_fsa, 'properties'):
        del sorted_fsa.properties

    return sorted_fsa


def intersect(a_fsa: Fsa, b_fsa: Fsa) -> Fsa:
    '''Compute the intersection of two FSAs.

    Note:
      We will arc_sort the input FSAs internally if they are not
      arc sorted.

    Args:
      a_fsa:
        The first input FSA.
      b_fsa:
        The second input FSA.

    Caution:
      We support only inputs containing a single FSA at present.

    Caution:
      The rules for assigning the attributes of the output Fsa are as follows:

      - (1) For attributes where only one source (a_fsa or b_fsa) has that
      attribute: Copy via arc_map, or use zero if arc_map has -1. This rule
      works for both floating point and integer attributes.

      - (2) For attributes where both sources (a_fsa and b_fsa) have that
      attribute: For floating point attributes: sum via arc_maps, or use zero
      if arc_map has -1. For integer attributes, it's not supported for now (the
      attributes will be discarded and will not be kept in the output FSA).

    Returns:
      The result of intersecting a_fsa and b_fsa.
    '''
    a_properties = getattr(a_fsa, 'properties', None)
    if a_properties is None or not is_arc_sorted(a_properties):
        a_fsa = arc_sort(a_fsa)

    b_properties = getattr(b_fsa, 'properties', None)
    if b_properties is None or not is_arc_sorted(b_properties):
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
        if hasattr(b_fsa, name):
            # Both a_fsa and b_fsa have this attribute.
            # We only support attributes with dtype ``torch.float32``.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                continue
            b_value = getattr(b_fsa, name)
            assert b_value.dtype == torch.float32

            # a_arc_map and b_arc_map have been offset by 1
            # so we need a padding here
            padding = a_value.new_zeros((1, *a_value.shape[1:]))
            a_value = torch.cat((padding, a_value), dim=0)
            b_value = torch.cat((padding, b_value), dim=0)

            value = a_value.index_select(0, a_arc_map) \
                    + b_value.index_select(0, b_arc_map)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            padding = a_value.new_zeros((1, *a_value.shape[1:]))
            a_value = torch.cat((padding, a_value), dim=0)
            value = a_value.index_select(0, a_arc_map)
            setattr(out_fsa, name, value)

    # now copy tensor attributes that are in b_fsa but are not in a_fsa
    for name, b_value in b_fsa.named_tensor_attr():
        if not hasattr(out_fsa, name):
            padding = b_value.new_zeros((1, *b_value.shape[1:]))
            b_value = torch.cat((padding, b_value), dim=0)
            value = b_value.index_select(0, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

    if hasattr(out_fsa, 'properties'):
        del out_fsa.properties

    return out_fsa


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
    properties = getattr(fsa, 'properties', None)
    if properties is not None \
            and is_accessible(properties) \
            and is_coaccessible(properties):
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.connect(fsa.arcs, need_arc_map=need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    if hasattr(out_fsa, 'properties'):
        del out_fsa.properties

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
    properties = getattr(fsa, 'properties', None)
    if properties is not None and is_arc_sorted(properties):
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.arc_sort(fsa.arcs, need_arc_map=need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    if hasattr(out_fsa, 'properties'):
        del out_fsa.properties

    return out_fsa


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

    if hasattr(out_fsa, 'properties'):
        del out_fsa.properties

    return out_fsa
