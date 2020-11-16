# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Union
from typing import List

import torch
import _k2


from .autograd import index_select
from .fsa import Fsa
from .fsa_properties import is_accessible
from .fsa_properties import is_arc_sorted
from .fsa_properties import is_coaccessible


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
    sorted_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(sorted_fsa, name, index_select(value, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(sorted_fsa, name, value)

    return sorted_fsa


def intersect(a_fsa: Fsa, b_fsa: Fsa) -> Fsa:
    '''Compute the intersection of two FSAs on CPU.

    Args:
      a_fsa:
        The first input FSA on CPU. It can be either a single FSA or a FsaVec.
      b_fsa:
        The second input FSA on CPU. it can be either a single FSA or a FsaVec.

    Caution:
      The two input FSAs MUST be arc sorted.

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
    treat_epsilons_specially = True
    need_arc_map = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(a_fsa.arcs, b_fsa.arcs,
                                                     treat_epsilons_specially,
                                                     need_arc_map)

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

            value = index_select(a_value, a_arc_map) \
                    + index_select(b_value, b_arc_map)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            value = index_select(a_value, a_arc_map)
            setattr(out_fsa, name, value)

    # now copy tensor attributes that are in b_fsa but are not in a_fsa
    for name, b_value in b_fsa.named_tensor_attr():
        if not hasattr(out_fsa, name):
            value = index_select(b_value, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

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
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index_select(value, arc_map))
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
    properties = getattr(fsa, 'properties', None)
    if properties is not None and is_arc_sorted(properties):
        return fsa

    need_arc_map = True
    ragged_arc, arc_map = _k2.arc_sort(fsa.arcs, need_arc_map=need_arc_map)
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index_select(value, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def shortest_path(fsa: Fsa, use_float_scores: bool) -> Fsa:
    '''Return the shortest paths as linear FSAs from the start state
    to the final state in the tropical semiring.

    Note:
      It uses the opposite sign. That is, It uses `max` instead of `min`.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or a FsaVec.
      use_float_scores:
        True to use float, i.e., single precision floating point, for scores.
        False to use double.

    Returns:
          FsaVec, it contains the best paths as linear FSAs
    '''
    entering_arcs = fsa.get_entering_arcs(use_float_scores)
    ragged_arc, ragged_int = _k2.shortest_path(fsa.arcs, entering_arcs)
    out_fsa = Fsa.from_ragged_arc(ragged_arc)

    arc_map = ragged_int.values()
    for name, value in fsa.named_tensor_attr():
        setattr(out_fsa, name, index_select(value, arc_map))

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def add_epsilon_self_loops(fsa: Fsa) -> Fsa:
    '''Add epsilon self-loops to an Fsa or FsaVec.

    This is required when composing using a composition method that does not
    treat epsilons specially, if the other FSA has epsilons in it.

    Args:
      fsa:
        The input FSA. It can be either a single FSA or a FsaVec.

    Returns:
      An instance of :class:`Fsa` that has an epsilon self-loop on every
      non-final state.
    '''

    need_arc_map = True
    ragged_arc, arc_map = _k2.add_epsilon_self_loops(fsa.arcs,
                                                     need_arc_map=need_arc_map)

    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    for name, value in fsa.named_tensor_attr():
        new_value = index_select(value, arc_map)
        setattr(out_fsa, name, new_value)

    for name, value in fsa.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa
