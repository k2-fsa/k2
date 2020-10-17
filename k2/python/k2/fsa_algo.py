# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from copy import deepcopy
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
    return Fsa.from_ragged_arc(ragged_arc)


def top_sort(fsa: Fsa) -> Fsa:
    '''Sort an FSA topologically.

    Note:
      It returns a new FSA. The input FSA is NOT changed.

    Args:
      fsa:
        The input fsa to be sorted. It can be either a single FSA
        or a vector of FSAs.
    Returns:
      It returns a single FSA if the input is a single FSA; it returns
      a vector of FSAs if the input is a vector of FSAs.
    '''
    need_arc_map = True
    ragged_arc, arc_map = _k2.top_sort(fsa.arcs, need_arc_map=need_arc_map)
    arc_map = arc_map.to(torch.int64)  # required by index_select
    sorted_fsa = Fsa.from_ragged_arc(ragged_arc)
    sorted_fsa.score = fsa.score.index_select(0, arc_map)
    for name, value in fsa.named_tensor_attr():
        setattr(sorted_fsa, name, value.index_select(0, arc_map))
    for name, value in fsa.named_non_tensor_attr():
        setattr(sorted_fsa, name, deepcopy(value))
    return sorted_fsa
