# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from .fsa import Fsa
from .array import IntArray1
from .array import IntArray2
from _k2 import _ArcSorter
from _k2 import _arc_sort


class ArcSorter(_ArcSorter):

    def __init__(self, fsa_in: Fsa):
        super().__init__(fsa_in.get_base())

    def get_sizes(self, array_size: IntArray2):
        super().get_sizes(array_size)

    def get_output(self, fsa_out: Fsa, arc_map: IntArray1 = None):
        super().get_output(fsa_out.get_base(),
                           arc_map.get_base() if arc_map is not None else None)


def arc_sort(fsa: Fsa, arc_map: IntArray1 = None):
    return _arc_sort(fsa.get_base(),
                     arc_map.get_base() if arc_map is not None else None)
