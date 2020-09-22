# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from _k2host import IntArray2Size
from _k2host import _AuxLabels1Mapper
from _k2host import _AuxLabels2Mapper
from _k2host import _FstInverter

from .fsa import Fsa
from .array import IntArray1
from .array import IntArray2

AuxLabels = IntArray2


class AuxLabels1Mapper(_AuxLabels1Mapper):

    def __init__(self, labels_in: AuxLabels, arc_map: IntArray1):
        super().__init__(labels_in.get_base(), arc_map.get_base())

    def get_sizes(self, aux_size: IntArray2Size) -> None:
        return super().get_sizes(aux_size)

    def get_output(self, labels_out: AuxLabels) -> None:
        return super().get_output(labels_out.get_base())


class AuxLabels2Mapper(_AuxLabels2Mapper):

    def __init__(self, labels_in: AuxLabels, arc_map: IntArray2):
        super().__init__(labels_in.get_base(), arc_map.get_base())

    def get_sizes(self, aux_size: IntArray2Size) -> None:
        return super().get_sizes(aux_size)

    def get_output(self, labels_out: AuxLabels) -> None:
        return super().get_output(labels_out.get_base())


class FstInverter(_FstInverter):

    def __init__(self, fsa_in: Fsa, labels_in: AuxLabels):
        super().__init__(fsa_in.get_base(), labels_in.get_base())

    def get_sizes(self, fsa_size: IntArray2Size,
                  aux_size: IntArray2Size) -> None:
        return super().get_sizes(fsa_size, aux_size)

    def get_output(self, fsa_out: Fsa, labels_out: AuxLabels) -> None:
        return super().get_output(fsa_out.get_base(), labels_out.get_base())
