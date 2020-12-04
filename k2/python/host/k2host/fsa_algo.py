# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from .fsa import Fsa
from .array import IntArray1
from .array import IntArray2
from .array import FloatArray1
from .array import LogSumArcDerivs
from .weights import WfsaWithFbWeights
from _k2host import IntArray2Size
from _k2host import _ArcSorter
from _k2host import _arc_sort
from _k2host import _TopSorter
from _k2host import _Connection
from _k2host import _Intersection
from _k2host import _DeterminizerPrunedMax
from _k2host import _DeterminizerPrunedLogSum
from _k2host import _EpsilonsRemoverPrunedMax
from _k2host import _EpsilonsRemoverPrunedLogSum


class ArcSorter(_ArcSorter):

    def __init__(self, fsa_in: Fsa):
        super().__init__(fsa_in.get_base())

    def get_sizes(self, array_size: IntArray2Size) -> None:
        return super().get_sizes(array_size)

    def get_output(self, fsa_out: Fsa, arc_map: IntArray1 = None) -> None:
        return super().get_output(
            fsa_out.get_base(),
            arc_map.get_base() if arc_map is not None else None)


def arc_sort(fsa: Fsa, arc_map: IntArray1 = None) -> None:
    return _arc_sort(fsa.get_base(),
                     arc_map.get_base() if arc_map is not None else None)


class TopSorter(_TopSorter):

    def __init__(self, fsa_in: Fsa):
        super().__init__(fsa_in.get_base())

    def get_sizes(self, array_size: IntArray2Size) -> None:
        return super().get_sizes(array_size)

    def get_output(self, fsa_out: Fsa, arc_map: IntArray1 = None) -> bool:
        return super().get_output(
            fsa_out.get_base(),
            arc_map.get_base() if arc_map is not None else None)


class Connection(_Connection):

    def __init__(self, fsa_in: Fsa):
        super().__init__(fsa_in.get_base())

    def get_sizes(self, array_size: IntArray2Size) -> None:
        return super().get_sizes(array_size)

    def get_output(self, fsa_out: Fsa, arc_map: IntArray1 = None) -> bool:
        return super().get_output(
            fsa_out.get_base(),
            arc_map.get_base() if arc_map is not None else None)


class Intersection(_Intersection):

    def __init__(self, fsa_a: Fsa, fsa_b: Fsa):
        super().__init__(fsa_a.get_base(), fsa_b.get_base())

    def get_sizes(self, array_size: IntArray2Size) -> None:
        return super().get_sizes(array_size)

    def get_output(self,
                   fsa_out: Fsa,
                   arc_map_a: IntArray1 = None,
                   arc_map_b: IntArray1 = None) -> bool:
        return super().get_output(
            fsa_out.get_base(),
            arc_map_a.get_base() if arc_map_a is not None else None,
            arc_map_b.get_base() if arc_map_b is not None else None)


class DeterminizerPrunedMax(_DeterminizerPrunedMax):

    def __init__(self, fsa_in: WfsaWithFbWeights, beam: float, max_step: int):
        super().__init__(fsa_in, beam, max_step)

    def get_sizes(self, fsa_size: IntArray2Size,
                  arc_derivs_size: IntArray2Size) -> None:
        return super().get_sizes(fsa_size, arc_derivs_size)

    def get_output(self, fsa_out: Fsa, arc_derivs: IntArray2) -> float:
        return super().get_output(fsa_out.get_base(), arc_derivs.get_base())


class DeterminizerPrunedLogSum(_DeterminizerPrunedLogSum):

    def __init__(self, fsa_in: WfsaWithFbWeights, beam: float, max_step: int):
        super().__init__(fsa_in, beam, max_step)

    def get_sizes(self, fsa_size: IntArray2Size,
                  arc_derivs_size: IntArray2Size) -> None:
        return super().get_sizes(fsa_size, arc_derivs_size)

    def get_output(self, fsa_out: Fsa, arc_derivs: LogSumArcDerivs) -> float:
        return super().get_output(fsa_out.get_base(), arc_derivs.get_base())


class EpsilonsRemoverPrunedMax(_EpsilonsRemoverPrunedMax):

    def __init__(self, fsa_in: WfsaWithFbWeights, beam: float):
        super().__init__(fsa_in, beam)

    def get_sizes(self, fsa_size: IntArray2Size,
                  arc_derivs_size: IntArray2Size) -> None:
        return super().get_sizes(fsa_size, arc_derivs_size)

    def get_output(self, fsa_out: Fsa, arc_derivs: IntArray2) -> None:
        return super().get_output(fsa_out.get_base(), arc_derivs.get_base())


class EpsilonsRemoverPrunedLogSum(_EpsilonsRemoverPrunedLogSum):

    def __init__(self, fsa_in: WfsaWithFbWeights, beam: float):
        super().__init__(fsa_in, beam)

    def get_sizes(self, fsa_size: IntArray2Size,
                  arc_derivs_size: IntArray2Size) -> None:
        return super().get_sizes(fsa_size, arc_derivs_size)

    def get_output(self, fsa_out: Fsa, arc_derivs: LogSumArcDerivs) -> None:
        return super().get_output(fsa_out.get_base(), arc_derivs.get_base())
