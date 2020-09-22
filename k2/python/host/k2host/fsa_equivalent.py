# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from .fsa import Fsa
from .array import IntArray1
from .array import FloatArray1
from _k2host import IntArray2Size
from _k2host import _RandPath
from _k2host import _is_rand_equivalent
from _k2host import _is_rand_equivalent_max_weight
from _k2host import _is_rand_equivalent_logsum_weight
from _k2host import _is_rand_equivalent_after_rmeps_pruned_logsum


class RandPath(_RandPath):

    def __init__(self, fsa_in: Fsa, no_eps_arc: bool, eps_arc_tries: int = 50):
        super().__init__(fsa_in.get_base(), no_eps_arc, eps_arc_tries)

    def get_sizes(self, array_size: IntArray2Size) -> None:
        return super().get_sizes(array_size)

    def get_output(self, fsa_out: Fsa, arc_map: IntArray1 = None) -> bool:
        return super().get_output(
            fsa_out.get_base(),
            arc_map.get_base() if arc_map is not None else None)


def is_rand_equivalent(fsa_a: Fsa, fsa_b: Fsa, npath: int = 100) -> bool:
    return _is_rand_equivalent(fsa_a.get_base(), fsa_b.get_base(), npath)


def is_rand_equivalent_max_weight(fsa_a: Fsa,
                                  fsa_b: Fsa,
                                  beam: float = float('inf'),
                                  delta: float = 1e-6,
                                  top_sorted: bool = True,
                                  npath: int = 100) -> bool:
    return _is_rand_equivalent_max_weight(fsa_a.get_base(), fsa_b.get_base(),
                                          beam, delta, top_sorted, npath)


def is_rand_equivalent_logsum_weight(fsa_a: Fsa,
                                     fsa_b: Fsa,
                                     beam: float = float('inf'),
                                     delta: float = 1e-6,
                                     top_sorted: bool = True,
                                     npath: int = 100) -> bool:
    return _is_rand_equivalent_logsum_weight(fsa_a.get_base(),
                                             fsa_b.get_base(), beam, delta,
                                             top_sorted, npath)


def is_rand_equivalent_after_rmeps_pruned_logsum(fsa_a: Fsa,
                                                 fsa_b: Fsa,
                                                 beam: float,
                                                 top_sorted: bool = True,
                                                 npath: int = 100) -> bool:
    return _is_rand_equivalent_after_rmeps_pruned_logsum(
        fsa_a.get_base(), fsa_b.get_base(), beam, top_sorted, npath)
