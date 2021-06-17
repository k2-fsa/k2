# Copyright      2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def is_rand_equivalent(fsa_a: Fsa,
                       fsa_b: Fsa,
                       treat_epsilons_specially: bool = True,
                       npath: int = 100) -> bool:
    return _is_rand_equivalent(fsa_a.get_base(), fsa_b.get_base(),
                               treat_epsilons_specially, npath)


def is_rand_equivalent_max_weight(fsa_a: Fsa,
                                  fsa_b: Fsa,
                                  beam: float = float('inf'),
                                  treat_epsilons_specially: bool = True,
                                  delta: float = 1e-6,
                                  top_sorted: bool = True,
                                  npath: int = 100) -> bool:
    return _is_rand_equivalent_max_weight(fsa_a.get_base(), fsa_b.get_base(),
                                          beam, treat_epsilons_specially,
                                          delta, top_sorted, npath)


def is_rand_equivalent_logsum_weight(fsa_a: Fsa,
                                     fsa_b: Fsa,
                                     beam: float = float('inf'),
                                     treat_epsilons_specially: bool = True,
                                     delta: float = 1e-6,
                                     top_sorted: bool = True,
                                     npath: int = 100) -> bool:
    return _is_rand_equivalent_logsum_weight(fsa_a.get_base(),
                                             fsa_b.get_base(), beam,
                                             treat_epsilons_specially, delta,
                                             top_sorted, npath)


def is_rand_equivalent_after_rmeps_pruned_logsum(fsa_a: Fsa,
                                                 fsa_b: Fsa,
                                                 beam: float,
                                                 top_sorted: bool = True,
                                                 npath: int = 100) -> bool:
    return _is_rand_equivalent_after_rmeps_pruned_logsum(
        fsa_a.get_base(), fsa_b.get_base(), beam, top_sorted, npath)
