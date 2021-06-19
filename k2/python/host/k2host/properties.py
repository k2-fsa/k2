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
from _k2host import _is_valid
from _k2host import _is_top_sorted
from _k2host import _is_arc_sorted
from _k2host import _has_self_loops
from _k2host import _is_acyclic
from _k2host import _is_deterministic
from _k2host import _is_epsilon_free
from _k2host import _is_connected
from _k2host import _is_empty


def is_valid(fsa: Fsa) -> bool:
    return _is_valid(fsa.get_base())


def is_top_sorted(fsa: Fsa) -> bool:
    return _is_top_sorted(fsa.get_base())


def is_arc_sorted(fsa: Fsa) -> bool:
    return _is_arc_sorted(fsa.get_base())


def has_self_loops(fsa: Fsa) -> bool:
    return _has_self_loops(fsa.get_base())


def is_acyclic(fsa: Fsa) -> bool:
    return _is_acyclic(fsa.get_base())


def is_deterministic(fsa: Fsa) -> bool:
    return _is_deterministic(fsa.get_base())


def is_epsilon_free(fsa: Fsa) -> bool:
    return _is_epsilon_free(fsa.get_base())


def is_connected(fsa: Fsa) -> bool:
    return _is_connected(fsa.get_base())


def is_empty(fsa: Fsa) -> bool:
    return _is_empty(fsa.get_base())
