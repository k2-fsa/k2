# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from .fsa import Fsa
from _k2 import _is_valid
from _k2 import _is_top_sorted
from _k2 import _is_arc_sorted
from _k2 import _has_self_loops
from _k2 import _is_acyclic
from _k2 import _is_deterministic
from _k2 import _is_epsilon_free
from _k2 import _is_connected
from _k2 import _is_empty


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
