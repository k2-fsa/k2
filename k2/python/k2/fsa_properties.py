# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
#
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

import torch  # noqa
import _k2

# The FSA properties are a bit-field; these constants can be used
# with '&' to determine the properties.

VALID = 0x01  # Valid from a formatting perspective
NONEMPTY = 0x02  # Nonempty as in, has at least one arc.
TOPSORTED = 0x04,  # FSA is top-sorted, but possibly with
# self-loops, dest_state >= src_state
TOPSORTED_AND_ACYCLIC = 0x08  # Fsa is topsorted, dest_state > src_state
ARC_SORTED = 0x10  # Fsa is arc-sorted: arcs leaving a state are are sorted by
# label first and then on `dest_state`, see operator< in
# struct Arc in /k2/csrc/fsa.h (Note: labels are treated as
# uint32 for purpose of sorting!)

ARC_SORTED_AND_DETERMINISTIC = 0x20  # Arcs leaving a given state are *strictly*
# sorted by label, i.e. no duplicates with
# the same label.
EPSILON_FREE = 0x40  # Label zero (epsilon) is not present..
ACCESSIBLE = 0x80  # True if there are no obvious signs
# of states not being accessible or
# co-accessible, i.e. states with no
# arcs entering them
COACCESSIBLE = 0x0100  # True if there are no obvious signs of
# states not being co-accessible, i.e.
# i.e. states with no arcs leaving them
ALL = 0x01FF


def to_str(p: int) -> str:
    '''Convert properties to a string for debug purpose.

    Args:
      p:
        An integer returned by :func:`get_properties`.

    Returns:
      A string representation of the input properties.
    '''
    return _k2.fsa_properties_as_str(p)
