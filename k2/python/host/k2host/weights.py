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

from _k2host import IntArray2Size
from _k2host import FbWeightType
from _k2host import _WfsaWithFbWeights

from .fsa import Fsa
from .array import IntArray1
from .array import FloatArray1
from .array import DoubleArray1
from .array import IntArray2


class WfsaWithFbWeights(_WfsaWithFbWeights):

    def __init__(self, fsa: Fsa, weight_type: FbWeightType,
                 forward_state_weights: DoubleArray1,
                 backward_state_weights: DoubleArray1):
        super().__init__(fsa.get_base(), weight_type,
                         forward_state_weights.get_base(),
                         backward_state_weights.get_base())
