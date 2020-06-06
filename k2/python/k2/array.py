# Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from _k2 import _IntArray1
from _k2 import _FloatArray1


class IntArray1(_IntArray1):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(to_dlpack(tensor))


class FloatArray1(_FloatArray1):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(to_dlpack(tensor))


class Array:
    # TODO(fangjun): wrap IntArray1 inside Array
    pass
