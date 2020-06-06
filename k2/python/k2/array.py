# Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from _k2 import _ArcArray1
from _k2 import _ArcArray2
from _k2 import _FloatArray1
from _k2 import _IntArray1


class IntArray1(_IntArray1):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(to_dlpack(tensor))


class FloatArray1(_FloatArray1):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(to_dlpack(tensor))


class ArcArray1(_ArcArray1):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(to_dlpack(tensor))


class ArcArray2(_ArcArray2):

    def __init__(self, indexes: torch.Tensor, data: torch.Tensor):
        super().__init__(to_dlpack(indexes), to_dlpack(data))


class Array:
    # TODO(fangjun): wrap IntArray1 inside Array
    pass
