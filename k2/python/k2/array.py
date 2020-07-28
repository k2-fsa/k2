# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from _k2 import IntArray2Size
from _k2 import DLPackIntArray2
from _k2 import DLPackIntArray1
from _k2 import DLPackFloatArray1
from _k2 import DLPackDoubleArray1
from _k2 import DLPackLogSumArcDerivs


class IntArray1(DLPackIntArray1):

    def __init__(self, data: torch.Tensor):
        assert data.dtype == torch.int32
        self.data = data
        super().__init__(to_dlpack(self.data))

    @staticmethod
    def create_array_with_size(size: int) -> 'IntArray1':
        data = torch.zeros(size, dtype=torch.int32)
        return IntArray1(data)


class FloatArray1(DLPackFloatArray1):

    def __init__(self, data: torch.Tensor):
        assert data.dtype == torch.float
        self.data = data
        super().__init__(to_dlpack(self.data))

    @staticmethod
    def create_array_with_size(size: int) -> 'FloatArray1':
        data = torch.zeros(size, dtype=torch.float)
        return FloatArray1(data)


class DoubleArray1(DLPackDoubleArray1):

    def __init__(self, data: torch.Tensor):
        assert data.dtype == torch.double
        self.data = data
        super().__init__(to_dlpack(self.data))

    @staticmethod
    def create_array_with_size(size: int) -> 'DoubleArray1':
        data = torch.zeros(size, dtype=torch.double)
        return DoubleArray1(data)


class IntArray2(DLPackIntArray2):

    def __init__(self, indexes: torch.Tensor, data: torch.Tensor):
        assert indexes.dtype == torch.int32
        assert data.dtype == torch.int32
        self.indexes = indexes
        self.data = data
        super().__init__(to_dlpack(self.indexes), to_dlpack(self.data))

    @staticmethod
    def create_array_with_size(array_size: IntArray2Size) -> 'IntArray2':
        indexes = torch.zeros(array_size.size1 + 1, dtype=torch.int32)
        data = torch.zeros(array_size.size2, dtype=torch.int32)
        return IntArray2(indexes, data)


class LogSumArcDerivs(DLPackLogSumArcDerivs):

    def __init__(self, indexes: torch.Tensor, data: torch.Tensor):
        assert indexes.dtype == torch.int32
        assert data.dtype == torch.float32
        assert data.shape[1] == 2
        self.indexes = indexes
        self.data = data
        super().__init__(to_dlpack(self.indexes), to_dlpack(self.data))

    @staticmethod
    def create_arc_derivs_with_size(
            array_size: IntArray2Size) -> 'LogSumArcDerivs':
        indexes = torch.zeros(array_size.size1 + 1, dtype=torch.int32)
        data = torch.zeros([array_size.size2, 2], dtype=torch.float32)
        return LogSumArcDerivs(indexes, data)
