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
from _k2host import DLPackIntArray2
from _k2host import DLPackIntArray1
from _k2host import DLPackStridedIntArray1
from _k2host import DLPackFloatArray1
from _k2host import DLPackDoubleArray1
from _k2host import DLPackLogSumArcDerivs


class IntArray1(DLPackIntArray1):

    def __init__(self, data: torch.Tensor, check_dtype: bool = True):
        if check_dtype:
            assert data.dtype == torch.int32
        self.data = data
        super().__init__(to_dlpack(self.data))

    @staticmethod
    def from_float_tensor(data: torch.Tensor) -> 'IntArray1':
        assert data.dtype == torch.float
        return IntArray1(data, False)

    @staticmethod
    def create_array_with_size(size: int) -> 'IntArray1':
        data = torch.zeros(size, dtype=torch.int32)
        return IntArray1(data)


class StridedIntArray1(DLPackStridedIntArray1):

    def __init__(self, data: torch.Tensor, check_dtype: bool = True):
        if check_dtype:
            assert data.dtype == torch.int32
        self.data = data
        super().__init__(to_dlpack(self.data))

    @staticmethod
    def from_float_tensor(data: torch.Tensor) -> 'StridedIntArray1':
        assert data.dtype == torch.float
        return StridedIntArray1(data, False)


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
    def create_arc_derivs_with_size(array_size: IntArray2Size
                                   ) -> 'LogSumArcDerivs':
        indexes = torch.zeros(array_size.size1 + 1, dtype=torch.int32)
        data = torch.zeros([array_size.size2, 2], dtype=torch.float32)
        return LogSumArcDerivs(indexes, data)
