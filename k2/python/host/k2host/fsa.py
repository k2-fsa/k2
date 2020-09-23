# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

# See ../../../LICENSE for clarification regarding multiple authors

import torch
from torch.utils.dlpack import to_dlpack

from _k2host import IntArray2Size
from _k2host import _Arc
from _k2host import DLPackFsa
from _k2host import IntArray2Size


class Arc(_Arc):

    def __init__(self, src_state: int, dest_state: int, label: int,
                 weight: float):
        super().__init__(src_state, dest_state, label, weight)

    def to_tensor(self):
        # TODO(fangjun): weight will be truncted to an int.
        return torch.tensor(
            [self.src_state, self.dest_state, self.label, self.weight],
            dtype=torch.int32)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'Arc':
        assert tensor.shape == torch.Size([4])
        assert tensor.dtype == torch.int32
        return Arc(*tensor.tolist())


class Fsa(DLPackFsa):
    """
    Corresponds to k2::Fsa class, initializes k2::Fsa with torch.Tensors.

    Note that we view each row of self.data as a k2::Arc, usually users 
    can convert the type between tensor and k2::Arc by calling 
    `k2.Arc.from_tensor()` and `k2.Arc.to_tensor()`.
    If users want to change the values in Fsa, just call `fsa.data[i] = some_tensor`.
    
    """

    def __init__(self, indexes: torch.Tensor, data: torch.Tensor):
        assert indexes.dtype == torch.int32
        assert data.dtype == torch.int32
        assert data.shape[1] == 4
        self.indexes = indexes
        self.data = data
        super().__init__(to_dlpack(self.indexes), to_dlpack(self.data))

    @staticmethod
    def create_fsa_with_size(array_size: IntArray2Size) -> 'Fsa':
        indexes = torch.zeros(array_size.size1 + 1, dtype=torch.int32)
        data = torch.zeros([array_size.size2, 4], dtype=torch.int32)
        return Fsa(indexes, data)
