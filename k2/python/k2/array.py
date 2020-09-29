# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Union

import torch

from _k2 import _ArcArray1
from _k2 import _FloatArray1
from _k2 import _Int32Array1


def _to_arc_array1(tensor: torch.Tensor) -> _ArcArray1:
    return _ArcArray1.from_tensor(tensor)


def _to_float_array1(tensor: torch.Tensor) -> _FloatArray1:
    return _FloatArray1.from_tensor(tensor)


def _to_int32_array1(tensor: torch.Tensor) -> _Int32Array1:
    return _Int32Array1.from_tensor(tensor)


Array1 = Union[_ArcArray1, _FloatArray1, _Int32Array1]


def _from_tensor(tensor: torch.Tensor) -> Array1:
    '''Return an `Array` sharing memory with the passed `torch.Tensor`.
    '''
    data: Array1
    if tensor.ndim == 1:
        if tensor.dtype == torch.int32:
            data = _to_int32_array1(tensor)
        elif tensor.dtype == torch.float:
            data = _to_float_array1(tensor)
        else:
            # TODO(fangjun): support other data types
            raise ValueError(f'Unsupported dtype {tensor.dtype}')
    elif tensor.ndim == 2:
        if tensor.dtype == torch.int32:
            # FIXME(fangjun): how to distinguish it
            # from normal Array2<int32_t> ?
            data = _to_arc_array1(tensor)
    else:
        # TODO(fangjun): support Array2
        raise ValueError(f'Unsupported dimension {tensor.ndim}')
    return data


class Array(object):
    '''This class wraps k2::Array1<T> and k2::Array2<T> from C++.

    It has only one method `tensor()` which returns a `torch.Tensor`.
    '''

    def __init__(
            self,
            data: Union[torch.Tensor, _ArcArray1, _FloatArray1, _Int32Array1]
    ) -> None:
        '''Construct an `Array` from a `torch.Tensor` or from one of
        `k2::Array1<T>` and `k2::Array2<T>`.
        '''
        if isinstance(data, torch.Tensor):
            self.data = _from_tensor(data)
        elif isinstance(data, (_ArcArray1, _FloatArray1, _Int32Array1)):
            self.data = data
        else:
            raise ValueError(f'Unsupported type {type(data)}')

    def tensor(self) -> torch.Tensor:
        '''Return a `torch.Tensor` sharing memory with the underlying `Array`.
        '''
        return self.data.tensor()
