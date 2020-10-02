# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Union

import torch

from _k2 import _ArcArray1
from _k2 import _FloatArray1
from _k2 import _FloatArray2
from _k2 import _Int32Array1
from _k2 import _Int32Array2


def _to_arc_array1(tensor: torch.Tensor) -> _ArcArray1:
    return _ArcArray1.from_tensor(tensor)


def _to_float_array1(tensor: torch.Tensor) -> _FloatArray1:
    return _FloatArray1.from_tensor(tensor)


def _to_float_array2(tensor: torch.Tensor) -> _FloatArray2:
    return _FloatArray2.from_tensor(tensor)


def _to_int32_array1(tensor: torch.Tensor) -> _Int32Array1:
    return _Int32Array1.from_tensor(tensor)


def _to_int32_array2(tensor: torch.Tensor) -> _Int32Array2:
    return _Int32Array2.from_tensor(tensor)


ArrayT = Union[_ArcArray1, _FloatArray1, _FloatArray2, _Int32Array1,
               _Int32Array2]


def _from_tensor(tensor: torch.Tensor, arcs: bool = False) -> ArrayT:
    '''Return an `Array` sharing memory with the passed `torch.Tensor`.
    '''
    data: ArrayT
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
            if arcs:
                data = _to_arc_array1(tensor)
            else:
                data = _to_int32_array2(tensor)
        elif tensor.dtype == torch.float32:
            data = _to_float_array2(tensor)
        else:
            # TODO(fangjun): support other data types
            raise ValueError(f'Unsupported dtype {tensor.dtype}')
    else:
        raise ValueError(
            f'Supported dimensions are 1 and 2. Given {tensor.ndim}')

    return data


class Array(object):
    '''This class wraps k2::Array1<T> and k2::Array2<T> from C++.

    It has only one method `tensor()` which returns a `torch.Tensor`.
    '''

    def __init__(self,
                 data: Union[torch.Tensor, _ArcArray1, _FloatArray1,
                             _FloatArray2, _Int32Array1, _Int32Array2],
                 arcs: bool = False) -> None:
        '''Construct an `Array` from a `torch.Tensor` or from one of
        `k2::Array1<T>` and `k2::Array2<T>`.

        Args:
          data:
            An instance of Array1<T>, Array2<T> or a `torch.Tensor`.
          arcs:
            True if `data` represents arcs information; we will check that
            `data` has 4 columns and is of type `torch.int32`.
        '''
        if isinstance(data, torch.Tensor):
            self.data = _from_tensor(data, arcs)
        elif isinstance(data, (_ArcArray1, _FloatArray1, _FloatArray2,
                               _Int32Array1, _Int32Array2)):
            self.data = data
        else:
            raise ValueError(f'Unsupported type {type(data)}')

    def tensor(self) -> torch.Tensor:
        '''Return a `torch.Tensor` sharing memory with the underlying `Array`.
        '''
        return self.data.tensor()
