# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang, Wei Kang)
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import List, Optional, Union

import torch
import _k2


class Ragged(object):
    '''A ragged tensor.

    It is a wrapper of :class:`_k2.RaggedFloat` and :class:`_k2.RaggedInt`
    '''

    def __init__(self, ragged: Union[_k2.RaggedInt, _k2.RaggedFloat],
                       requires_grad: bool = False):
        '''Construct an instance of :class:`k2.Ragged` from _k2.RaggedInt or
        _k2.RaggedFloat.

        Args:
          ragged:
            It can be an instance of :class:`_k2.RaggedFloat` or an instance of
            :class:`_k2.RaggedInt`.
        '''
        assert isinstance(ragged, _k2.RaggedFloat) or\
               isinstance(ragged, _k2.RaggedInt)

        if requires_grad:
            assert isinstance(ragged, _k2.RaggedFloat)

        self.ragged = ragged
        self._values = ragged.values()
        self.requires_grad_(requires_grad)

    def __str__(self) -> str:
        return str(self.ragged)

    def __eq__(self, other: 'Ragged') -> bool:
        return self.ragged == other.ragged

    def __ne__(self, other: 'Ragged') -> bool:
        return self.ragged != other.ragged

    def __getstate__(self):
        '''Custom the behavior of pickling.

        This function will be called when pickling this instance, and the
        returned object is pickled as the contents for the instance, we only
        pickle out `self.ragged` for efficiency.
        '''
        return self.ragged

    def __setstate__(self, state):
        '''Restore this instance from pickled state.
        '''
        self.ragged = state
        self._values = self.ragged.values()

    @property
    def grad(self) -> torch.Tensor:
        '''This attribute is None by default and becomes a Tensor the first
        time a call to backward() computes gradients for this tensor.
        The attribute will then contain the gradients computed and future calls
        to backward() will accumulate (add) gradients into it.
        '''
        return self._values.grad

    @property
    def requires_grad(self) -> bool:
        '''
        Return True if this object requires grad.
        Return False otherwise.
        '''
        return self._values.requires_grad

    def requires_grad_(self, requires_grad: bool) -> 'RaggedFloat':
        '''Change if autograd should record operations on this tensor.

        Sets the `values`'s requires_grad attribute in-place.
        Returns this object.
        You can test whether this object has the requires_grad property
        true or false by accessing self.requires_grad property.

        Caution:
          This is an **in-place** operation as you can see that the function
          name ends with `_`.

        Args:
          requires_grad:
            If autograd should record operations on this object or not.

        Returns:
          This object itself.
        '''
        self._values.requires_grad_(requires_grad)
        return self

    def clone(self) -> 'Ragged':
        return Ragged(self.ragged.clone())

    def to(self, device: Union[torch.device, str]) -> 'Ragged':
        '''Move the RaggedFloat onto a given device.

        Args:
          device:
            An instance of `torch.device` or a string that can be used to
            construct a `torch.device`, e.g., 'cpu', 'cuda:0'.
            It supports only cpu and cuda devices.

        Returns:
          Returns a new RaggedFloat which is this object copied to the given
          device (or this object itself, if the device was the same).
        '''
        if isinstance(device, str):
            device = torch.device(device)

        assert device.type in ('cpu', 'cuda')
        if device == self.values().device:
            return self

        return Ragged(self.ragged.to(device))

    def values(self) -> torch.Tensor:
        '''Return the underlying array as a 1-D torch.Tensor.
        '''
        return self._values

    @property
    def is_cpu(self) -> bool:
        '''Is True if the Ragged is stored on the CPU, False otherwise.'''
        return self.ragged.is_cpu()

    @property
    def is_cuda(self) -> bool:
        '''Is True if the Ragged is stored on the CPU, False otherwise.'''
        return self.ragged.is_cuda()

    @property
    def shape(self) -> _k2.RaggedShape:
        '''Return the shape of the Ragged.'''
        return self.ragged.shape()

    def dim0(self) -> int:
        '''Return the elements number of axis 0.'''
        return self.ragged.dim0()

    def num_elements(self) -> _k2.RaggedShape:
        '''Return the elements number of axis `num_axes() - 1`'''
        return self.ragged.num_elements()

    def num_axes(self) -> int:
        '''Return the axis number of the Ragged.'''
        return self.ragged.num_axes()

    def tot_size(self, axis: int) -> int:
        '''Return the elements number of the given axis'''
        return self.ragged.tot_size(axis)

    def tot_sizes(self) -> List:
        '''Return the elements numbers of all axises in a list'''
        return self.ragged.tot_sizes()

    def row_ids(self, axis: int) -> torch.Tensor:
        '''Return the row_ids of given axis.'''
        return self.ragged.row_ids(axis)

    def row_splits(self, axis: int) -> torch.Tensor:
        '''Return the row_splits of given axis.'''
        return self.ragged.row_splits(axis)

    def arange(self, axis: int, begin: int, end: int) -> 'Ragged': 
        return Ragged(self.ragged.arange(axis, begin, end))

    def max_per_sublist(self, initial_value: float) -> torch.Tensor:
        return _k2.max_per_sublist(self.ragged, initial_value)

    def min_per_sublist(self, initial_value: float) -> torch.Tensor:
        return _k2.min_per_sublist(self.ragged, initial_value)

    def logsum_per_sublist(self, initial_value: float) -> torch.Tensor:
        return _k2.logsum_per_sublist(self.ragged, initial_value)

    def argmax_per_sublist(self, initial_value: float) -> torch.Tensor:
        return _k2.argmax_per_sublist(self.ragged, initial_value)

    def sort_sublists(self, descending: bool = False,
            need_new2old_indexes: bool = False) -> Optional[torch.Tensor]:
        return _k2.sort_sublists(self.ragged, descending, need_new2old_indexes)

    def remove_axis(self, axis: int) -> 'Ragged':
        return Ragged(_k2.remove_axis(self.ragged, axis))


class RaggedFloat(Ragged):
    '''A ragged float tensor.

    It is a wrapper of :class:`_k2.RaggedFloat`, which supports autograd.
    '''

    def __init__(self,
                 ragged: Union[str, List[List[float]],
                               _k2.RaggedFloat, _k2.RaggedShape],
                 values: Optional[torch.Tensor] = None,
                 requires_grad: bool = False):
        '''Construct an instance of :class:`k2.RaggedFloat`.

        Args:
          ragged:
            It can be one of the following types:

                - A string. Example value::

                    [ [1 2] [] [5 10 20] ]

                - An instance of :class:`_k2.RaggedFloat`

                - An instance of :class:`_k2.RaggedShape`. In this case, you
                  have to provide the additional argument `values`.

                - An instance of List[List[float]]. Only support RaggedFloat
                  with `num_axes = 2` in this case.
          values:
            Required only when `ragged` is an instance of
            :class:`_k2.RaggedShape`. It is a 1-D torch.Tensor with dtype
            torch.float32.
        '''
        if isinstance(ragged, str) or isinstance(ragged, list):
            ragged = _k2.RaggedFloat(ragged)
            assert values is None
        elif isinstance(ragged, _k2.RaggedShape):
            assert values is not None
            ragged = _k2.RaggedFloat(ragged, values)

        assert isinstance(ragged, _k2.RaggedFloat)

        self.ragged = ragged
        if values is not None:
            self._values = values
        else:
            self._values = ragged.values()

        self.requires_grad_(requires_grad)


class RaggedInt(Ragged):
     def __init__(self,
                 ragged: Union[str, List[List[int]],
                               _k2.RaggedInt, _k2.RaggedShape],
                 values: Optional[torch.Tensor] = None):
        '''Construct an instance of :class:`k2.RaggedInt`.

        Args:
          ragged:
            It can be one of the following types:

                - A string. Example value::

                    [ [1 2] [] [5 10 20] ]

                - An instance of :class:`_k2.RaggedInt`

                - An instance of :class:`_k2.RaggedShape`. In this case, you
                  have to provide the additional argument `values`.

                - An instance of List[List[int]]. Only support RaggedFloat
                  with `num_axes = 2` in this case.
          values:
            Required only when `ragged` is an instance of
            :class:`_k2.RaggedShape`. It is a 1-D torch.Tensor with dtype
            torch.int32.
        '''
        if isinstance(ragged, str) or isinstance(ragged, list):
            ragged = _k2.RaggedInt(ragged)
            assert values is None
        elif isinstance(ragged, _k2.RaggedShape):
            assert values is not None
            ragged = _k2.RaggedInt(ragged, values)

        assert isinstance(ragged, _k2.RaggedInt)

        self.ragged = ragged
        if values is not None:
            self._values = values
        else:
            self._values = ragged.values()

