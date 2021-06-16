# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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

from typing import Optional
from typing import Union

import torch
import _k2


# Create a class `RaggedFloat` in python for backprop.
#
# TODO(fangjun): wrap methods from _k2.RaggedFloat if needed.
class RaggedFloat(object):
    '''A ragged float tensor.

    It is a wrapper of :class:`_k2.RaggedFloat`, whose purpose
    is to implement autograd for :class:`_k2.RaggedFloat`.

    Currently, it is used only in `k2.ragged.normalize_scores`.
    '''

    def __init__(self,
                 ragged: Union[str, _k2.RaggedFloat, _k2.RaggedShape],
                 values: Optional[torch.Tensor] = None):
        '''Construct an instance of :class:`k2.RaggedFloat`.

        Args:
          ragged:
            It can be one of the following types:

                - A string. Example value::

                    [ [1 2] [] [5 10 20] ]

                - An instance of :class:`_k2.RaggedFloat`

                - An instance of :class:`_k2.RaggedShape`. In this case, you
                  have to provide the additional argument `values`.
          values:
            Required only when `ragged` is an instance of
            :class:`_k2.RaggedShape`. It is a 1-D torch.Tensor with dtype
            torch.float32.
        '''
        if isinstance(ragged, str):
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

    def __str__(self) -> str:
        return str(self.ragged)

    @property
    def values(self) -> torch.Tensor:
        '''Return the underlying array as a 1-D torch.Tensor.
        '''
        return self._values

    @property
    def grad(self) -> torch.Tensor:
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

    def to(self, device: Union[torch.device, str]) -> 'RaggedFloat':
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
        if device == self.values.device:
            return self

        ragged_shape = self.ragged.shape().to(device)
        values = self.values.to(device)

        return RaggedFloat(ragged_shape, values)
