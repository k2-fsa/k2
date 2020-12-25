# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
# See ../../../../LICENSE for clarification regarding multiple authors

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
    '''

    def __init__(self, ragged: Union[str, _k2.RaggedFloat]):
        if isinstance(ragged, str):
            ragged = _k2.RaggedFloat(ragged)

        self.ragged = ragged
        self._scores = ragged.values()

    def __str__(self) -> str:
        return str(self.ragged)

    @property
    def scores(self) -> torch.Tensor:
        '''Return the underlying array as a 1-D torch.Tensor.
        '''
        return self._scores

    @property
    def grad(self) -> torch.Tensor:
        return self._scores.grad

    @property
    def requires_grad(self) -> bool:
        '''
        Return True if this object requires grad.
        Return False otherwise.
        '''
        return self._scores.requires_grad

    def requires_grad_(self, requires_grad: bool) -> 'RaggedFloat':
        '''Change if autograd should record operations on this tensor.

        Sets the `scores`'s requires_grad attribute in-place.
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
        self._scores.requires_grad_(requires_grad)
        return self
