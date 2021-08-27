from typing import Optional, overload

import torch


class Tensor(object):
    @overload
    def __init__(self):
        """Create an empty ragged tensor."""
        pass

    @overload
    def __init__(self, data: list, dtype: Optional[torch.dtype] = None) -> None:
        """Create a ragged tensor with two axes.

        Args:
          data:
            A list-of-list of integers or real numbers.
          dtype:
            Optional. If None, it infers the dtype from `data`
            automatically, which is either `torch.int32` or
            `torch.float32.
        """
        pass

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of this tensor."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the device of this tensor."""
        pass

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying memory as a 1-D tensor."""
        pass

    @property
    def requires_grad(self) -> bool:
        """Return ``True`` if gradients need to be computed for this tensor.
        Return ``False`` otherwise.
        """
        pass

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        """Set the requires grad attribute of this tensor."""
        pass

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Change if autograd should record operations on this tensor: sets
        this tensor's :attr:`requires_grad` attribute **in-place**. Returns
        this tensor.

        Note::
          If this tensor is not a float tensor, PyTorch will throw a
          RuntimeError exception.

        Caution:
          This method ends with an underscore, meaning it changes this tensor
          **in-place**.

        Args:
          requires_grad:
            If autograd should record operations on this tensor.
        Returns:
          Return this tensor.
        """
        pass

    @property
    def grad(self) -> torch.Tensor:
        """This attribute is ``None`` by default. PyTorch will set it
        during ``backward()``.

        The attribute will contain the gradients computed and future
        calls to ``backward()`` will accumulate (add) gradients into it.
        """
        pass

    def sum(self, initial_value: float = 0) -> torch.Tensor:
        """Compute the sum of sublists over the last axis of this tensor.

        Note::
          If a sublist is empty, the sum for it is the provided
          ``initial_value``.

        Note::
          This operation supports autograd if this tensor is a float tensor,
          i.e., with dtype being torch.float32 or torch.float64.

        Args:
          initial_value:
            This value is added to the sum of each sublist. So when
            a sublist is empty, its sum is this value.
        Returns:
          Return a 1-D tensor with the same dtype of this tensor
          containing the computed sum.
        """

    def __str__(self) -> str:
        """Return a string representation of this tensor"""
        pass

    def __eq__(self, other: "Tensor") -> bool:
        """Compare two ragged tensors.

        Caution::
          The two tensors MUST have the same dtype. Otherwise,
          it throws.

        Args:
          other:
            The tensor to be compared.
        Returns:
          Return True if the two tensors are equal.
          Return False otherwise.

        """
        pass

    def __ne__(self, other: "Tensor") -> bool:
        """Compare two ragged tensors.

        Caution::
          The two tensors MUST have the same dtype. Otherwise,
          it throws.

        Args:
          other:
            The tensor to be compared.
        Returns:
          Return True if the two tensors are NOT equal.
          Return False otherwise.

        """
        pass

    def clone(self) -> "Tensor":
        """Return a copy of this tensor."""
        pass

    @overload
    def to(self, o: torch.device) -> "Tensor":
        """Transfer this tensor to a given device.

        Note::
          If `self` is already on the specified device, return a
          ragged tensor sharing the underlying memory with `self`.
          Otherwise, a new tensor is returned.

        Args:
          o:
            The target device to move this tensor.

        Returns:
          Return a tensor on the given device.
        """
        pass

    @overload
    def to(self, o: torch.dtype) -> "Tensor":
        """Convert this tensor to a specific dtype.

        Note::
          If `self` is already of the specified `dtype`, return
          a ragged tensor sharing the underlying memory with `self`.
          Otherwise, a new tensor is returned.

        Caution::
          Currently, only for dtypes torch.int32, torch.float32, and
          torch.float64 are implemented. We can support other types
          if needed.

        Args:
          o:
            The `dtype` this tensor should be converted to.

        Returns:
          Return a tensor of the given `dtype`.
        """
        pass
