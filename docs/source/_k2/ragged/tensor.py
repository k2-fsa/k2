from typing import Optional, overload

import torch
from k2.ragged import RaggedShape


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

    @overload
    def __init__(self, s: str, dtype: Optional[torch.dtype] = None) -> None:
        """Create a ragged tensor from its string representation.

        An example string for a 2-axis ragged tensor is given below::

            [ [1]  [2] ]

        An example string for a 3-axis ragged tensor is given below::

            [ [[1] [2 3]]  [[2] [] [3, 4,]] ]

        Note::
          Number of spaces in `s` does not affect the result.
          Of course, numbers have to be separated by at least one space.
        Args:
          s:
            A string representation of the tensor.
          dtype:
            The desired dtype of the tensor. If it's ``None``, it tries
            to infer the correct dtype from `s`, which is assumed to be
            either ``torch.int32`` or ``torch.float32``.
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
        """Change if autograd should record operations on this tensor: Set
        this tensor's :attr:`requires_grad` attribute **in-place**.

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

    @property
    def is_cuda(self) -> bool:
        """
        Returns:
          Return ``True`` if the tensor is stored on the GPU, ``False``
          otherwise.
        """
        pass

    @property
    def shape(self) -> RaggedShape:
        """
        Returns:
          Return the shape of this tensor.
        """
        pass

    def numel(self) -> int:
        """
        Returns:
          Return number of elements in this tensor. It equals to
          `self.data.numel()
        """
        pass

    def num_axes(self) -> int:
        """
        Returns:
          Return number of axes of this tensor, which is at least 2.
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

    def __getitem__(self, i) -> "Tensor":
        """Select the i-th sublist along axis 0.

        Caution:
          Support for autograd is to be implemented.

        Note::
          It requires that this tensor has at least 3 axes.

        Args:
          i:
            The i-th sublist along axis 0.
        Returns:
          Return a new ragged tensor with one fewer axis.
        """
        pass

    def __getstate__(
        self,
    ) -> Union[
        Tuple[torch.Tensor, str, torch.Tensor],
        Tuple[torch.Tensor, str, torch.Tensor, str, torch.Tensor],
    ]:
        """Requires a tensor with 2 axes or 3 axes. Other number
        of axes are not implemented yet.

        This method is to support ``pickle``, e.g., used by torch.save().
        You are not expected to call it by yourself.

        Returns:
          If this tensor has 2 axes, return a tuple containing
          (self.row_splits(1), "row_ids1", self.data).
          If this tensor has 3 axes, return a tuple containing
          (self.row_splits(1), "row_ids1", self.row_splits(1),
          "row_ids2", self.data)

        Note::
          "row_ids1" and "row_ids2" in the returned value is for
          backward compatibility.
        """
        pass

    def __setstate__(self, arg0: tuple):
        """Set the content of this class from ``arg0``.

        This method is to support ``pickle``, e.g., used by torch.load().
        You are not expected to call it by yourself.

        Args:
          arg0:
            It is the return value from the method ``__getstate__``.
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
