import torch
from typing import Optional

from .tensor import Tensor


def tensor(data: any, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Return a k2.ragged.Tensor with two axes.

    Args:
      data:
        A list-of-list of integers or real numbers.
      dtype:
        Optional. If None, it infers the dtype from `data`
        automatically, which is either `torch.int32` or
        `torch.float32.
    Returns:
      Return a k2 ragged tensor with two axes.
    """
    pass
