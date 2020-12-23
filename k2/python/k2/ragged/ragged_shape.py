# Copyright (c)  2020  Mobvoi Inc.        (authors: Liyong Guo)
#                      Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)

import _k2
from _k2 import RaggedShape
from _k2 import random_ragged_shape
import torch
from typing import Optional


# These functions are wrapped here for the sake of documentation.
def create_ragged_shape2(row_splits: Optional[torch.Tensor] = None,
                         row_ids: Optional[torch.Tensor] = None,
                         cached_tot_size: Optional[int] = -1
                        ) -> _k2.RaggedShape:  # noqa
    '''Construct a RaggedShape from row_ids and/or row_splits vectors.  For
    the overall concepts, please see comments in k2/csrc/utils.h.

    Args:
      row_splits:
        Optionally, a torch.Tensor with dtype=torch.int32 and one axis
      row_ids:
        Optionally, a torch.Tensor with dtype=torch.int32 and one axis.
      cached_tot_size:
        The number of elements (length of row_ids, even if row_ids
        is not provided); would be identical to the last element of row_splits,
        but can avoid a GPU to CPU transfer if known.
    Returns:
      an instance of _k2.RaggedShape, with ans.num_axes() == 2.
  '''
    return _k2.create_ragged_shape2(row_splits, row_ids, cached_tot_size)


def compose_ragged_shapes(a: _k2.RaggedShape,
                          b: _k2.RaggedShape) -> _k2.RaggedShape:
    '''Compose two RaggedShape objects where a.num_elements() == b.dim0().

    Args:
       a:
         First shape to be composed
       b:
         First shape to be composed; b.dim0() must equal a.num_elements()
    Returns:
      an instance of _k2.RaggedShape with
      `ans.num_axes() == a.num_axes() + b.num_axes() - 1`.
      May share metadata with a and b.
    '''
    return _k2.compose_ragged_shapes(a, b)


# for remove_axis, please see ops.py where we put it because the
# name also applies to ragged.
