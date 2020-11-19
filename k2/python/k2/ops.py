# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

import torch
import _k2


from .autograd import index_select
from .fsa import Fsa
from .ragged import index as ragged_index


def index(src: Fsa, indexes: torch.Tensor) -> Fsa:
    '''Select a list of FSAs from `src`.

    Args:
      src:
        An FsaVec.
      indexes:
        A 1-D torch.Tensor of dtype `torch.int32` containing
        the ids of FSAs to select.

    Returns:
      Return an FsaVec containing only those FSAs specified by `indexes`.
    '''
    ragged_arc, value_indexes = ragged_index(src.arcs,
                                             indexes=indexes,
                                             need_value_indexes=True)
    out_fsa = Fsa(ragged_arc)

    for name, value in src.named_tensor_attr():
        setattr(out_fsa, name, index_select(value, value_indexes))

    for name, value in src.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def index_add(index: torch.Tensor, value: torch.Tensor,
              in_out: torch.Tensor) -> None:
    '''It implements in_out[index[i]] += value[i].

    Caution:
      It has similar semantics with `torch.Tensor.index_add_` except
      that (1) index.dtype == torch.int32; (2) -1 <= index[i] < in_out.shape[0].
      index[i] == -1 is ignored.

    Caution:
      `in_out` is modified **in-place**.

    Caution:
      This functions does NOT support autograd.

    Args:
      index:
        A 1-D tensor with dtype torch.int32.  -1 <= index[i] < in_out.shape[0]
      value:
        A 1-D tensor with dtype torch.float32. index.numel() == value.numel()
      in_out:
        A 1-D tensor with dtype torch.float32.

    Returns:
      Return None.
    '''

    _k2.index_add(index, value, in_out)
