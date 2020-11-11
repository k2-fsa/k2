# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

import torch

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
    out_fsa = Fsa.from_ragged_arc(ragged_arc)
    value_indexes = value_indexes.to(torch.int64)

    for name, value in src.named_tensor_attr():
        setattr(out_fsa, name, value.index_select(0, value_indexes))

    for name, value in src.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa
