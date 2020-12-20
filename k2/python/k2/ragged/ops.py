# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors

from typing import Tuple, Optional
from typing import Union
import torch
import _k2


def index(src: Union[_k2.RaggedArc, _k2.RaggedInt],
          indexes: torch.Tensor,
          need_value_indexes: bool = True
         ) -> Tuple[Union[_k2.RaggedArc, _k2.RaggedInt],  # noqa
                    Optional[torch.Tensor]]:  # noqa
    '''Indexing operation on ragged tensor, returns src[indexes], where
    the elements of `indexes` are interpreted as indexes into axis 0 of
    `src`.

    Caution:
      `indexes` is a 1-D tensor and `indexes.dtype == torch.int32`.

    Args:
      src:
        Source ragged tensor to index.
      indexes:
        Array of indexes, which will be interpreted as indexes into axis 0
        of `src`, i.e. with 0 <= indexes[i] < src.dim0().
      need_value_indexes:
        If true, it will return a torch.Tensor containing the indexes into
        `src.values()` that `ans.values()` has,
        as in `ans.values() = src.values()[value_indexes]`.

    Returns:
      Return a tuple containing:
       - `ans` of type `_k2.RaggedArc` or `_k2.RaggedInt` (same as the type
         of `src`).
       - None if `need_value_indexes` is False; a 1-D torch.tensor of
         dtype `torch.int32` containing the indexes into `src.values()` that
         `ans.values()` has.
    '''
    ans, value_indexes = _k2.index(src=src,
                                   indexes=indexes,
                                   need_value_indexes=need_value_indexes)
    return ans, value_indexes


def remove_values_leq(src: _k2.RaggedInt, cutoff: int) -> _k2.RaggedInt:
    '''Remove values less than or equal to `cutoff` from a ragged tensor.

    Args:
      src:
        The source ragged tensor.
      cutoff:
        The threshold. Elements less than or equal to this threshold is removed
        from `src`.
    Returns:
      A new ragged tensor whose elements are all **greater than** `cutoff`.
    '''
    return _k2.remove_values_leq(src, cutoff)


def remove_values_equal(src: _k2.RaggedInt, target: int) -> _k2.RaggedInt:
    '''Remove values equal to `target` from a ragged tensor.

    Args:
      src:
        The source ragged tensor.
      target:
        The target value. Elements whose value equal to `target` are removed
        from `src`.
    Returns:
      A new ragged tensor whose elements do **not equal to** `target`.
    '''
    return _k2.remove_values_equal(src, target)
