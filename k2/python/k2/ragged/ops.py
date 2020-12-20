# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors

from typing import Tuple, Optional, List
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


def remove_values_eq(src: _k2.RaggedInt, target: int) -> _k2.RaggedInt:
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
    return _k2.remove_values_eq(src, target)

def remove_axis(src: _k2.RaggedInt, axis: int) -> _k2.RaggedInt:
    '''Remove an axis from a ragged tensor.

    Args:
      src:
        The source ragged tensor.
      axis:
        The axis to remove.  Must satisfy `0 <= axis < src.num_axes() - 1`.
    Returns:
       A new ragged tensor with one fewer axis than `src`.
       The vector of `ans.tot_sizes()` will be the same as `src.tot_sizes()`,
       but with element `axis` removed.
    '''
    return _k2.remove_axis(src, axis)

def to_list(src: _k2.RaggedInt) -> List:
    '''Turn a ragged tensor of ints into a List of Lists [of Lists..] of ints.

    Args:
      src:
        The source ragged tensor.
    Returns:
       A list of list of ints containing the same elements and structure as `src`.
    '''
    return _k2.to_list(src)
