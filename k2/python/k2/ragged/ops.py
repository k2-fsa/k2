# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang
#                                                   Daniel Povey
#                                                   Haowen Qiu)
#
# See ../../../../LICENSE for clarification regarding multiple authors

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import _k2


def index(
        src: Union[_k2.RaggedArc, _k2.RaggedInt, _k2.RaggedShape],
        indexes: torch.Tensor,
        need_value_indexes: bool = True,
        axis: int = 0
) -> Tuple[Union[_k2.RaggedArc, _k2.RaggedInt, _k2.RaggedShape],  # noqa
           Optional[torch.Tensor]]:  # noqa
    '''Indexing operation on ragged tensor, returns src[indexes], where
    the elements of `indexes` are interpreted as indexes into axis `axis` of
    `src`.

    Caution:
      `indexes` is a 1-D tensor and `indexes.dtype == torch.int32`.

    Args:
      src:
        Source ragged tensor or ragged shape to index.
      axis:
        The axis to be indexed. Must satisfy 0 <= axis < src.num_axes()
      indexes:
        Array of indexes, which will be interpreted as indexes into axis `axis`
        of `src`, i.e. with 0 <= indexes[i] < src.tot_size(axis).
        Note that if `axis` is 0, then -1 is also a valid entry in `index`.
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
                                   axis=axis,
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
    return _k2.ragged_int_remove_values_leq(src, cutoff)


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
    return _k2.ragged_int_remove_values_eq(src, target)


def remove_axis(src: Union[_k2.RaggedInt, _k2.RaggedShape],
                axis: int) -> _k2.RaggedInt:
    '''Remove an axis from a ragged tensor.

    Args:
      src:
        The source ragged tensor or ragged shape. Must have `num_axes() > 2`.
      axis:
        The axis to remove.  If src is a _k2.RaggedShape it must satisfy
        `0 <= axis < src.num_axes()`;
        otherwise it must satisfy `0 <= axis < src.num_axes() - 1` (we can't
        remove the last axis in this case as the dimension of the values
        would change).
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
       A list of list of ints containing the same elements and structure
       as `src`.
    '''
    return _k2.ragged_int_to_list(src)


def sum_per_sublist(src: _k2.RaggedFloat,
                    initial_value: float = 0) -> torch.Tensor:
    '''Return the sum of each sublist.

    For example, if `src` has the following values::

        [ [a b] [h j k] [m] ]

    Then it returns a 1-D tensor with 3 entries:

        - entry 0: a + b + initial_value
        - entry 1: h + j + k + initial_value
        - entry 2: m + initial_value

    Args:
      src:
        A ragged float tensor. Note that the sum is performed on the last axis.
    Returns:
      Return a 1-D torch.Tensor with dtype torch.float32. Its `numel` equals to
      `src.tot_size(src.num_axes() - 2)`.
    '''
    return _k2.sum_per_sublist(src, initial_value)


def append(srcs: List[_k2.RaggedInt], axis=0) -> _k2.RaggedInt:
    '''Concatenate a list of :class:`_k2.RaggedInt` along a given axis.

    Args:
      srcs:
        The input.
      axis:
        It can be either 0 or 1.
    Returns:
      A single ragged tensor.
    '''
    assert axis in (0, 1)
    return _k2.append(srcs, axis)


def create_ragged2(vecs: Union[List[List[int]], List[List[float]]]
                  ) -> Union[_k2.RaggedInt, _k2.RaggedFloat]:  # noqa
    '''
    Construct a Ragged with 2 axes.
    Args:
      vecs:
        Input of a list of list
    Returns:
      A single ragged array.
    '''
    return _k2.create_ragged2(vecs)


def get_layer(src: _k2.RaggedShape, layer: int) -> _k2.RaggedShape:
    '''Returns a `sub-shape` of `src`.

    Args:
      src:
        Source RaggedShape.
      layer:
        Layer that is desired, from `0 .. src.num_axes() - 2` (inclusive).

    Returns:
      This returned shape will have `num_axes() == 2`, the minimal case of
      a RaggedShape.
    '''
    return _k2.get_layer(src, layer)


def unique_sequences(src: _k2.RaggedInt, need_num_repeats: bool = True
                    ) -> Tuple[_k2.RaggedInt, Optional[_k2.RaggedInt]]:  # noqa
    '''Remove repeated sequences.

    If `src` has two axes, this will return the unique sub-lists (in a possibly
    different order, but without repeats).  If `src` has 3 axes, it will
    do the above but separately for each index on axis 0; if more than 3 axes,
    the earliest axes will be ignored.

    Caution:
      It does not completely guarantee that all unique sequences will be
      present in the output, as it relies on a hash and ignores collisions.
      If several sequences have the same hash, only one of them is kept, even
      if the actual content in the sequence is different.

    Args:
      src:
        The input ragged tensor. Must have `src.num_axes() == 2`
        or `src_num_axes() == 3`
      need_num_repeats:
        If True, it also returns the number of repeats of each sequence.

    Returns:
     Returns a tuple containing:
       - ans: A ragged tensor with the same number of axes as `src` and possibly
         fewer elements due to removing repeated sequences on the last axis
         (and with the last-but-one indexes possibly in a different order).

       - num_repeats: A tensor containing number of repeats of each returned
         sequence if `need_num_repeats` is True; it is None otherwise. If it is
         not None, num_repeats.num_axes() is always 2. If ans.num_axes() is 2,
         then num_repeats.dim0() == 1 and
         num_repeats.num_elements() == ans.dim0().
         If ans.num_axes() is 3, then num_repeats.dim0() == ans.dim0() and
         num_repeats.num_elements() == ans.tot_size(1).
    '''
    return _k2.unique_sequences(src, need_num_repeats=need_num_repeats)
