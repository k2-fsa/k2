# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang
#                                                   Daniel Povey
#                                                   Haowen Qiu
#                                                   Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

try:
    from typing import Literal  # for Python >= 3.8
except ImportError:
    from typing_extensions import Literal  # for python < 3.8

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


def pad(src: Union[_k2.RaggedInt, _k2.RaggedFloat],
        mode: Literal['constant', 'replicate'] = 'constant',
        value: Union[int, float] = 0) -> torch.Tensor:
    '''Pad a ragged tensor to a torch tensor.

    For example, if `src` has the following values::

        [ [1 2 3] [4] [5 6 7 8] ]

    Then it returns a 2-D tensor as follows if value is 0 and
    mode is `constant`::

        tensor([[1, 2, 3, 0],
                [4, 0, 0, 0],
                [5, 6, 7, 8]])

    Args:
      src:
        The source ragged tensor, MUST have `num_axes() == 2`.
      mode:
        Valid values are: `constant`, `replicate`. If it is `constant`, the
        given `value` is used for filling. If it is `replicate`,
        the last entry in a list is used for filling. If a list is empty,
        then the given `value` is also used for filling.
      value:
        The filling value.

    Returns:
      A 2-D torch.Tensor whose dtype is torch.int32 if the input is
      _k2.RaggedInt tensor or torch.float32 if the input is _k2.RaggedFloat
      tensor. The tensor returned is on the same device as `src`.
    '''
    assert mode in ['constant', 'replicate'], f'mode: {mode}'
    return _k2.pad_ragged(src, mode, value)


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


def cat(srcs: List[_k2.RaggedInt], axis=0) -> _k2.RaggedInt:
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
    return _k2.cat(srcs, axis)


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


def unique_sequences(
        src: _k2.RaggedInt,
        need_num_repeats: bool = True,
        need_new2old_indexes: bool = False) -> \
                Tuple[_k2.RaggedInt, Optional[_k2.RaggedInt], Optional[torch.Tensor]]:  # noqa
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

    Caution:
      Even if there are no repeated sequences, the output may be different
      from `src`. That is, `new2old_indexes` may NOT be an identity map even if
      nothing was removed.

    Args:
      src:
        The input ragged tensor. Must have `src.num_axes() == 2`
        or `src_num_axes() == 3`
      need_num_repeats:
        If True, it also returns the number of repeats of each sequence.
      need_new2old_indexes:
        If true, it returns an extra 1-D tensor `new2old_indexes`.
        If `src` has 2 axes, this tensor contains `src_idx0`;
        if `src` has 3 axes, this tensor contains `src_idx01`.

        Caution:
          For repeated sublists, only one of them is kept.
          The choice of which one to keep is **deterministic** and
          is an implementation detail.


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

       - new2old_indexes: A 1-D tensor whose i-th element specifies the
         input sublist that the i-th output sublist corresponds to.
    '''
    return _k2.unique_sequences(src,
                                need_num_repeats=need_num_repeats,
                                need_new2old_indexes=need_new2old_indexes)


def regular_ragged_shape(dim0: int, dim1: int) -> _k2.RaggedShape:
    '''Returns a RaggedShape with dim0() == dim0 and tot_size(1) == dim0 * dim1.

    Require dim0 >= 0 and dim1 >= 0.

    Caution:
      The returned ragged shape is on CPU. You can use its `to` method to move
      it to GPU.

    Args:
      dim0:
        It equals to ans.dim0()
      dim1:
        It equals to ans.tot_size(1) / dim0
    Returns:
      Return a ragged shape with 2 axes.
    '''
    return _k2.regular_ragged_shape(dim0, dim1)


def argmax_per_sublist(src: Union[_k2.RaggedFloat, _k2.RaggedInt],
                       initial_value: float = torch.finfo(torch.float32).min
                      ) -> torch.Tensor:  # noqa
    '''Compute the argmax per sublist for a ragged tensor.

    The argmax is computed on the last axis.

    Args:
      src:
        The input ragged tensor.
      initial_value:
        The initial value used to compute the argmax.
    Returns:
      Return a 1-D tensor with dtype torch.int32.
    '''
    return _k2.argmax_per_sublist(src, initial_value)


def max_per_sublist(src: Union[_k2.RaggedFloat, _k2.RaggedInt],
                    initial_value: float = torch.finfo(torch.float32).min
                   ) -> torch.Tensor:  # noqa
    '''Compute the max per sublist for a ragged tensor (including
    `initial_value` in the maximum)

    The max is computed on the last layer, ignoring other layers, so it's
    as if you removed other layers first.

    Args:
      src:
        The input ragged tensor.
      initial_value:
        The initial value that is included with the elements of each
        sub-list when computing the maximum.
    Returns:
      Return a 1-D tensor with dtype torch.int32.
    '''
    return _k2.max_per_sublist(src, initial_value)


def sort_sublist(in_out: Union[_k2.RaggedFloat, _k2.RaggedInt],
                 descending: bool = False,
                 need_new2old_indexes: bool = False) -> Optional[torch.Tensor]:
    '''Sort a ragged tensor **in-place**.

    Args:
      in_out:
        The ragged tensor to be sorted. The sort operation is applied to
        the last axis. Caution: It is sorted **in-place**.
      descending:
        True to sort in descending order. False to sort in ascending order.
      need_new2old_indexes:
        If True, also returns a 1-D tensor, which contains the indexes mapping
        from the sorted elements to the unsorted elements. We can use
        `in_out.clone().values()[ans_tensor]` to get a sorted tensor.
    Returns:
      If `need_new2old_indexes` is False, returns None. Otherwise, returns
      a 1-D tensor of dtype torch.int32.
    '''
    return _k2.sort_sublists(in_out=in_out,
                             descending=descending,
                             need_new2old_indexes=need_new2old_indexes)
