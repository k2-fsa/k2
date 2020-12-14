# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Tuple
from typing import Union
import torch
import _k2
from .ragged import index as ragged_index


class _IndexSelectFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        '''Returns a new tensor which indexes the input tensor along dimension 0
        using the entries in `index`.

        If the entry in `index` is -1, then the corresponding entry in the
        returned tensor is 0.

        Caution:
          `index.dtype == torch.int32` and `index.ndim == 1`.

        Args:
          src:
            The input tensor. Either 1-D or 2-D with dtype torch.int32 or
            torch.float32.
          index:
            1-D tensor of dtype torch.int32 containing the indexes.
            If an entry is -1, the corresponding entry in the returned value
            is 0. The elements of `index` should be in the range
            `[-1..src.shape[0]-1]`.

        Returns:
          A tensor with shape (index.numel(), *src.shape[1:]) and dtype the
          same as `src`, e.g. if `src.ndim == 1`, ans.shape would be
          (index.shape[0],); if `src.ndim == 2`, ans.shape would be
          (index.shape[0], src.shape[1]).
          Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
          or `ans[i,j] == src[index[i],j]` if `src.ndim == 2`, except for
          entries where `index[i] == -1` which will be zero.
        '''
        ctx.save_for_backward(src, index)
        return _k2.index_select(src, index)

    @staticmethod
    def backward(ctx, out_grad) -> Tuple[torch.Tensor, None]:
        src, index = ctx.saved_tensors

        ans = torch.zeros(src.shape,
                          dtype=torch.float32,
                          device=src.device,
                          requires_grad=False)
        _k2.index_add(index, out_grad, ans)
        return ans, None


# put index_select stead of `auto_grad.py` here to break circular import
def index_select(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    '''Returns a new tensor which indexes the input tensor along dimension 0
    using the entries in `index`.

    If the entry in `index` is -1, then the corresponding entry in the
    returned tensor is 0.

    Caution:
      `index.dtype == torch.int32` and `index.ndim == 1`.

    Args:
      src:
        The input tensor. Either 1-D or 2-D with dtype torch.int32 or
        torch.float32.
      index:
        1-D tensor of dtype torch.int32 containing the indexes.
        If an entry is -1, the corresponding entry in the returned value
        is 0. The elements of `index` should be in the range
        `[-1..src.shape[0]-1]`.

    Returns:
      A tensor with shape (index.numel(), *src.shape[1:]) and dtype the
      same as `src`, e.g. if `src.ndim == 1`, ans.shape would be
      (index.shape[0],); if `src.ndim == 2`, ans.shape would be
      (index.shape[0], src.shape[1]).
      Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
      or `ans[i,j] == src[index[i],j]` if `src.ndim == 2`, except for
      entries where `index[i] == -1` which will be zero.
    '''
    ans = _IndexSelectFunction.apply(src, index)
    return ans


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
        CAUTION: It has to be contiguous.
      value:
        A 1-D or 2-D tensor with dtype torch.float32 or torch.float32.
        index.shape[0] == value.shape[0]
      in_out:
        A 1-D or 2-D tensor with the same dtype as `value`. It satisfies
        in_out.shape[1] == value.shape[1]

    Returns:
      Return None.
    '''

    _k2.index_add(index, value, in_out)


def index_ragged_int(src: _k2.RaggedInt,
                     indexes: Union[torch.Tensor, _k2.RaggedInt]
                    ) -> _k2.RaggedInt:  # noqa
    '''Indexing ragged tensor with a 1-D tensor or a ragged tensor.

    Args:
      src:
        Source ragged tensor to index; must have num_axes() == 2.
      indexes:
        It can be a tensor or a ragged tensor.
        If it's a tensor, it must be a 1-D tensor and
        `indexes.dtype == torch.int32`.
        Values in it will be interpreted as indexes into axis 0 of `src`,
        i.e. 0 <= indexes[i] < src.dim0().
        If it's a ragged tensor, `indexes.values` will be interpreted as
        indexes into axis 0 of `src`, i.e. 0 <= indexes.values[i] < src.dim0();
        Must have num_axes() == 2.

    Returns:
      Return the indexed ragged tensor with ans.num_axes() == 2
       - If `indexes` is a 1-D tensor, then ans.dim0() == indexes.numel().
       - If `indexes` is a ragged tensor, then ans.dim0() = indexes.dim0().
    '''
    if isinstance(indexes, torch.Tensor):
        ans, _ = ragged_index(src, indexes)
        return ans
    else:
        return _k2.index_ragged_with_ragged_int(src, indexes)


def index_tensor_with_ragged_int(src: torch.Tensor,
                                 indexes: _k2.RaggedInt) -> _k2.RaggedInt:
    '''Indexing a 1-D tensor with a ragged tensor.

    Args:
      src:
        Source 1-D tensor to index, must have `src.dtype == torch.int32`
      indexes:
        A ragged tensor, `indexes.values` will be interpreted as indexes
        into `src`.
        i.e. 0 <= indexes.values[i] < src.numel();

    Returns:
      Returns ragged tensor with shape `indexes.shape` and
      values `src[indexes.values]`.
    '''
    return _k2.index_tensor_with_ragged_int(src, indexes)


def index_tensor(src: torch.Tensor, indexes: Union[torch.Tensor, _k2.RaggedInt]
                ) -> Union[torch.Tensor, _k2.RaggedInt]:  # noqa
    '''It's a wrapper of index_tensor and index_ragged_int above

    Args:
      src:
        Source 1-D tensor to index, must have `src.dtype == torch.int32`
      indexes:
        If it's a ragged tensor, `indexes.values` will be interpreted as
        indexes into `src`.
        i.e. 0 <= indexes.values[i] < src.numel();
        If it's a tensor, its values will be interpreted as indexes into `src`.

    Returns:
      Returns a tensor or a ragged tensor (depending on the type of `indexes`)
    '''
    if isinstance(indexes, torch.Tensor):
        return index_select(src, indexes)
    else:
        # TODO(haowen): it does not autograd now.
        return index_tensor_with_ragged_int(src, indexes)


def index_attr(src: Union[torch.Tensor, _k2.RaggedInt],
               indexes: Union[torch.Tensor, _k2.RaggedInt]
              ) -> Union[torch.Tensor, _k2.RaggedInt]:  # noqa
    '''Indexing a 1-D tensor with a tensor a ragged tensor, it's a wrapper
    of index_tensor_with_ragged_int and index_select.
    '''
    if isinstance(src, torch.Tensor):
        return index_tensor(src, indexes)
    else:
        return index_ragged_int(src, indexes)
