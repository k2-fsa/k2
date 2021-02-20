# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Tuple
from typing import Union
import torch
import k2
import _k2
from .fsa import Fsa
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
                          dtype=out_grad.dtype,
                          device=src.device,
                          requires_grad=False)
        _k2.index_add(index, out_grad, ans)
        return ans, None


class _IndexAndSumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, src: torch.Tensor, indexes: k2.RaggedInt) -> torch.Tensor:
        '''Index a 1-D tensor with a ragged tensor of indexes, perform
        a sum-per-sublist operation, and return the resulting 1-D tensor.

        Note:
          It supports autograd.

        Args:
          src:
            1-D tensor with dtype torch.float32. For example, it can
            be a float tensor attribute of an FSA.
          indexes:
            A ragged tensor with two axes. For example, it can be
            the arc map from :func:`_k2.remove_epsilon`
        Returns:
          1-D torch.Tensor with dtype being `torch.float32`.
        '''
        assert src.ndim == 1
        assert src.dtype == torch.float32
        assert indexes.num_axes() == 2
        ctx.save_for_backward(src)
        ctx.indexes = indexes
        ans = _k2.index_and_sum(src, indexes)
        return ans

    @staticmethod
    def backward(ctx, out_grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        indexes = ctx.indexes
        src, = ctx.saved_tensors
        expanded = _k2.index_select(out_grad, indexes.row_ids(1))
        ans = torch.zeros(src.shape,
                          dtype=torch.float32,
                          device=src.device,
                          requires_grad=False)
        _k2.index_add(indexes.values(), expanded, ans)
        return ans, None


# put index_select here instead of in `auto_grad.py` to break circular import
def index_select(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    '''Returns a new tensor which indexes the input tensor along dimension 0
    using the entries in `index`.

    If the entry in `index` is -1, then the corresponding entry in the
    returned tensor is 0.

    Caution:
      `index.dtype == torch.int32` and `index.ndim == 1`.

    Args:
      src:
        The input tensor. Either 1-D or 2-D with dtype `torch.int32`,
        `torch.int64`, `torch.float32`, or `torch.float64`.
      index:
        1-D tensor of dtype `torch.int32` containing the indexes.
        If an entry is -1, the corresponding entry in the returned value
        is 0. The elements of `index` should be in the range
        `[-1..src.shape[0]-1]`.

    Returns:
      A tensor with shape ``(index.numel(), *src.shape[1:])`` and dtype the
      same as `src`, e.g. if `src.ndim == 1`, `ans.shape` would be
      `(index.shape[0],)`; if `src.ndim == 2`, `ans.shape` would be
      `(index.shape[0], src.shape[1])`.
      Will satisfy `ans[i] == src[index[i]]` if `src.ndim == 1`,
      or `ans[i, j] == src[index[i], j]` if `src.ndim == 2`, except for
      entries where `index[i] == -1` which will be zero.
    '''
    ans = _IndexSelectFunction.apply(src, index)
    return ans


def index_add(index: torch.Tensor, value: torch.Tensor,
              in_out: torch.Tensor) -> None:
    '''It implements in_out[index[i]] += value[i].

    Caution:
      It has similar semantics with `torch.Tensor.index_add_` except
      that:

        - `index.dtype == torch.int32`
        - `-1 <= index[i] < in_out.shape[0]`
        - `index[i] == -1` is ignored.
        - `index` has to be a 1-D **contiguous** tensor.

    Caution:
      `in_out` is modified **in-place**.

    Caution:
      This functions does NOT support autograd.

    Args:
      index:
        A 1-D **contiguous** tensor with dtype `torch.int32`.
        Must satisfy `-1 <= index[i] < in_out.shape[0]`
      value:
        A 1-D or 2-D tensor with dtype `torch.int32`, `torch.float32`,
        or `torch.float64`.
        Must satisfy `index.shape[0] == value.shape[0]`
      in_out:
        A 1-D or 2-D tensor with the same dtype as `value`. It satisfies
        `in_out.shape[1] == value.shape[1]` if it is a 2-D tensor.

    Returns:
      Return None.
    '''

    _k2.index_add(index, value, in_out)


def index_and_sum(src: torch.Tensor, indexes: k2.RaggedInt) -> torch.Tensor:
    '''Index a 1-D tensor with a ragged tensor of indexes, perform
    a sum-per-sublist operation, and return the resulting 1-D tensor.

    Note:
      It supports autograd.

    Args:
      src:
        1-D tensor with dtype torch.float32. For example, it can
        be a float tensor attribute of an FSA.
      indexes:
        A ragged tensor with two axes. For example, it can be
        the arc map from :func:`_k2.remove_epsilon`
    Returns:
      1-D torch.Tensor with dtype being `torch.float32`.
    '''
    return _IndexAndSumFunction.apply(src, indexes)


def index_fsa(src: Fsa, indexes: torch.Tensor) -> Fsa:
    '''Select a list of FSAs from `src` with a 1-D tensor.

    Args:
      src:
        An FsaVec.
      indexes:
        A 1-D `torch.Tensor` of dtype `torch.int32` containing
        the ids of FSAs to select.

    Returns:
      Return an FsaVec containing only those FSAs specified by `indexes`.
    '''
    ragged_arc, value_indexes = k2.ragged.index(src.arcs,
                                                indexes=indexes,
                                                need_value_indexes=True)
    out_fsa = Fsa(ragged_arc)

    for name, value in src.named_tensor_attr():
        setattr(out_fsa, name, k2.ops.index_select(value, value_indexes))

    for name, value in src.named_non_tensor_attr():
        setattr(out_fsa, name, value)

    return out_fsa


def index_ragged(src: _k2.RaggedInt,
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
        i.e. -1 <= indexes[i] < src.dim0(). If indexes[i] is -1, then
        the i-th value of ans is empty.
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
        return _k2.index(src, indexes)


def index_tensor(src: torch.Tensor, indexes: Union[torch.Tensor, _k2.RaggedInt]
                ) -> Union[torch.Tensor, _k2.RaggedInt]:  # noqa
    '''Indexing a 1-D tensor with a 1-D tensor a ragged tensor.

    Args:
      src:
        Source 1-D tensor to index, must have `src.dtype == torch.int32`
        or `src.dtype == torch.float32`.
      indexes:
        It satisfies -1 <= indexes.values()[i] < src.numel().
        - If it's a tensor, its values will be interpreted as indexes into
        `src`; if indexes.values()[i] is -1, then ans[i] is 0.

        - If it's a ragged tensor, `indexes.values()` will be interpreted as
        indexes into `src`. If src.dtype is torch.int32, it returns
        a _k2.RaggedInt; if src.dtype is torch.float32, it performs an extra
        sum-per-sublist operation and returns 1-D torch.Tensor.

    Returns:
      Returns a tensor or a ragged tensor (depending on the type of `indexes`)
    '''
    if isinstance(indexes, torch.Tensor):
        return index_select(src, indexes)
    else:
        assert isinstance(indexes, k2.RaggedInt)
        if src.dtype == torch.int32:
            return _k2.index(src, indexes)
        else:
            assert src.dtype == torch.float32
            return index_and_sum(src, indexes)


def index(src: Union[Fsa, torch.Tensor, _k2.RaggedInt],
          indexes: Union[torch.Tensor, _k2.RaggedInt]
         ) -> Union[Fsa, torch.Tensor, _k2.RaggedInt]:  # noqa
    '''Indexing an Fsa or a 1-D tensor with a tensor or a ragged tensor.
    It's a wrapper of above function `index_fsa`, `index_tensor` and
    `index_ragged`.
    '''
    if isinstance(src, Fsa):
        # currently we only support index Fsa with a tensor.
        assert isinstance(indexes, torch.Tensor)
        return index_fsa(src, indexes)
    elif isinstance(src, torch.Tensor):
        return index_tensor(src, indexes)
    else:
        assert isinstance(src, _k2.RaggedInt)
        return index_ragged(src, indexes)
