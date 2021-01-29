# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

import torch


class _AbsFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse_tensor: torch.Tensor) -> torch.Tensor:
        '''Compute the `abs` of a sparse tensor.

        Args:
          sparse_tensor:
            A sparse tensor. It has to satisfy::

                assert sparse_tensor.is_coalesced()

        Returns:
          The absolute value of the sparse tensor.
          The `abs` operation is applied element-wise.
        '''
        assert sparse_tensor.is_sparse
        assert sparse_tensor.is_coalesced()

        indices = sparse_tensor.indices().clone()
        values = sparse_tensor.values()
        size = sparse_tensor.size()

        values_abs = values.abs()

        ans = torch.sparse_coo_tensor(indices=indices,
                                      values=values_abs,
                                      size=size,
                                      dtype=sparse_tensor.dtype,
                                      device=sparse_tensor.device)

        ctx.save_for_backward(sparse_tensor)
        return ans

    @staticmethod
    def backward(ctx, ans_grad: torch.Tensor) -> torch.Tensor:
        sparse_tensor, = ctx.saved_tensors

        indices = sparse_tensor.indices().clone()
        values = sparse_tensor.values()
        size = sparse_tensor.size()

        sparse_tensor_grad_values = ans_grad.values() * values.sign()

        sparse_tensor_grad = torch.sparse_coo_tensor(
            indices=indices,
            values=sparse_tensor_grad_values,
            size=size,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device)

        return sparse_tensor_grad


def abs(sparse_tensor: torch.Tensor) -> torch.Tensor:
    '''Compute the `abs` of a sparse tensor.

    It supports autograd.

    Args:
      sparse_tensor:
        A sparse tensor. It has to satisfy::

            assert sparse_tensor.is_coalesced()

    Returns:
      The absolute value of the sparse tensor.
      The `abs` operation is applied element-wise.
    '''
    return _AbsFunction.apply(sparse_tensor)
