# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

import torch


class _SumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse_tensor: torch.Tensor) -> torch.Tensor:
        assert sparse_tensor.is_sparse
        assert sparse_tensor.is_coalesced()

        values = sparse_tensor._values()
        values_sum = values.sum()

        ctx.save_for_backward(sparse_tensor)

        return values_sum

    @staticmethod
    def backward(ctx, ans_grad) -> torch.Tensor:
        sparse_tensor, = ctx.saved_tensors

        indices = sparse_tensor._indices().clone()
        values = sparse_tensor._values()
        size = sparse_tensor.size()

        sparse_tensor_grad_values = torch.ones_like(values) * ans_grad
        sparse_tensor_grad = torch.sparse_coo_tensor(
            indices=indices,
            values=sparse_tensor_grad_values,
            size=size,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device)

        return sparse_tensor_grad


class _AbsFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse_tensor: torch.Tensor) -> torch.Tensor:
        assert sparse_tensor.is_sparse
        assert sparse_tensor.is_coalesced()

        indices = sparse_tensor._indices().clone()
        values = sparse_tensor._values()
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
    def backward(ctx, ans_grad) -> torch.Tensor:
        sparse_tensor, = ctx.saved_tensors

        indices = sparse_tensor._indices().clone()
        values = sparse_tensor._values()
        size = sparse_tensor.size()

        sparse_tensor_grad_values = ans_grad._values() * values.sign()

        sparse_tensor_grad = torch.sparse_coo_tensor(
            indices=indices,
            values=sparse_tensor_grad_values,
            size=size,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device)

        return sparse_tensor_grad


def sum(sparse_tensor: torch.Tensor) -> torch.Tensor:
    '''Compute the sum of a sparse tensor'''
    return _SumFunction.apply(sparse_tensor)


def abs(sparse_tensor: torch.Tensor) -> torch.Tensor:
    '''Compute the abs of a sparse tensor'''
    return _AbsFunction.apply(sparse_tensor)
