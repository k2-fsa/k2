#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R sparse_sum_test_py

import unittest

import k2.sparse
import torch


class TestSparseSum(unittest.TestCase):

    def test_no_repeat(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            row_indexes = torch.tensor([0, 0, 1, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            values = torch.tensor([1, 2, -3, 4],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = k2.sparse.sum(sparse_tensor)
            assert s.item() == 4

            scale = 2
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)

    def test_with_repeats(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            row_indexes = torch.tensor([0, 0, 1, 2, 0, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1, 0, 1]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            values = torch.tensor([1, 2, -3, 4, 2, 6],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = k2.sparse.sum(sparse_tensor)
            assert s.item() == 12
            scale = -3
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)


if __name__ == '__main__':
    unittest.main()
