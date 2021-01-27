#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R sparse_abs_test_py

import unittest

import k2.sparse
import torch


class TestSparseAbs(unittest.TestCase):

    def test_no_repeats(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            row_indexes = torch.tensor([0, 0, 1, 2, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1, 0]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            values = torch.tensor([1, -2, -3, 4, 0],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            ans = k2.sparse.abs(sparse_tensor)

            assert ans.is_sparse
            assert torch.all(
                torch.eq(ans._values(),
                         sparse_tensor._values().abs()))

            s = k2.sparse.sum(ans.coalesce())
            assert s.item() == 10
            scale = 2
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().abs().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)

    def test_with_repeats(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            row_indexes = torch.tensor([0, 0, 1, 2, 0, 2, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1, 0, 1, 0]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            # (0, 0): 1 + (-2) = -1
            # (2, 1): 4 + (-6) == -2
            values = torch.tensor([1, 2, -3, 4, -2, -6, 0],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            ans = k2.sparse.abs(sparse_tensor)
            assert ans.is_sparse
            assert torch.all(
                torch.eq(ans._values(),
                         sparse_tensor._values().abs()))

            s = k2.sparse.sum(ans.coalesce())
            assert s.item() == 8
            scale = -3
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().abs().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)

            if False:
                # minus, abs, sum
                values.grad = None
                sparse_tensor1 = torch.sparse_coo_tensor(indexes,
                                                         values=values,
                                                         size=size).coalesce()
                sparse_tensor2 = torch.sparse_coo_tensor(indexes,
                                                         values=values * 2,
                                                         size=size).coalesce()
                sparse_tensor = sparse_tensor2 - sparse_tensor1
                s1 = k2.sparse.sum(
                    k2.sparse.abs(sparse_tensor.coalesce()).coalesce())
                #  s1.backward() # throws an exception

                #  grad1 = values.grad.clone()
                sparse_tensor1 = torch.sparse_coo_tensor(indexes,
                                                         values=values,
                                                         size=size).coalesce()
                sparse_tensor2 = torch.sparse_coo_tensor(indexes,
                                                         values=values * 2,
                                                         size=size).coalesce()
                s2 = (sparse_tensor2 - sparse_tensor1).to_dense().abs()
                #  s2.backward() # throws an exception

                assert s1 == s2
                #  print(grad1, values.grad)


if __name__ == '__main__':
    unittest.main()
