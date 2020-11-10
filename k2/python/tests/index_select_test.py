#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R index_select_test_py

import unittest

import k2
import torch


class TestIndexSelect(unittest.TestCase):

    def test_1d(self):
        a = torch.tensor([
            10,  # 0
            -1,  # 1
            100,  # 2
            0,  # 3
            3,  # 4
            9,  # 5
            12,  # 6
        ]).to(torch.int32)

        b = torch.tensor([0, -1, 0, 2, 1, 3, 6, -1, 5, 6, 0,
                          2]).to(torch.int32)

        c = k2.index_select(a, b)
        assert c.dtype == a.dtype
        assert c.numel() == b.numel()
        assert torch.allclose(
            c,
            torch.tensor([10, 0, 10, 100, -1, 0, 12, 0, 9, 12, 10,
                          100]).to(torch.int32))

        device = torch.device('cuda', 0)
        a = a.to(device)
        b = b.to(device)
        c = k2.index_select(a, b)
        assert c.dtype == a.dtype
        assert c.is_cuda
        assert c.numel() == b.numel()
        assert torch.allclose(
            c,
            torch.tensor([10, 0, 10, 100, -1, 0, 12, 0, 9, 12, 10, 100]).to(c))

        # now for float32
        a = a.to(torch.float32).requires_grad_(True)
        c = k2.index_select(a, b)
        assert c.dtype == a.dtype
        c.sum().backward()
        assert torch.allclose(a.grad,
                              torch.tensor([3, 1, 2, 1, 0, 1, 2.]).to(a.grad))

        # now for cpu
        a.grad = None
        c = k2.index_select(a.cpu(), b.cpu())
        assert c.dtype == a.dtype
        c.sum().backward()
        assert torch.allclose(a.grad,
                              torch.tensor([3, 1, 2, 1, 0, 1, 2.]).to(a.grad))


if __name__ == '__main__':
    unittest.main()
