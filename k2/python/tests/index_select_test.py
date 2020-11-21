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

        padded_a = torch.cat([torch.tensor([0]).to(a), a])
        expected = padded_a.index_select(0, (b + 1).to(torch.int64))

        assert torch.allclose(c, expected)

        if k2.use_cuda():
            device = torch.device('cuda', 0)
            a = a.to(device)
            b = b.to(device)
            c = k2.index_select(a, b)
            assert c.dtype == a.dtype
            assert c.is_cuda
            assert c.numel() == b.numel()
            assert torch.allclose(c, expected.to(c))

        # now for float32
        a = a.to(torch.float32).requires_grad_(True)
        c = k2.index_select(a, b)
        assert c.dtype == a.dtype
        c.sum().backward()

        new_a = a.detach().requires_grad_(True)
        padded_a = torch.cat([torch.tensor([0]).to(new_a), new_a])
        expected = padded_a.index_select(0, (b + 1).to(torch.int64))
        expected.sum().backward()

        assert torch.allclose(a.grad, new_a.grad)

        # now for cpu
        a.grad = None
        c = k2.index_select(a.cpu(), b.cpu())
        assert c.dtype == a.dtype
        c.sum().backward()

        assert torch.allclose(a.grad, new_a.grad.to(a.grad))

    def test_1d_non_contiguous(self):
        a = torch.arange(20).to(torch.int32)[::2]
        b = torch.tensor([
            -1, -2, -1, -2, 1, -2, 2, -2, 0, -2, 1, -2, 5, -2, 3, -2, 9, -2,
            -1, -2, 8, -2, 7, -2, 7, -2, 6, -2
        ]).to(torch.int32)[::2]
        padded_a = torch.cat([torch.tensor([0]).to(a), a])
        assert a.is_contiguous() is False
        assert b.is_contiguous() is False
        c = k2.index_select(a, b)
        expected = padded_a.index_select(0, (b + 1).to(torch.int64))
        assert torch.allclose(c, expected)

        a = a.to(torch.float32).requires_grad_(True)
        c = k2.index_select(a, b)

        new_a = a.detach().clone().requires_grad_(True)
        padded_a = torch.cat([torch.tensor([0]).to(a), new_a])
        expected = padded_a.index_select(0, (b + 1).to(torch.int64))

        assert torch.allclose(c, expected.to(c))

        c.sum().backward()
        expected.sum().backward()

        assert torch.allclose(a.grad, new_a.grad)

        # now for cuda
        if k2.use_cuda():
            device = torch.device('cuda', 0)
            b = b.to(device)

            a.requires_grad_(False)
            a = a.to(device).requires_grad_(True)
            c = k2.index_select(a, b)

            new_a.requires_grad_(False)
            new_a = new_a.to(device).requires_grad_(True)
            padded_a = torch.cat([torch.tensor([0]).to(a), new_a])
            expected = padded_a.index_select(0, (b + 1).to(torch.int64))

            assert torch.allclose(c, expected.to(c))

            c.sum().backward()
            expected.sum().backward()

            assert torch.allclose(a.grad, new_a.grad)


if __name__ == '__main__':
    unittest.main()
