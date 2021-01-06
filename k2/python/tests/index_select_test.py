#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (authors: Daniel Povey)
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

        if torch.cuda.is_available():
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
        b = torch.tensor([-1, -1, 1, 2, 0, 1, 5, 3, 9, -1, 8, 7, 7,
                          6]).to(torch.int32)
        padded_a = torch.cat([torch.tensor([0]).to(a), a])
        assert a.is_contiguous() is False
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
        if torch.cuda.is_available():
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

    def test_2d(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            num_rows = torch.randint(1, 2000, size=(1,)).item()
            num_cols = torch.randint(1, 2000, size=(1,)).item()
            a = torch.randint(-1000,
                              1000,
                              size=(num_rows, num_cols),
                              dtype=torch.int32,
                              device=device).contiguous()
            b = torch.randint(-1,
                              num_rows,
                              size=(10000,),
                              dtype=torch.int32,
                              device=device)
            assert a.is_contiguous()
            c = k2.index_select(a, b)
            assert c.dtype == a.dtype
            assert c.device == a.device
            assert c.shape[1] == a.shape[1]
            assert c.shape[0] == b.shape[0]

            padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(a), a])
            expected = padded_a.index_select(0, (b + 1).to(torch.int64))

            # now for float32
            a = a.to(torch.float32).requires_grad_(True)
            assert a.is_contiguous()
            assert b.is_contiguous()
            c = k2.index_select(a, b)

            assert c.dtype == a.dtype
            c.sum().backward()

            new_a = a.detach().requires_grad_(True)
            padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(new_a), new_a])
            expected = padded_a.index_select(0, (b + 1).to(torch.int64))
            expected.sum().backward()

            assert torch.allclose(a.grad, new_a.grad)

    def test_2d_non_contiguous(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            num_rows = torch.randint(20, 2000, size=(1,)).item()
            num_cols = torch.randint(1, 2000, size=(1,)).item()
            stride = torch.randint(2, num_rows // 10 + 1, size=(1,)).item()
            a = torch.randint(-1000,
                              1000,
                              size=(num_rows, num_cols),
                              dtype=torch.int32,
                              device=device).contiguous()
            a = a[::stride]
            num_rows = a.shape[0]
            b = torch.randint(-1,
                              num_rows,
                              size=(10000,),
                              dtype=torch.int32,
                              device=device)
            assert a.is_contiguous() is False
            c = k2.index_select(a, b)
            assert c.dtype == a.dtype
            assert c.device == a.device
            assert c.shape[1] == a.shape[1]
            assert c.shape[0] == b.shape[0]

            padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(a), a])
            expected = padded_a.index_select(0, (b + 1).to(torch.int64))

            # now for float32
            a = a.to(torch.float32).requires_grad_(True)
            assert a.is_contiguous()
            assert b.is_contiguous()
            c = k2.index_select(a, b)

            assert c.dtype == a.dtype
            c.sum().backward()

            new_a = a.detach().requires_grad_(True)
            padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(new_a), new_a])
            expected = padded_a.index_select(0, (b + 1).to(torch.int64))
            expected.sum().backward()

            assert torch.allclose(a.grad, new_a.grad)


class TestSimpleRaggedIndexSelect(unittest.TestCase):

    def test_1d(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            row_splits1 = torch.tensor([0, 3, 5, 6, 6, 9],
                                       dtype=torch.int32,
                                       device=device)
            # we don't need to call shape2.to(device) here as shape2
            # will be on the same device as row_splits
            shape2 = k2.ragged.create_ragged_shape2(row_splits1, None, 9)
            values = torch.tensor([1, 0, 4, 2, 3, 0, 4, 5, 2],
                                  dtype=torch.int32,
                                  device=device)
            ragged2 = k2.RaggedInt(shape2, values)

            # contiguous
            src = torch.tensor([0, 2, 0, 10, 0, -1],
                               dtype=torch.int32,
                               device=device)
            ans = k2.simple_ragged_index_select(src, ragged2)
            self.assertEqual(ans.dtype, src.dtype)
            self.assertEqual(ans.numel(), shape2.dim0())
            expected = torch.tensor([2, 10, 0, 0, -1],
                                    dtype=torch.int32,
                                    device=device)
            self.assertTrue(torch.allclose(ans, expected))

            # non-contiguous
            src = src.expand(3, -1).t().flatten()[::3]
            self.assertFalse(src.is_contiguous())
            self.assertEqual(src.stride(0), 3)
            ans = k2.simple_ragged_index_select(src, ragged2)
            self.assertEqual(ans.dtype, src.dtype)
            self.assertEqual(ans.numel(), shape2.dim0())
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.stride(0), 1)
            expected = torch.tensor([2, 10, 0, 0, -1],
                                    dtype=torch.int32,
                                    device=device)
            self.assertTrue(torch.allclose(ans, expected))


if __name__ == '__main__':
    unittest.main()
