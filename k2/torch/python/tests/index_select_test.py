#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (authors: Daniel Povey,
#                                                   Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors
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

# To run this single test, use
#
#  ctest --verbose -R index_select_test_py

import unittest

import k2
import torch


class TestIndexSelect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_1d(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(1, 2000, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows,),
                                  dtype=dtype,
                                  device=device)
                assert a.is_contiguous()
                assert a.dtype == dtype

                num_indexes = torch.randint(1, 200000, size=(1,)).item()
                b = torch.randint(-1,
                                  num_rows,
                                  size=(num_indexes,),
                                  dtype=torch.int32,
                                  device=device)

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                assert c.numel() == b.numel()

                padded_a = torch.cat([torch.tensor([0]).to(a), a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                assert torch.allclose(c, expected)

            for dtype in [torch.float32, torch.float64]:
                num_rows = torch.randint(1, 2000, size=(1,)).item()
                a = torch.rand(num_rows,
                               dtype=dtype,
                               device=device,
                               requires_grad=True)
                assert a.is_contiguous()
                assert a.dtype == dtype

                num_indexes = torch.randint(1, 200000, size=(1,)).item()
                b = torch.randint(-1,
                                  num_rows,
                                  size=(num_indexes,),
                                  dtype=torch.int32,
                                  device=device)

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                c.sum().backward()

                new_a = a.detach().requires_grad_(True)
                padded_a = torch.cat([torch.tensor([0]).to(new_a), new_a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(a.grad, new_a.grad)

    def test_1d_empty_index(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(0, 10, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows,),
                                  dtype=dtype,
                                  device=device)
                assert a.is_contiguous()
                assert a.dtype == dtype

                b = torch.empty(0, dtype=torch.int32, device=device)

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                assert c.numel() == b.numel()

            for dtype in [torch.float32, torch.float64]:
                num_rows = torch.randint(0, 10, size=(1,)).item()
                a = torch.rand(num_rows,
                               dtype=dtype,
                               device=device,
                               requires_grad=True)
                assert a.is_contiguous()
                assert a.dtype == dtype

                b = torch.empty(0, dtype=torch.int32, device=device)

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                c.sum().backward()

                new_a = a.detach().requires_grad_(True)
                padded_a = torch.cat([torch.tensor([0]).to(new_a), new_a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(a.grad, new_a.grad)

    def test_1d_non_contiguous(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(20, 2000, size=(1,)).item()
                stride = torch.randint(2, num_rows // 10 + 1, size=(1,)).item()

                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows,),
                                  dtype=dtype,
                                  device=device)
                a = a[::stride]
                assert a.is_contiguous() is False
                assert a.dtype == dtype

                b = torch.randint(-1,
                                  a.shape[0],
                                  size=(10000,),
                                  dtype=torch.int32,
                                  device=device)

                c = k2.index_select(a, b)

                padded_a = torch.cat([torch.tensor([0]).to(a), a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                assert torch.allclose(c, expected)

            for dtype in [torch.float32, torch.float64]:
                num_rows = torch.randint(20, 2000, size=(1,)).item()
                stride = torch.randint(2, num_rows // 10 + 1, size=(1,)).item()

                a_contiguous = torch.rand(num_rows, dtype=dtype, device=device)
                a = a_contiguous[::stride]
                a.requires_grad_(True)
                assert a.is_contiguous() is False
                assert a.dtype == dtype

                b = torch.randint(-1,
                                  a.shape[0],
                                  size=(10000,),
                                  dtype=torch.int32,
                                  device=device)

                c = k2.index_select(a, b)

                new_a = a_contiguous[::stride]
                new_a.requires_grad_(True)
                padded_a = torch.cat([torch.tensor([0]).to(a), new_a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                c.sum().backward()
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(a.grad, new_a.grad)

    def test_2d(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(1, 2000, size=(1,)).item()
                num_cols = torch.randint(1, 2000, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows, num_cols),
                                  dtype=dtype,
                                  device=device)
                b = torch.randint(-1,
                                  num_rows,
                                  size=(10000,),
                                  dtype=torch.int32,
                                  device=device)
                assert a.is_contiguous()
                assert a.dtype == dtype

                c = k2.index_select(a, b)

                assert c.dtype == a.dtype
                assert c.device == a.device
                assert c.shape[1] == a.shape[1]
                assert c.shape[0] == b.shape[0]

                padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(a), a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                assert torch.allclose(c, expected)

            for dtype in [torch.float32, torch.float64]:
                src = a.to(dtype).requires_grad_(True)
                assert src.is_contiguous()
                assert b.is_contiguous()
                c = k2.index_select(src, b)

                assert c.dtype == src.dtype
                c.sum().backward()

                new_src = src.detach().requires_grad_(True)
                padded_src = torch.cat(
                    [torch.zeros(1, src.shape[1]).to(new_src), new_src])
                expected = padded_src.index_select(0, (b + 1).to(torch.int64))
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(src.grad, new_src.grad)

    def test_2d_empty_index(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(0, 10, size=(1,)).item()
                num_cols = torch.randint(0, 10, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows, num_cols),
                                  dtype=dtype,
                                  device=device)
                b = torch.empty(0, dtype=torch.int32, device=device)

                assert a.is_contiguous()
                assert a.dtype == dtype

                c = k2.index_select(a, b)

                assert c.dtype == a.dtype
                assert c.device == a.device
                assert c.shape[1] == a.shape[1]
                assert c.shape[0] == b.shape[0]

                padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(a), a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                assert torch.allclose(c, expected)

            for dtype in [torch.float32, torch.float64]:
                src = a.to(dtype).requires_grad_(True)
                assert src.is_contiguous()
                assert b.is_contiguous()
                c = k2.index_select(src, b)

                assert c.dtype == src.dtype
                c.sum().backward()

                new_src = src.detach().requires_grad_(True)
                padded_src = torch.cat(
                    [torch.zeros(1, src.shape[1]).to(new_src), new_src])
                expected = padded_src.index_select(0, (b + 1).to(torch.int64))
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(src.grad, new_src.grad)

    def test_2d_non_contiguous(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.int64]:
                num_rows = torch.randint(20, 2000, size=(1,)).item()
                num_cols = torch.randint(1, 2000, size=(1,)).item()
                stride = torch.randint(2, num_rows // 10 + 1, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows, num_cols),
                                  dtype=dtype,
                                  device=device).contiguous()
                a = a[::stride]
                num_rows = a.shape[0]
                b = torch.randint(-1,
                                  num_rows,
                                  size=(10000,),
                                  dtype=torch.int32,
                                  device=device)
                assert a.is_contiguous() is False
                assert a.dtype == dtype

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                assert c.device == a.device
                assert c.shape[1] == a.shape[1]
                assert c.shape[0] == b.shape[0]

                padded_a = torch.cat([torch.zeros(1, a.shape[1]).to(a), a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))

                assert torch.allclose(c, expected)

            for dtype in [torch.float32, torch.float64]:
                num_rows = torch.randint(20, 2000, size=(1,)).item()
                num_cols = torch.randint(1, 2000, size=(1,)).item()
                stride = torch.randint(2, num_rows // 10 + 1, size=(1,)).item()
                a = torch.randint(-1000,
                                  1000,
                                  size=(num_rows, num_cols),
                                  dtype=dtype,
                                  device=device).contiguous()
                a = a[::stride]
                num_rows = a.shape[0]
                b = torch.randint(-1,
                                  num_rows,
                                  size=(10000,),
                                  dtype=torch.int32,
                                  device=device)
                assert a.is_contiguous() is False
                assert a.dtype == dtype
                a.requires_grad_(True)

                c = k2.index_select(a, b)
                assert c.dtype == a.dtype
                assert c.device == a.device
                assert c.shape[1] == a.shape[1]
                assert c.shape[0] == b.shape[0]

                c.sum().backward()

                new_a = a.detach().requires_grad_(True)
                padded_a = torch.cat(
                    [torch.zeros(1, a.shape[1]).to(new_a), new_a])
                expected = padded_a.index_select(0, (b + 1).to(torch.int64))
                expected.sum().backward()

                assert torch.allclose(c, expected)
                assert torch.allclose(a.grad, new_a.grad)


class TestSimpleRaggedIndexSelect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_1d(self):
        for device in self.devices:
            row_splits1 = torch.tensor([0, 3, 5, 6, 6, 9],
                                       dtype=torch.int32,
                                       device=device)
            # we don't need to call shape2.to(device) here as shape2
            # will be on the same device as row_splits
            shape2 = k2.ragged.create_ragged_shape2(row_splits1, None, 9)
            values = torch.tensor([1, 0, 4, 2, 3, 0, 4, 5, 2],
                                  dtype=torch.int32,
                                  device=device)
            ragged2 = k2.RaggedTensor(shape2, values)

            # contiguous
            src = torch.tensor([0, 2, 0, 10, 0, -1],
                               dtype=torch.int32,
                               device=device)
            ans = k2.simple_ragged_index_select(src, ragged2)
            self.assertEqual(ans.dtype, src.dtype)
            self.assertEqual(ans.numel(), shape2.dim0)
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
            self.assertEqual(ans.numel(), shape2.dim0)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.stride(0), 1)
            expected = torch.tensor([2, 10, 0, 0, -1],
                                    dtype=torch.int32,
                                    device=device)
            self.assertTrue(torch.allclose(ans, expected))


if __name__ == '__main__':
    unittest.main()
