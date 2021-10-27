#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
#  ctest --verbose -R index_add_test_py

import unittest

import k2
import torch


class TestIndexAdd(unittest.TestCase):

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
            for dtype in [torch.int32, torch.float32, torch.float64]:
                num_elements = torch.randint(10, 1000, (1,)).item()
                src = torch.randint(-1000,
                                    1000,
                                    size=(num_elements,),
                                    dtype=dtype,
                                    device=device)

                num_indexes = num_elements * torch.randint(2, 10, (1,)).item()
                index = torch.randint(-1,
                                      num_elements,
                                      size=(num_indexes,),
                                      dtype=torch.int32,
                                      device=device)

                value = torch.randint(-1000,
                                      1000,
                                      size=(num_indexes,),
                                      dtype=dtype,
                                      device=device)

                assert src.is_contiguous()
                assert index.is_contiguous()
                assert value.is_contiguous()
                assert src.dtype == value.dtype == dtype
                assert index.dtype == torch.int32
                assert src.device == value.device == index.device == device

                saved = src.clone()
                k2.index_add(index, value, src)

                saved = torch.cat([torch.tensor([0]).to(saved), saved])

                saved.index_add_(0, index.to(torch.int64) + 1, value)
                assert torch.all(torch.eq(src, saved[1:]))

    def test_1d_non_contiguous(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.float32, torch.float64]:
                num_elements = torch.randint(20, 1000, (1,)).item()
                src_stride = torch.randint(2,
                                           num_elements // 10 + 1,
                                           size=(1,)).item()
                src = torch.randint(-1000,
                                    1000,
                                    size=(num_elements,),
                                    dtype=dtype,
                                    device=device)
                src = src[::src_stride]
                num_indexes = src.numel() * torch.randint(2, 10, (1,)).item()
                index = torch.randint(-1,
                                      src.numel(),
                                      size=(num_indexes,),
                                      dtype=torch.int32,
                                      device=device)

                value_stride = torch.randint(2, 6, (1,)).item()
                value = torch.randint(-1000,
                                      1000,
                                      size=(num_indexes * value_stride,),
                                      dtype=dtype,
                                      device=device)
                value = value[::value_stride]

                assert src.is_contiguous() is False
                assert index.is_contiguous()
                assert value.is_contiguous() is False
                assert src.dtype == value.dtype == dtype
                assert index.dtype == torch.int32
                assert src.device == value.device == index.device == device

                saved = src.clone()
                k2.index_add(index, value, src)

                saved = torch.cat([torch.tensor([0]).to(saved), saved])

                saved.index_add_(0, index.to(torch.int64) + 1, value)
                assert torch.all(torch.eq(src, saved[1:]))

    def test_2d(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.float32, torch.float64]:
                num_rows = torch.randint(10, 1000, (1,)).item()
                num_cols = torch.randint(10, 1000, (1,)).item()
                src = torch.randint(-1000,
                                    1000,
                                    size=(num_rows, num_cols),
                                    dtype=dtype,
                                    device=device)

                num_indexes = num_rows * torch.randint(2, 10, (1,)).item()
                index = torch.randint(-1,
                                      num_rows,
                                      size=(num_indexes,),
                                      dtype=torch.int32,
                                      device=device)

                value = torch.randint(-1000,
                                      1000,
                                      size=(num_indexes, num_cols),
                                      dtype=dtype,
                                      device=device)

                assert src.is_contiguous()
                assert index.is_contiguous()
                assert value.is_contiguous()
                assert src.dtype == value.dtype == dtype
                assert index.dtype == torch.int32
                assert src.device == value.device == index.device == device

                saved = src.clone()
                k2.index_add(index, value, src)

                saved = torch.cat(
                    [torch.zeros(1, saved.shape[1]).to(saved), saved])

                saved.index_add_(0, index.to(torch.int64) + 1, value)
                assert torch.all(torch.eq(src, saved[1:]))

    def test_2d_non_contiguous(self):
        for device in self.devices:
            for dtype in [torch.int32, torch.float32, torch.float64]:
                col_stride = torch.randint(2, 8, (1,)).item()

                num_rows = torch.randint(10, 1000, (1,)).item()
                num_cols = torch.randint(10, 1000, (1,)).item() * col_stride
                src = torch.randint(-1000,
                                    1000,
                                    size=(num_rows, num_cols),
                                    dtype=dtype,
                                    device=device)
                src = src[:, ::col_stride]

                num_indexes = num_rows * torch.randint(2, 10, (1,)).item()
                index = torch.randint(-1,
                                      num_rows,
                                      size=(num_indexes,),
                                      dtype=torch.int32,
                                      device=device)

                value_stride = torch.randint(2, 8, (1,)).item()
                value = torch.randint(-1000,
                                      1000,
                                      size=(num_indexes,
                                            num_cols * value_stride),
                                      dtype=dtype,
                                      device=device)
                value = value[:, ::(col_stride * value_stride)]

                assert src.is_contiguous() is False
                assert index.is_contiguous()
                assert value.is_contiguous() is False
                assert src.dtype == value.dtype == dtype
                assert index.dtype == torch.int32
                assert src.device == value.device == index.device == device

                saved = src.clone()
                k2.index_add(index, value, src)

                saved = torch.cat(
                    [torch.zeros(1, saved.shape[1]).to(saved), saved])

                saved.index_add_(0, index.to(torch.int64) + 1, value)
                assert torch.all(torch.eq(src, saved[1:]))


if __name__ == '__main__':
    unittest.main()
