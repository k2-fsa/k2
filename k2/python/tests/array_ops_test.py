#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation (author: Wei kang)
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
#  ctest --verbose -R array_ops_test_py

import unittest

import random
import torch
import k2


class TestArrayOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

        cls.dtypes = [torch.int32, torch.float32, torch.float64]

    def test_monotonic_lower_bound(self):
        for device in self.devices:
            for dtype in self.dtypes:
                # simple case
                src = torch.tensor([2, 1, 3, 7, 5, 8, 20, 15],
                                   dtype=dtype,
                                   device=device)
                expected = torch.tensor([1, 1, 3, 5, 5, 8, 15, 15],
                                        dtype=dtype,
                                        device=device)
                dest = k2.monotonic_lower_bound(src)
                assert torch.allclose(dest, expected)
                assert torch.allclose(
                    src,
                    torch.tensor([2, 1, 3, 7, 5, 8, 20, 15],
                                 dtype=dtype,
                                 device=device))
                k2.monotonic_lower_bound(src, inplace=True)
                assert torch.allclose(src, expected)

                # random case
                src = torch.randint(100, (10, 100), dtype=dtype, device=device)
                dest = k2.monotonic_lower_bound(src)
                expected = torch.zeros_like(src, device=torch.device("cpu"))
                dest = dest.to("cpu")
                for i in range(src.shape[0]):
                    min_value = 101
                    for j in range(src.shape[1] - 1, -1, -1):
                        min_value = min(dest[i][j], min_value)
                        expected[i][j] = min_value
                assert torch.allclose(dest, expected)

                k2.monotonic_lower_bound(src, inplace=True)
                src = src.to("cpu")
                assert torch.allclose(src, expected)


if __name__ == '__main__':
    unittest.main()
