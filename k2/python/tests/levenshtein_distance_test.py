#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corporation      (authors: Wei Kang)
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
#  ctest --verbose -R levenshtein_distance_test_py

import unittest

import k2
import torch


class TestLevenshteinDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test_basic(self):
        for device in self.devices:
            px = torch.tensor([1, 3, 4, 9, 5], dtype=torch.int32, device=device)
            py = torch.tensor(
                [2, 3, 4, 5, 9, 7], dtype=torch.int32, device=device
            )
            d = k2.levenshtein_distance(px=px, py=py)
            expected = torch.tensor(
                [
                    [
                        [0, 1, 2, 3, 4, 5, 6],
                        [1, 1, 2, 3, 4, 5, 6],
                        [2, 2, 1, 2, 3, 4, 5],
                        [3, 3, 2, 1, 2, 3, 4],
                        [4, 4, 3, 2, 2, 2, 3],
                        [5, 5, 4, 3, 2, 3, 3],
                    ]
                ],
                dtype=torch.int32,
                device=device,
            )
            assert torch.equal(d, expected)

            # with boundary
            boundary = torch.tensor([[1, 2, 4, 5]], device=device)
            d = k2.levenshtein_distance(px=px, py=py, boundary=boundary)
            expected = torch.tensor(
                [[[0, 1, 2, 3], [1, 1, 2, 3], [2, 1, 2, 3], [3, 2, 2, 2]]],
                dtype=torch.int32,
                device=device,
            )
            assert torch.equal(d[:, 1:5, 2:6], expected)

    def test_empty(self):
        for device in self.devices:
            px = torch.tensor([], dtype=torch.int32, device=device)
            py = torch.tensor(
                [2, 3, 4, 5, 9, 7], dtype=torch.int32, device=device
            )
            d = k2.levenshtein_distance(px=px, py=py)
            expected = torch.tensor(
                [[[0, 1, 2, 3, 4, 5, 6]]], dtype=torch.int32, device=device
            )
            assert torch.equal(d, expected)

            px = torch.tensor([1, 3, 4, 9, 5], dtype=torch.int32, device=device)
            py = torch.tensor([], dtype=torch.int32, device=device)
            d = k2.levenshtein_distance(px=px, py=py)
            expected = torch.tensor(
                [[[0], [1], [2], [3], [4], [5]]],
                dtype=torch.int32,
                device=device,
            )
            assert torch.equal(d, expected)

    def test_random(self):
        batch = torch.randint(10, 50, (1,)).item()
        S = torch.randint(10, 1000, (1,)).item()
        U = torch.randint(10, 1000, (1,)).item()
        px_ = torch.randint(0, 100, (batch, S), dtype=torch.int32)
        py_ = torch.randint(0, 100, (batch, U), dtype=torch.int32)
        px_s = torch.randint(0, S // 2, (batch, 1))
        px_e = torch.randint(S // 2, S, (batch, 1))
        py_s = torch.randint(0, U // 2, (batch, 1))
        py_e = torch.randint(U // 2, U, (batch, 1))
        boundary_ = torch.cat([px_s, py_s, px_e, py_e], dim=1)

        for device in self.devices:
            px = px_.to(device)
            py = py_.to(device)
            boundary = boundary_.to(device)
            d = k2.levenshtein_distance(px=px, py=py, boundary=boundary)
            if device == torch.device("cpu"):
                expected_ = d
            expected = expected_.to(device)
            for i in range(batch):
                assert torch.equal(
                    d[
                        i,
                        boundary[i][0]: boundary[i][2] + 1,
                        boundary[i][1]: boundary[i][3] + 1,
                    ],
                    expected[
                        i,
                        boundary[i][0]: boundary[i][2] + 1,
                        boundary[i][1]: boundary[i][3] + 1,
                    ],
                )


if __name__ == "__main__":
    unittest.main()
