#!/usr/bin/env python3
#
# Copyright      2023  Xiaomi Corporation      (authors: Liyong Guo)
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
#  ctest --verbose -R pruned_ranges_to_lattice_test_py

import unittest

import k2
import torch


class TestPrunedRangesToLattice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))
        cls.float_dtypes = [torch.float32, torch.float64, torch.float16]

    def _common_test_part(self, ranges, frames, symbols, logits):
        ofsa, arc_map = k2.pruned_ranges_to_lattice(
            ranges, frames, symbols, logits
        )

        assert torch.equal(
            arc_map,
            torch.tensor(
                [
                    8,
                    16,
                    9,
                    24,
                    18,
                    32,
                    27,
                    36,
                    52,
                    60,
                    54,
                    68,
                    63,
                    76,
                    72,
                    81,
                    96,
                    104,
                    99,
                    112,
                    108,
                    120,
                    117,
                    126,
                    140,
                    148,
                    156,
                    164,
                    -1,
                    182,
                    180,
                    192,
                    189,
                    202,
                    198,
                    212,
                    207,
                    216,
                    227,
                    237,
                    247,
                    257,
                    252,
                    261,
                    275,
                    285,
                    295,
                    305,
                    -1,
                ],
                dtype=torch.int32,
            ),
        )
        lattice = k2.Fsa(ofsa)

        scores_tracked_by_autograd = k2.index_select(
            logits.reshape(-1).to(torch.float32), arc_map
        )

        assert torch.allclose(
            lattice.scores.to(torch.float32), scores_tracked_by_autograd
        )

        assert torch.equal(
            lattice.arcs.values()[:, :3],
            torch.tensor(
                [
                    [0, 1, 8],
                    [1, 2, 7],
                    [1, 5, 0],
                    [2, 3, 6],
                    [2, 6, 0],
                    [3, 4, 5],
                    [3, 7, 0],
                    [4, 8, 0],
                    [5, 6, 7],
                    [6, 7, 6],
                    [6, 10, 0],
                    [7, 8, 5],
                    [7, 11, 0],
                    [8, 9, 4],
                    [8, 12, 0],
                    [9, 13, 0],
                    [10, 11, 6],
                    [11, 12, 5],
                    [11, 15, 0],
                    [12, 13, 4],
                    [12, 16, 0],
                    [13, 14, 3],
                    [13, 17, 0],
                    [14, 18, 0],
                    [15, 16, 5],
                    [16, 17, 4],
                    [17, 18, 3],
                    [18, 19, 2],
                    [19, 20, -1],
                    [0, 1, 2],
                    [0, 5, 0],
                    [1, 2, 3],
                    [1, 6, 0],
                    [2, 3, 4],
                    [2, 7, 0],
                    [3, 4, 5],
                    [3, 8, 0],
                    [4, 9, 0],
                    [5, 6, 2],
                    [6, 7, 3],
                    [7, 8, 4],
                    [8, 9, 5],
                    [8, 10, 0],
                    [9, 11, 0],
                    [10, 11, 5],
                    [11, 12, 6],
                    [12, 13, 7],
                    [13, 14, 8],
                    [14, 15, -1],
                ],
                dtype=torch.int32,
            ),
        )

    def test(self):
        ranges = torch.tensor(
            [
                [
                    [0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                ],
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [3, 4, 5, 6, 7],
                    [3, 4, 5, 6, 7],
                ],
            ],
            dtype=torch.int32,
        )
        B, T, s_range = ranges.size()
        C = 9

        frames = torch.tensor([4, 3], dtype=torch.int32)
        symbols = torch.tensor(
            [[8, 7, 6, 5, 4, 3, 2], [2, 3, 4, 5, 6, 7, 8]], dtype=torch.long
        )
        logits = torch.tensor(
            [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=torch.float32
        ).expand(B, T, s_range, C)
        logits = logits + torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        ).reshape(B, T, 1, 1)
        logits = logits + torch.tensor([0.0, 1, 2, 3, 4]).reshape(
            1, 1, s_range, 1
        )
        for dtype in self.float_dtypes:
            tmp_logits = logits.to(dtype)
            self._common_test_part(ranges, frames, symbols, tmp_logits)


if __name__ == "__main__":
    unittest.main()
