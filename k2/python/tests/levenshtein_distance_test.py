#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation      (authors: Wei Kang)
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

from typing import List

import random
import unittest

import k2
import torch


def levenshtein_distance(arr1: List[int], arr2: List[int]) -> int:
    m = len(arr1) + 1
    n = len(arr2) + 1
    dp = [[0] * n for _ in range(m)]
    for r in range(1, m):
        dp[r][0] = r
    for c in range(1, n):
        dp[0][c] = c
    for i in range(1, m):
        for j in range(1, n):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]


class TestLevenshteinDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test(self):
        for device in self.devices:
            refs_vec = [[1, 2, 3, 4, 5]]
            hyps_vec = [[1, 2, 3, 3, 5], [1, 2, 4, 5], [1, 2, 3, 4, 5, 6]]
            refs = k2.levenshtein_graph(refs_vec, device=device)
            hyps = k2.levenshtein_graph(hyps_vec, device=device)

            alignment, distance = k2.levenshtein_distance(
                refs,
                hyps,
                b_to_a_map=torch.tensor(
                    [0, 0, 0], dtype=torch.int32, device=device
                ),
                sorted_match_a=True,
            )

            distance_refs = torch.tensor([1.0, 1.0, 1.0])
            assert torch.allclose(distance.to("cpu"), distance_refs)

            align_refs = torch.tensor(
                [1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5, 0, -1],
                dtype=torch.int32,
            )
            align_hyps = torch.tensor(
                [1, 2, 3, 3, 5, -1, 1, 2, 0, 4, 5, -1, 1, 2, 3, 4, 5, 6, -1],
                dtype=torch.int32,
            )
            assert torch.all(torch.eq(alignment.labels.to("cpu"), align_refs))
            assert torch.all(
                torch.eq(alignment.aux_labels.to("cpu"), align_hyps)
            )

    def test_random(self):
        for device in self.devices:
            refs_num = random.randint(2, 10)
            hyps_num = random.randint(2, 10)
            refs_vec = [[random.randint(1, 100) for i in range(refs_num)]]
            hyps_vec = [
                [random.randint(1, 100) for i in range(hyps_num)]
                for j in range(hyps_num)
            ]
            refs = k2.levenshtein_graph(refs_vec, device=device)
            hyps = k2.levenshtein_graph(hyps_vec, device=device)

            alignment, distance = k2.levenshtein_distance(
                refs,
                hyps,
                b_to_a_map=torch.tensor(
                    [0] * hyps_num, dtype=torch.int32, device=device
                ),
                sorted_match_a=True,
            )
            distance_vec = []
            for i in range(hyps_num):
                distance_vec.append(
                    levenshtein_distance(refs_vec[0], hyps_vec[i])
                )

            distance_refs = torch.tensor(distance_vec, dtype=torch.float32)
            assert torch.allclose(distance.to("cpu"), distance_refs)


if __name__ == "__main__":
    unittest.main()
