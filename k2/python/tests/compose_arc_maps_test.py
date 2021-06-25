#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
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
#  ctest --verbose -R compose_arc_maps_test_py

import unittest

import k2
import torch


class TestComposeArcMaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_simple_case(self):
        for device in self.devices:
            #                             0   1  2  3  4   5  6
            step1_arc_map = torch.tensor([-1, 0, 0, 2, 3, -1, 3],
                                         dtype=torch.int32,
                                         device=device)

            #                             0  1  2  3   4  5   6  7  8  9
            step2_arc_map = torch.tensor([0, 6, 3, -1, 4, 3, -1, 2, 1, 0],
                                         dtype=torch.int32,
                                         device=device)

            ans_arc_map = k2.compose_arc_maps(step1_arc_map, step2_arc_map)
            expected_arc_map = torch.tensor([-1, 3, 2, -1, 3, 2, -1, 0, 0, -1],
                                            dtype=torch.int32,
                                            device=device)
            assert torch.all(torch.eq(ans_arc_map, expected_arc_map))

    def test_random_case(self):
        for device in self.devices:
            step1_dim = torch.randint(low=1, high=100, size=(1,)).item()
            step1_min_val = -1
            step1_max_val = torch.randint(low=0, high=100, size=(1,)).item()

            step2_dim = torch.randint(low=1, high=100, size=(1,)).item()
            step2_min_val = -1
            step2_max_val = step1_dim - 1

            step1_arc_map = torch.randint(low=step1_min_val,
                                          high=step1_max_val + 1,
                                          size=(step1_dim,),
                                          dtype=torch.int32,
                                          device=device)

            step2_arc_map = torch.randint(low=step2_min_val,
                                          high=step2_max_val + 1,
                                          size=(step2_dim,),
                                          dtype=torch.int32,
                                          device=device)

            ans_arc_map = k2.compose_arc_maps(step1_arc_map, step2_arc_map)
            assert ans_arc_map.device == device

            step1_arc_map = step1_arc_map.tolist()
            step2_arc_map = step2_arc_map.tolist()
            ans_arc_map = ans_arc_map.tolist()

            assert len(step2_arc_map) == len(ans_arc_map)
            for i in range(step2_dim):
                if step2_arc_map[i] == -1:
                    assert ans_arc_map[i] == -1
                else:
                    assert ans_arc_map[i] == step1_arc_map[step2_arc_map[i]]


if __name__ == '__main__':
    unittest.main()
