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
#  ctest --verbose -R top_sort_test_py

import unittest

import k2
import torch


class TestTopSort(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        # arc 0: 0 -> 1, weight 1
        # arc 1: 0 -> 2, weight 2
        # arc 2: 1 -> 3, weight 3
        # arc 3: 2 -> 1, weight 4
        # the shortest path is 0 -> 1 -> 3, weight is 4
        # That is, (arc 0) -> (arc 2)
        s = '''
            0 1 1 1
            0 2 2 2
            1 3 -1 3
            2 1 3 4
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            sorted_fsa = k2.top_sort(fsa)

            # the shortest path in the sorted fsa is (arc 0) -> (arc 3)
            loss = (sorted_fsa.scores[0] + sorted_fsa.scores[3]) / 2
            loss.backward()
            assert torch.allclose(
                fsa.scores.grad,
                torch.tensor([0.5, 0, 0.5, 0],
                             dtype=torch.float32,
                             device=device))


if __name__ == '__main__':
    unittest.main()
