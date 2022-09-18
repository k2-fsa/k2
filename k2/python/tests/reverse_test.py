#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                2022  ASLP@NWPU          (authors: Hang Lyu)
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
#  ctest --verbose -R reverse_test_py

import unittest

import k2
import torch


class TestReverse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
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
            reversed_fsa = k2.reverse(fsa)

            loss = reversed_fsa.scores.sum()
            loss.backward()
            assert torch.allclose(fsa.scores.grad,
                                  torch.tensor([1, 1, 1, 1],
                                               dtype=torch.float32,
                                               device=device))

            actual_str = k2.to_str_simple(reversed_fsa.to('cpu'))
            expected_str = '\n'.join(['0 1 0 3', '1 3 1 1', '1 2 3 4',
                                      '2 3 2 2', '3 4 -1 0', '4'])
            assert actual_str.strip() == expected_str


if __name__ == '__main__':
    unittest.main()
