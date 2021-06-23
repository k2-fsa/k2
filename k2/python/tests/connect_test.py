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
#  ctest --verbose -R connect_test_py

import unittest

import k2
import torch


class TestConnect(unittest.TestCase):

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
            0 1 1 0.1
            0 2 2 0.2
            1 4 -1 0.3
            3 4 -1 0.4
            4
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            expected_str = '\n'.join(['0 1 1 0.1', '1 2 -1 0.3', '2'])
            connected_fsa = k2.connect(fsa)

            loss = connected_fsa.scores.sum()
            loss.backward()
            assert torch.allclose(fsa.scores.grad,
                                  torch.tensor([1, 0, 1, 0],
                                               dtype=torch.float32,
                                               device=device))

            actual_str = k2.to_str_simple(connected_fsa.to('cpu'))
            assert actual_str.strip() == expected_str


if __name__ == '__main__':
    unittest.main()
