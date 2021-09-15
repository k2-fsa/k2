#!/usr/bin/env python3
#
# Copyright      2020   Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R  ragged_test_py

import os
import unittest

import k2
import torch


class TestRagged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            torch.cuda.set_device(0)
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))
        cls.dtypes = [torch.int32, torch.float32, torch.float64]

    def test_ragged_int_from_str(self):
        s = '''
        [ [1 2] [3] ]
        '''
        for device in self.devices:
            ragged_int = k2.RaggedTensor(s).to(device)
            print(ragged_int)
            assert torch.all(
                torch.eq(ragged_int.values, torch.tensor([1, 2, 3],
                                                         device=device)))
            assert ragged_int.dim0 == 2
            assert torch.all(
                torch.eq(ragged_int.shape.row_splits(1),
                         torch.tensor([0, 2, 3], device=device)))

            assert ragged_int.shape.tot_sizes() == (2, 3)

    def test_pickle_ragged(self):
        for device in self.devices:
            # test num_axes == 2
            raggeds = ("[ ]", "[ [ ] ]", "[ [1 2] [3] ]")
            for s in raggeds:
                for dtype in self.dtypes:
                    ragged = k2.RaggedTensor(s, dtype).to(device)
                    torch.save(ragged, "ragged.pt")
                    ragged_reload = torch.load("ragged.pt")
                    assert ragged == ragged_reload
                    os.remove("ragged.pt")

            # test num_axes == 3
            raggeds = ("[ [ [ ] ] ]", "[ [ [1 2] [3] ] [ [4 5] [6] ] ]")
            for s in raggeds:
                for dtype in self.dtypes:
                    ragged = k2.RaggedTensor(s, dtype).to(device)
                    torch.save(ragged, "ragged.pt")
                    ragged_reload = torch.load("ragged.pt")
                    assert ragged == ragged_reload
                    os.remove("ragged.pt")


if __name__ == '__main__':
    unittest.main()
