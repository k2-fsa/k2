#!/usr/bin/env python3
#
# Copyright      2020-2021  Xiaomi Corporation (authors: Haowen Qiu
#                                                        Fangjun Kuang)
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
#  ctest --verbose -R invert_test_py

import unittest

import torch
import k2


class TestInvert(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_aux_as_tensor(self):
        s = '''
            0 1 1 1 0
            0 1 0 2 0
            0 3 2 3 0
            1 2 3 4 0
            1 3 4 5 0
            2 1 5 6 0
            2 5 -1 -1 0
            3 1 6 7 0
            4 5 -1 -1 0
            5
        '''
        fsa = k2.Fsa.from_str(s, num_aux_labels=1)
        assert fsa.device.type == 'cpu'
        dest = k2.invert(fsa)
        print(dest)

    def test_aux_as_ragged(self):
        s = '''
            0 1 1 0
            0 1 0 0
            0 3 2 0
            1 2 3 0
            1 3 4 0
            2 1 5 0
            2 5 -1 0
            3 1 6 0
            4 5 -1 0
            5
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.device.type == 'cpu'
        aux_row_splits = torch.tensor([0, 2, 3, 3, 6, 6, 7, 8, 10, 11],
                                      dtype=torch.int32)
        aux_shape = k2.ragged.create_ragged_shape2(aux_row_splits, None, 11)
        aux_values = torch.tensor([1, 2, 3, 5, 6, 7, 8, -1, 9, 10, -1],
                                  dtype=torch.int32)
        fsa.aux_labels = k2.RaggedTensor(aux_shape, aux_values)
        dest = k2.invert(fsa)
        print(dest)  # will print aux_labels as well
        # TODO(haowen): wrap C++ code to check equality for Ragged?

    def test_aux_ragged(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                0 2 2 0.2
                1 3 3 0.3
                2 3 4 0.6
                3 4 -1 0.7
                4
            '''
            # https://git.io/JqNiR
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.aux_labels = k2.RaggedTensor('[[2 3] [3 4] [] [5] [-1]]').to(
                device)
            fsa.tensor_attr1 = torch.tensor([1, 2, 3, 4, 5]).to(device)
            # https://git.io/JqNiw
            ans = k2.invert(fsa)
            assert torch.all(
                torch.eq(ans.tensor_attr1,
                         torch.tensor([1, 2, 0, 0, 3, 4, 5], device=device)))
            assert torch.all(
                torch.eq(ans.aux_labels,
                         torch.tensor([1, 2, 0, 0, 3, 4, -1], device=device)))
            assert torch.all(
                torch.eq(ans.labels,
                         torch.tensor([2, 3, 3, 4, 0, 5, -1], device=device)))


if __name__ == '__main__':
    unittest.main()
