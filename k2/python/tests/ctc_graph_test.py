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
#  ctest --verbose -R ctc_graph_test_py

import unittest

import k2
import torch


class TestCtcGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        for device in self.devices:
            s = '''
            [ [1 2 2] [1 2 3] ]
            '''
            ragged_int = k2.RaggedTensor(s).to(device)
            fsa_vec_ragged = k2.ctc_graph(ragged_int)

            fsa_vec = k2.ctc_graph([[1, 2, 2], [1, 2, 3]], False, device)
            expected_str0 = '\n'.join([
                '0 0 0 0 0', '0 1 1 1 0', '1 2 0 0 0', '1 1 1 0 0',
                '1 3 2 2 0', '2 2 0 0 0', '2 3 2 2 0', '3 4 0 0 0',
                '3 3 2 0 0', '4 4 0 0 0', '4 5 2 2 0', '5 6 0 0 0',
                '5 5 2 0 0', '5 7 -1 -1 0', '6 6 0 0 0', '6 7 -1 -1 0', '7'
            ])
            expected_str1 = '\n'.join([
                '0 0 0 0 0', '0 1 1 1 0', '1 2 0 0 0', '1 1 1 0 0',
                '1 3 2 2 0', '2 2 0 0 0', '2 3 2 2 0', '3 4 0 0 0',
                '3 3 2 0 0', '3 5 3 3 0', '4 4 0 0 0', '4 5 3 3 0',
                '5 6 0 0 0', '5 5 3 0 0', '5 7 -1 -1 0', '6 6 0 0 0',
                '6 7 -1 -1 0', '7'
            ])
            actual_str_ragged0 = k2.to_str_simple(fsa_vec_ragged[0].to('cpu'))
            actual_str_ragged1 = k2.to_str_simple(fsa_vec_ragged[1].to('cpu'))
            actual_str0 = k2.to_str_simple(fsa_vec[0].to('cpu'))
            actual_str1 = k2.to_str_simple(fsa_vec[1].to('cpu'))
            assert actual_str0.strip() == expected_str0
            assert actual_str1.strip() == expected_str1
            assert actual_str_ragged0.strip() == expected_str0
            assert actual_str_ragged1.strip() == expected_str1

    def test_simplified(self):
        for device in self.devices:
            s = '''
            [ [1 2 2] [1 2 3] ]
            '''
            ragged_int = k2.RaggedTensor(s).to(device)
            fsa_vec_ragged = k2.ctc_graph(ragged_int, True)

            fsa_vec = k2.ctc_graph([[1, 2, 2], [1, 2, 3]], True, device)
            expected_str0 = '\n'.join([
                '0 0 0 0 0', '0 1 1 1 0', '1 2 0 0 0', '1 1 1 0 0',
                '1 3 2 2 0', '2 2 0 0 0', '2 3 2 2 0', '3 4 0 0 0',
                '3 3 2 0 0', '3 5 2 2 0', '4 4 0 0 0', '4 5 2 2 0',
                '5 6 0 0 0', '5 5 2 0 0', '5 7 -1 -1 0', '6 6 0 0 0',
                '6 7 -1 -1 0', '7'
            ])
            expected_str1 = '\n'.join([
                '0 0 0 0 0', '0 1 1 1 0', '1 2 0 0 0', '1 1 1 0 0',
                '1 3 2 2 0', '2 2 0 0 0', '2 3 2 2 0', '3 4 0 0 0',
                '3 3 2 0 0', '3 5 3 3 0', '4 4 0 0 0', '4 5 3 3 0',
                '5 6 0 0 0', '5 5 3 0 0', '5 7 -1 -1 0', '6 6 0 0 0',
                '6 7 -1 -1 0', '7'
            ])
            actual_str_ragged0 = k2.to_str_simple(fsa_vec_ragged[0].to('cpu'))
            actual_str_ragged1 = k2.to_str_simple(fsa_vec_ragged[1].to('cpu'))
            actual_str0 = k2.to_str_simple(fsa_vec[0].to('cpu'))
            actual_str1 = k2.to_str_simple(fsa_vec[1].to('cpu'))
            assert actual_str0.strip() == expected_str0
            assert actual_str1.strip() == expected_str1
            assert actual_str_ragged0.strip() == expected_str0
            assert actual_str_ragged1.strip() == expected_str1


if __name__ == '__main__':
    unittest.main()
