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
#  ctest --verbose -R levenshtein_graph_test_py

import unittest

import k2
import torch


class TestLevenshteinGraph(unittest.TestCase):

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
            for weight in [-1, -1.01, -1.02]:
                s = '''
                [ [1 2 3] [ ] [4 5 6] ]
                '''
                ragged_int = k2.RaggedTensor(s).to(device)
                fsa_vec_ragged, bias_ragged = k2.levenshtein_graph(
                    ragged_int, self_loop_weight=weight, need_weight_bias=True)

                fsa_vec, bias = k2.levenshtein_graph(
                    [[1, 2, 3], [], [4, 5, 6]], device=device,
                    self_loop_weight=weight, need_weight_bias=True)

                expected_str0 = '\n'.join([
                    f'0 0 0 0 {weight}', '0 1 0 1 -1', '0 1 1 1 0',
                    f'1 1 0 0 {weight}', '1 2 0 2 -1', '1 2 2 2 0',
                    f'2 2 0 0 {weight}', '2 3 0 3 -1', '2 3 3 3 0',
                    f'3 3 0 0 {weight}', '3 4 -1 -1 0', '4'
                ])
                expected_str1 = '\n'.join([
                    f'0 0 0 0 {weight}', '0 1 -1 -1 0', '1'
                ])
                expected_str2 = '\n'.join([
                    f'0 0 0 0 {weight}', '0 1 0 4 -1', '0 1 4 4 0',
                    f'1 1 0 0 {weight}', '1 2 0 5 -1', '1 2 5 5 0',
                    f'2 2 0 0 {weight}', '2 3 0 6 -1', '2 3 6 6 0',
                    f'3 3 0 0 {weight}', '3 4 -1 -1 0', '4'
                ])
                actual_str_ragged0 = k2.to_str_simple(fsa_vec_ragged[0].to('cpu'))
                actual_str_ragged1 = k2.to_str_simple(fsa_vec_ragged[1].to('cpu'))
                actual_str_ragged2 = k2.to_str_simple(fsa_vec_ragged[2].to('cpu'))
                actual_str0 = k2.to_str_simple(fsa_vec[0].to('cpu'))
                actual_str1 = k2.to_str_simple(fsa_vec[1].to('cpu'))
                actual_str2 = k2.to_str_simple(fsa_vec[2].to('cpu'))
                assert actual_str0.strip() == expected_str0
                assert actual_str1.strip() == expected_str1
                assert actual_str2.strip() == expected_str2
                assert actual_str_ragged0.strip() == expected_str0
                assert actual_str_ragged1.strip() == expected_str1
                assert actual_str_ragged2.strip() == expected_str2

                bias_value = weight + 1
                expected_bias = torch.tensor([
                    bias_value, 0, 0, bias_value, 0, 0, bias_value, 0, 0,
                    bias_value, 0, bias_value, 0,
                    bias_value, 0, 0, bias_value, 0, 0, bias_value, 0, 0,
                    bias_value, 0], dtype=torch.float32)

                bias_ragged = bias_ragged.to('cpu')
                bias = bias.to('cpu')
                assert torch.allclose(expected_bias, bias_ragged)
                assert torch.allclose(expected_bias, bias)

if __name__ == '__main__':
    unittest.main()
