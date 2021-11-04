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
            for score in [-0.5, -0.501, -0.502]:
                s = '''
                [ [1 2 3] [ ] [4 5 6] ]
                '''
                ragged_int = k2.RaggedTensor(s).to(device)
                fsa_vec_ragged = k2.levenshtein_graph(
                    ragged_int, ins_del_score=score)

                fsa_vec = k2.levenshtein_graph(
                    [[1, 2, 3], [], [4, 5, 6]], device=device,
                    ins_del_score=score)

                expected_str0 = '\n'.join([
                    f'0 0 0 0 {score}', '0 1 0 1 -0.5', '0 1 1 1 0',
                    f'1 1 0 0 {score}', '1 2 0 2 -0.5', '1 2 2 2 0',
                    f'2 2 0 0 {score}', '2 3 0 3 -0.5', '2 3 3 3 0',
                    f'3 3 0 0 {score}', '3 4 -1 -1 0', '4'
                ])
                expected_str1 = '\n'.join([
                    f'0 0 0 0 {score}', '0 1 -1 -1 0', '1'
                ])
                expected_str2 = '\n'.join([
                    f'0 0 0 0 {score}', '0 1 0 4 -0.5', '0 1 4 4 0',
                    f'1 1 0 0 {score}', '1 2 0 5 -0.5', '1 2 5 5 0',
                    f'2 2 0 0 {score}', '2 3 0 6 -0.5', '2 3 6 6 0',
                    f'3 3 0 0 {score}', '3 4 -1 -1 0', '4'
                ])
                actual_str_ragged0 = k2.to_str_simple(
                    fsa_vec_ragged[0].to('cpu'))
                actual_str_ragged1 = k2.to_str_simple(
                    fsa_vec_ragged[1].to('cpu'))
                actual_str_ragged2 = k2.to_str_simple(
                    fsa_vec_ragged[2].to('cpu'))
                actual_str0 = k2.to_str_simple(fsa_vec[0].to('cpu'))
                actual_str1 = k2.to_str_simple(fsa_vec[1].to('cpu'))
                actual_str2 = k2.to_str_simple(fsa_vec[2].to('cpu'))
                assert actual_str0.strip() == expected_str0
                assert actual_str1.strip() == expected_str1
                assert actual_str2.strip() == expected_str2
                assert actual_str_ragged0.strip() == expected_str0
                assert actual_str_ragged1.strip() == expected_str1
                assert actual_str_ragged2.strip() == expected_str2

                offset_value = score - (-0.5)
                expected_offset = torch.tensor([
                    offset_value, 0, 0, offset_value, 0, 0, offset_value, 0, 0,
                    offset_value, 0, offset_value, 0,
                    offset_value, 0, 0, offset_value, 0, 0, offset_value, 0, 0,
                    offset_value, 0], dtype=torch.float32)

                offset_ragged = getattr(
                    fsa_vec_ragged, "__ins_del_score_offset_internal_attr_")
                offset_ragged = offset_ragged.to('cpu')
                offset = getattr(
                    fsa_vec, "__ins_del_score_offset_internal_attr_").to('cpu')
                assert torch.allclose(expected_offset, offset_ragged)
                assert torch.allclose(expected_offset, offset)


if __name__ == '__main__':
    unittest.main()
