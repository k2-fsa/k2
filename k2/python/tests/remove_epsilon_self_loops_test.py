#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R remove_epsilon_self_loops_test_py

import unittest

import k2
import torch


class TestRemoveEpsilonSelfLoops(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_single_fsa(self):
        for device in self.devices:
            # See https://git.io/JY7r4
            s = '''
                0 1 0 0.1
                0 2 0 0.2
                0 0 0 0.3
                1 1 0 0.4
                1 2 0 0.5
                2 3 -1 0.6
                3
            '''
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            scores_copy = src.scores.detach().clone().requires_grad_(True)

            src.attr1 = "hello"
            src.attr2 = "k2"
            float_attr = torch.tensor([0.1, 0.2, 0.3, 4, 5, 6],
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)

            src.float_attr = float_attr.detach().clone().requires_grad_(True)
            src.int_attr = torch.tensor([1, 2, 3, 4, 5, 6],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40,
                                                          50], [60, 70], [80],
                                               [], [0]]).to(device)

            dest = k2.remove_epsilon_self_loops(src)
            # arc map is [0, 1, 4, 5]

            # See https://git.io/JY7oC
            expected_fsa = k2.Fsa.from_str('''
                0 1 0 0.1
                0 2 0 0.2
                1 2 0 0.5
                2 3 -1 0.6
                3
            ''')
            assert k2.to_str_simple(dest) == k2.to_str_simple(
                expected_fsa), f'{str(dest)}\n{str(expected_fsa)}'

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            expected_int_attr = torch.tensor([1, 2, 5, 6],
                                             dtype=torch.int32,
                                             device=device)
            assert torch.all(torch.eq(dest.int_attr, expected_int_attr))

            expected_ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50], [],
                                                    [0]]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = float_attr[0]
            expected_float_attr[1] = float_attr[1]
            expected_float_attr[2] = float_attr[4]
            expected_float_attr[3] = float_attr[5]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = scores_copy[0]
            expected_scores[1] = scores_copy[1]
            expected_scores[2] = scores_copy[4]
            expected_scores[3] = scores_copy[5]

            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30, 40]).to(float_attr)

            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()
            assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()
            assert torch.all(torch.eq(src.scores.grad, scores_copy.grad))

    def test_fsa_vec(self):
        for device in self.devices:
            # See https://git.io/JY7r4
            s = '''
                0 1 0 0.1
                0 2 0 0.2
                0 0 0 0.3
                1 1 0 0.4
                1 2 0 0.5
                2 3 -1 0.6
                3
            '''
            fsa1 = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            scores_copy1 = fsa1.scores.detach().clone().requires_grad_(True)
            fsa1.attr1 = "hello"
            float_attr1 = torch.tensor([0.1, 0.2, 0.3, 4, 5, 6],
                                       dtype=torch.float32,
                                       requires_grad=True,
                                       device=device)
            fsa1.float_attr = float_attr1
            fsa1.int_attr = torch.tensor([1, 2, 3, 4, 5, 6],
                                         dtype=torch.int32,
                                         device=device)
            fsa1.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40,
                                                           50], [60, 70], [80],
                                                [], [0]]).to(device)

            fsa2 = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            scores_copy2 = fsa2.scores.detach().clone().requires_grad_(True)
            fsa2.attr2 = "k2"
            float_attr2 = torch.tensor([1, 2, 3, 40, 50, 60],
                                       dtype=torch.float32,
                                       requires_grad=True,
                                       device=device)
            fsa2.float_attr = float_attr2
            fsa2.int_attr = torch.tensor([10, 20, 30, 4, 5, 6],
                                         dtype=torch.int32,
                                         device=device)
            fsa2.ragged_attr = k2.RaggedTensor([[100, 200], [300, 400, 500],
                                                [600, 700], [800], [22],
                                                [33, 55]]).to(device)

            src = k2.create_fsa_vec([fsa1, fsa2])

            dest = k2.remove_epsilon_self_loops(src)
            # arc map is[0, 1, 4, 5, 6, 7, 10, 11]

            # See https://git.io/JY7oC
            expected_fsa = k2.Fsa.from_str('''
                0 1 0 0.1
                0 2 0 0.2
                1 2 0 0.5
                2 3 -1 0.6
                3
            ''')
            assert k2.to_str_simple(dest[0]) == k2.to_str_simple(expected_fsa)
            assert k2.to_str_simple(dest[1]) == k2.to_str_simple(expected_fsa)

            assert dest.attr1 == fsa1.attr1
            assert dest.attr2 == fsa2.attr2

            expected_int_attr = torch.tensor([1, 2, 5, 6, 10, 20, 5, 6],
                                             dtype=torch.int32,
                                             device=device)
            assert torch.all(torch.eq(dest.int_attr, expected_int_attr))

            expected_ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50], [],
                                                    [0], [100, 200],
                                                    [300, 400, 500], [22],
                                                    [33, 55]]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = float_attr1[0]
            expected_float_attr[1] = float_attr1[1]
            expected_float_attr[2] = float_attr1[4]
            expected_float_attr[3] = float_attr1[5]
            expected_float_attr[4] = float_attr2[0]
            expected_float_attr[5] = float_attr2[1]
            expected_float_attr[6] = float_attr2[4]
            expected_float_attr[7] = float_attr2[5]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = scores_copy1[0]
            expected_scores[1] = scores_copy1[1]
            expected_scores[2] = scores_copy1[4]
            expected_scores[3] = scores_copy1[5]
            expected_scores[4] = scores_copy2[0]
            expected_scores[5] = scores_copy2[1]
            expected_scores[6] = scores_copy2[4]
            expected_scores[7] = scores_copy2[5]

            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30, 40, 50, 60, 70,
                                  80]).to(dest.float_attr)

            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()

            assert torch.all(torch.eq(fsa1.float_attr.grad, float_attr1.grad))
            assert torch.all(torch.eq(fsa2.float_attr.grad, float_attr2.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()

            assert torch.all(torch.eq(fsa1.scores.grad, scores_copy1.grad))
            assert torch.all(torch.eq(fsa2.scores.grad, scores_copy2.grad))


if __name__ == '__main__':
    unittest.main()
