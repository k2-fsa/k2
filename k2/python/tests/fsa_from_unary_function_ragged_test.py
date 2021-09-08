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
#  ctest --verbose -R  fsa_from_unary_function_ragged_test_py

import unittest

import k2
import torch
import _k2


class TestFsaFromUnaryFunctionRagged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_without_empty_list(self):
        for device in self.devices:
            s = '''
                0 1 0 0
                0 1 1 0
                1 2 -1 0
                2
            '''
            scores = torch.tensor([1, 2, 3],
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)
            scores_copy = scores.detach().clone().requires_grad_(True)
            src = k2.Fsa.from_str(s).to(device)
            src.scores = scores
            # see https://git.io/Jufpl
            src.attr1 = "hello"
            src.attr2 = "k2"
            float_attr = torch.tensor([0.1, 0.2, 0.3],
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)

            src.float_attr = float_attr.detach().clone().requires_grad_(True)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50],
                                               [60, 70]]).to(device)

            ragged_arc, arc_map = _k2.remove_epsilon(src.arcs, src.properties)
            # see https://git.io/Jufpe
            dest = k2.utils.fsa_from_unary_function_ragged(
                src, ragged_arc, arc_map)
            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            expected_arc_map = k2.RaggedTensor([[1], [0, 2], [2]]).to(device)
            assert arc_map == expected_arc_map

            expected_int_attr = k2.RaggedTensor([[2], [1, 3], [3]]).to(device)
            assert dest.int_attr == expected_int_attr

            expected_ragged_attr = k2.RaggedTensor([[30, 40, 50],
                                                    [10, 20, 60, 70],
                                                    [60, 70]]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = float_attr[1]
            expected_float_attr[1] = float_attr[0] + float_attr[2]
            expected_float_attr[2] = float_attr[2]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = scores_copy[1]
            expected_scores[1] = scores_copy[0] + scores_copy[2]
            expected_scores[2] = scores_copy[2]

            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30]).to(float_attr)

            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()
            assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()
            assert torch.all(torch.eq(scores.grad, scores_copy.grad))

    def test_with_empty_list(self):
        for device in self.devices:
            s = '''
                0 1 0 0
                0 1 1 0
                1 2 -1 0
                2
            '''
            scores = torch.tensor([1, 2, 3],
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)
            scores_copy = scores.detach().clone().requires_grad_(True)
            src = k2.Fsa.from_str(s).to(device)
            src.scores = scores
            # see https://git.io/JuJTk
            src.attr1 = "hello"
            src.attr2 = "k2"
            float_attr = torch.tensor([0.1, 0.2, 0.3],
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)

            src.float_attr = float_attr.detach().clone().requires_grad_(True)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50],
                                               [60, 70]]).to(device)

            ragged_arc, arc_map = _k2.remove_epsilon_and_add_self_loops(
                src.arcs, src.properties)
            dest = k2.utils.fsa_from_unary_function_ragged(
                src, ragged_arc, arc_map)
            # see https://git.io/JuJTZ
            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            expected_arc_map = k2.RaggedTensor([[], [1], [0, 2], [],
                                                [2]]).to(device)
            assert arc_map == expected_arc_map

            expected_int_attr = k2.RaggedTensor([[], [2], [1, 3], [],
                                                 [3]]).to(device)
            assert dest.int_attr == expected_int_attr

            expected_ragged_attr = k2.RaggedTensor([[], [30, 40, 50],
                                                    [10, 20, 60, 70], [],
                                                    [60, 70]]).to(device)

            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = 0
            expected_float_attr[1] = float_attr[1]
            expected_float_attr[2] = float_attr[0] + float_attr[2]
            expected_float_attr[3] = 0
            expected_float_attr[4] = float_attr[2]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = 0
            expected_scores[1] = scores_copy[1]
            expected_scores[2] = scores_copy[0] + scores_copy[2]
            expected_scores[3] = 0
            expected_scores[4] = scores_copy[2]
            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30, 40, 50]).to(float_attr)
            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()
            assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()
            assert torch.all(torch.eq(scores.grad, scores_copy.grad))


if __name__ == '__main__':
    unittest.main()
