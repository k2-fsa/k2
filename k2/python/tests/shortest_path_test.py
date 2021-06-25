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
#  ctest --verbose -R shortest_path_test_py

import unittest

import k2
import torch


class TestShortestPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_single_fsa(self):
        s = '''
            0 4 1 1
            0 1 1 1
            1 2 1 2
            1 3 1 3
            2 7 1 4
            3 7 1 5
            4 6 1 2
            4 8 1 3
            5 9 -1 4
            6 9 -1 3
            7 9 -1 5
            8 9 -1 6
            9
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa = k2.create_fsa_vec([fsa])
            fsa.requires_grad_(True)
            best_path = k2.shortest_path(fsa, use_double_scores=False)

            # we recompute the total_scores for backprop
            total_scores = best_path.scores.sum()

            assert total_scores == 14
            expected = torch.zeros(12)
            expected[torch.tensor([1, 3, 5, 10])] = 1
            total_scores.backward()
            assert torch.allclose(fsa.scores.grad, expected.to(device))

    def test_fsa_vec(self):
        # best path:
        #  states: 0 -> 1 -> 3 -> 7 -> 9
        #  arcs:     1 -> 3 -> 5 -> 10
        s1 = '''
            0 4 1 1
            0 1 1 1
            1 2 1 2
            1 3 1 3
            2 7 1 4
            3 7 1 5
            4 6 1 2
            4 8 1 3
            5 9 -1 4
            6 9 -1 3
            7 9 -1 5
            8 9 -1 6
            9
        '''

        #  best path:
        #   states: 0 -> 2 -> 3 -> 4 -> 5
        #   arcs:     1 -> 4 -> 5 -> 7
        s2 = '''
            0 1 1 1
            0 2 2 6
            1 2 3 3
            1 3 4 2
            2 3 5 4
            3 4 6 3
            3 5 -1 2
            4 5 -1 0
            5
        '''

        #  best path:
        #   states: 0 -> 2 -> 3
        #   arcs:     1 -> 3
        s3 = '''
            0 1 1 10
            0 2 2 100
            1 3 -1 3.5
            2 3 -1 5.5
            3
        '''

        for device in self.devices:
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)
            fsa3 = k2.Fsa.from_str(s3).to(device)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)
            fsa3.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2, fsa3])
            assert fsa_vec.shape == (3, None, None)

            best_path = k2.shortest_path(fsa_vec, use_double_scores=False)

            # we recompute the total_scores for backprop
            total_scores = best_path.scores.sum()
            total_scores.backward()

            fsa1_best_arc_indexes = torch.tensor([1, 3, 5, 10], device=device)
            assert torch.all(
                torch.eq(fsa1.scores.grad[fsa1_best_arc_indexes],
                         torch.ones(4, device=device)))
            assert fsa1.scores.grad.sum() == 4

            fsa2_best_arc_indexes = torch.tensor([1, 4, 5, 7], device=device)
            assert torch.all(
                torch.eq(fsa2.scores.grad[fsa2_best_arc_indexes],
                         torch.ones(4, device=device)))
            assert fsa2.scores.grad.sum() == 4

            fsa3_best_arc_indexes = torch.tensor([1, 3], device=device)
            assert torch.all(
                torch.eq(fsa3.scores.grad[fsa3_best_arc_indexes],
                         torch.ones(2, device=device)))
            assert fsa3.scores.grad.sum() == 2


if __name__ == '__main__':
    unittest.main()
