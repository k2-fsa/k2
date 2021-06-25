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
#  ctest --verbose -R add_epsilon_self_loops_test_py

import unittest

import k2
import torch


class TestAddEpsilonSelfLoops(unittest.TestCase):

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
            0 1 1 0.1
            0 2 1 0.2
            1 3 2 0.3
            2 3 3 0.4
            3 4 -1 0.5
            4
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            new_fsa = k2.add_epsilon_self_loops(fsa)
            assert torch.all(
                torch.eq(
                    new_fsa.arcs.values()[:, :3],
                    torch.tensor(
                        [[0, 0, 0], [0, 1, 1], [0, 2, 1], [1, 1, 0], [1, 3, 2],
                         [2, 2, 0], [2, 3, 3], [3, 3, 0], [3, 4, -1]],
                        dtype=torch.int32,
                        device=device)))

            assert torch.allclose(
                new_fsa.scores,
                torch.tensor([0, 0.1, 0.2, 0, 0.3, 0, 0.4, 0, 0.5],
                             device=device))
            scale = torch.arange(new_fsa.scores.numel(), device=device)
            (new_fsa.scores * scale).sum().backward()
            assert torch.allclose(
                fsa.scores.grad,
                torch.tensor([1., 2., 4., 6., 8.], device=device))

    def test_two_fsas(self):
        s1 = '''
            0 1 1 0.1
            0 2 1 0.2
            1 3 2 0.3
            2 3 3 0.4
            3 4 -1 0.5
            4
        '''
        s2 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            2 3 -1 0.4
            3
        '''

        for device in self.devices:
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
            new_fsa_vec = k2.add_epsilon_self_loops(fsa_vec)
            assert torch.all(
                torch.eq(
                    new_fsa_vec.arcs.values()[:, :3],
                    torch.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 1], [1, 1, 0],
                                  [1, 3, 2], [2, 2, 0], [2, 3, 3], [3, 3, 0],
                                  [3, 4, -1], [0, 0, 0], [0, 1, 1], [0, 2, 2],
                                  [1, 1, 0], [1, 2, 3], [2, 2, 0], [2, 3, -1]],
                                 dtype=torch.int32,
                                 device=device)))

            assert torch.allclose(
                new_fsa_vec.scores,
                torch.tensor([
                    0, 0.1, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0, 0.1, 0.2, 0, 0.3,
                    0, 0.4
                ]).to(device))

            scale = torch.arange(new_fsa_vec.scores.numel(), device=device)
            (new_fsa_vec.scores * scale).sum().backward()

            assert torch.allclose(
                fsa1.scores.grad,
                torch.tensor([1., 2., 4., 6., 8.], device=device))

            assert torch.allclose(
                fsa2.scores.grad,
                torch.tensor([10., 11., 13., 15.], device=device))


if __name__ == '__main__':
    unittest.main()
