#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R closure_test_py

import unittest

import k2
import torch


class TestClosure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_simple_fsa(self):
        s = '''
            0 1 1 0.1
            1 2 2 0.2
            2 3 -1 0.3
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa.aux_labels = torch.tensor([10, 20, -1],
                                          dtype=torch.int32).to(device)
            ans = k2.closure(fsa)

            assert torch.allclose(
                ans.arcs.values()[:, :3],
                torch.tensor([
                    [0, 1, 1],
                    [0, 3, -1],  # this arc is added by closure
                    [1, 2, 2],
                    [2, 0, 0],  # this arc's dest state and label are changed
                ]).to(torch.int32).to(device))
            scale = torch.arange(ans.arcs.values().shape[0]).to(device)
            (ans.scores * scale).sum().backward()
            assert torch.allclose(fsa.grad,
                                  torch.tensor([0., 2., 3.]).to(fsa.grad))

            assert torch.allclose(
                ans.scores,
                torch.tensor([0.1, 0.0, 0.2, 0.3]).to(device))

            assert torch.allclose(
                ans.aux_labels,
                torch.tensor([10, -1, 20, 0]).to(ans.aux_labels))

    def test_complex_fsa(self):
        s = '''
            0 1 1 0.0
            0 2 2 0.1
            0 3 -1 0.2
            1 0 0 0.3
            1 2 2 0.4
            1 3 -1 0.5
            2 0 0 0.6
            2 1 1 0.7
            2 2 2 0.8
            2 3 -1 0.9
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa.aux_labels = torch.tensor([0, 1, -1, 2, 3, -1, 4, 5, 6, -1],
                                          dtype=torch.int32).to(device)
            ans = k2.closure(fsa)
            assert torch.allclose(
                ans.arcs.values()[:, :3],
                torch.tensor([
                    [0, 1, 1],
                    [0, 2, 2],
                    [0, 0, 0],
                    [0, 3, -1],
                    [1, 0, 0],
                    [1, 2, 2],
                    [1, 0, 0],
                    [2, 0, 0],
                    [2, 1, 1],
                    [2, 2, 2],
                    [2, 0, 0],
                ]).to(torch.int32).to(device))
            assert torch.allclose(
                ans.scores,
                torch.tensor(
                    [0., 0.1, 0.2, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9]).to(device))
            scale = torch.arange(ans.arcs.values().shape[0]).to(device)
            (ans.scores * scale).sum().backward()
            assert torch.allclose(
                fsa.grad,
                torch.tensor([0., 1., 2., 4, 5, 6, 7, 8, 9, 10.]).to(fsa.grad))

            assert torch.allclose(
                ans.aux_labels,
                torch.tensor([0, 1, 0, -1, 2, 3, 0, 4, 5, 6,
                              0]).to(ans.aux_labels))


if __name__ == '__main__':
    unittest.main()
