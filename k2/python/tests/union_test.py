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
#  ctest --verbose -R union_test_py

import unittest

import k2
import torch


class TestUnion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        s0 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            1 3 -1 0.4
            2 3 -1 0.5
            2 1 5 0.55
            3
        '''
        s1 = '''
            0 1 -1 0.6
            1
        '''
        s2 = '''
            0 1 6 0.7
            1 0 7 0.8
            1 0 8 0.9
            1 2 -1 1.0
            2
        '''
        for device in self.devices:
            fsa0 = k2.Fsa.from_str(s0).to(device)
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)

            fsa0.tensor_attr = torch.tensor([1, 2, 3, 4, 5, 6],
                                            dtype=torch.int32,
                                            device=device)
            fsa0.ragged_tensor_attr = k2.RaggedTensor(
                fsa0.tensor_attr.unsqueeze(-1))

            fsa1.tensor_attr = torch.tensor([7],
                                            dtype=torch.int32,
                                            device=device)

            fsa1.ragged_tensor_attr = k2.RaggedTensor(
                fsa1.tensor_attr.unsqueeze(-1))

            fsa2.tensor_attr = torch.tensor([8, 9, 10, 11],
                                            dtype=torch.int32,
                                            device=device)

            fsa2.ragged_tensor_attr = k2.RaggedTensor(
                fsa2.tensor_attr.unsqueeze(-1))

            fsa_vec = k2.create_fsa_vec([fsa0, fsa1, fsa2]).to(device)

            fsa = k2.union(fsa_vec)

            expected_tensor_attr = torch.tensor(
                [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11]).to(fsa.tensor_attr)
            assert torch.all(torch.eq(fsa.tensor_attr, expected_tensor_attr))

            expected_ragged_tensor_attr = k2.RaggedTensor(
                expected_tensor_attr.unsqueeze(-1)).remove_values_eq(0)
            assert str(expected_ragged_tensor_attr) == str(
                fsa.ragged_tensor_attr)

            assert torch.allclose(
                fsa.arcs.values()[:, :3],
                torch.tensor([
                    [0, 1, 0],  # fsa 0
                    [0, 4, 0],  # fsa 1
                    [0, 5, 0],  # fsa 2
                    # now for fsa0
                    [1, 2, 1],
                    [1, 3, 2],
                    [2, 3, 3],
                    [2, 7, -1],
                    [3, 7, -1],
                    [3, 2, 5],
                    # fsa1
                    [4, 7, -1],
                    # fsa2
                    [5, 6, 6],
                    [6, 5, 7],
                    [6, 5, 8],
                    [6, 7, -1]
                ]).to(torch.int32).to(device))
            assert torch.allclose(
                fsa.scores,
                torch.tensor([
                    0., 0., 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8,
                    0.9, 1.0
                ]).to(device))

    def test_autograd(self):
        s0 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 3 -1 0.3
            1 2 2 0.4
            2 3 -1 0.5
            3
        '''

        s1 = '''
            0 2 -1 0.6
            0 1 1 0.7
            1 2 -1 0.8
            2
        '''

        s2 = '''
            0 1 1 1.1
            1 2 -1 1.2
            2
        '''
        for device in self.devices:
            fsa0 = k2.Fsa.from_str(s0).to(device).requires_grad_(True)
            fsa1 = k2.Fsa.from_str(s1).to(device).requires_grad_(True)
            fsa2 = k2.Fsa.from_str(s2).to(device).requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa0, fsa1, fsa2])
            fsa = k2.union(fsa_vec)
            fsa_vec = k2.create_fsa_vec([fsa])
            log_like = fsa_vec.get_tot_scores(log_semiring=True,
                                              use_double_scores=False)
            # expected log_like and gradients are computed using gtn.
            # See https://bit.ly/35uVaUv
            log_like.backward()

            expected_log_like = torch.tensor([3.1136]).to(log_like)
            assert torch.allclose(log_like, expected_log_like)

            expected_grad_fsa0 = torch.tensor([
                0.18710044026374817, 0.08949274569749832, 0.06629786640405655,
                0.12080258131027222, 0.21029533445835114
            ]).to(device)

            expected_grad_fsa1 = torch.tensor([
                0.08097638934850693, 0.19916976988315582, 0.19916976988315582
            ]).to(device)

            expected_grad_fsa2 = torch.tensor(
                [0.4432605803012848, 0.4432605803012848]).to(device)

            assert torch.allclose(fsa0.grad, expected_grad_fsa0)
            assert torch.allclose(fsa1.grad, expected_grad_fsa1)
            assert torch.allclose(fsa2.grad, expected_grad_fsa2)


if __name__ == '__main__':
    unittest.main()
