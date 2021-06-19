#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corp.       (author: Fangjun Kuang)
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
#  ctest --verbose -R get_forward_scores_test_py

import unittest

import k2
import torch


class TestGetForwardScores(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_simple_fsa_case_1(self):
        # see https://git.io/JtttZ
        s = '''
            0 1 1 0.0
            0 1 2 0.1
            0 2 3 2.2
            1 2 4 0.5
            1 2 5 0.6
            1 3 -1 3.0
            2 3 -1 0.8
            3
        '''
        for device in self.devices:
            for use_double_scores in [True, False]:
                fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
                fsa_vec = k2.create_fsa_vec([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_forward_scores = torch.tensor([
                    0,  # start state
                    0.1,  # state 1, arc: 0 -> 1 (2/0.1)
                    2.2,  # state 2, arc: 0 -> 2 (3/2.2)
                    3.1,  # state 3, arc: 1 -> 3 (-1/3.0)
                ]).to(forward_scores)
                assert torch.allclose(forward_scores, expected_forward_scores)
                scale = torch.arange(forward_scores.numel()).to(device)
                (scale * forward_scores).sum().backward()
                expected_grad = torch.tensor([0, 4, 2, 0, 0, 3,
                                              0]).to(fsa.grad)
                assert torch.allclose(fsa.grad, expected_grad)

                # now for log semiring
                fsa.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_forward_scores = torch.empty_like(forward_scores)
                expected_forward_scores[0] = 0
                expected_forward_scores[1] = scores[:2].exp().sum().log()
                expected_forward_scores[2] = (
                    scores[2].exp() +
                    (expected_forward_scores[1] + scores[3]).exp() +
                    (expected_forward_scores[1] + scores[4]).exp()).log()
                expected_forward_scores[3] = (
                    (expected_forward_scores[1] + scores[5]).exp() +
                    (expected_forward_scores[2] + scores[6]).exp()).log()
                assert torch.allclose(forward_scores, expected_forward_scores)

                (scale * forward_scores).sum().backward()
                (scale * expected_forward_scores).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

    def test_simple_fsa_case_2(self):
        # see https://git.io/Jttmm
        s = '''
            0 1 1 0.1
            0 1 2 0.2
            1 2 3 0.3
            1 2 4 0.4
            1 2 5 0.5
            2 3 6 0.6
            2 3 7 0.7
            3 4 -1 0.8
            4
        '''
        for device in self.devices:
            for use_double_scores in [True, False]:
                fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
                fsa_vec = k2.create_fsa_vec([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_forward_scores = torch.tensor([
                    0,  # start state
                    0.2,  # state 1
                    0.5 + 0.2,  # state 2
                    0.5 + 0.2 + 0.7,  # state 3
                    0.5 + 0.2 + 0.7 + 0.8,  # state 4
                ]).to(forward_scores)
                # [0, 0.2, 0.7, 1.4, 2.2]

                assert torch.allclose(forward_scores, expected_forward_scores)
                scale = torch.arange(forward_scores.numel()).to(device)
                (scale * forward_scores).sum().backward()

                expected_grad = torch.zeros_like(fsa.scores)
                expected_grad[7] = scale[4]
                expected_grad[6] = scale[3] + scale[4]
                expected_grad[4] = scale[2] + scale[3] + scale[4]
                expected_grad[1] = scale[1] + scale[2] + scale[3] + scale[4]
                # [0, 10, 0, 0, 9, 0, 7, 4]
                assert torch.allclose(fsa.grad, expected_grad)

                # now for log semiring
                fsa.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_forward_scores = torch.empty_like(forward_scores)
                expected_forward_scores[0] = 0
                expected_forward_scores[1] = scores[:2].exp().sum().log()
                expected_forward_scores[2] = (
                    (expected_forward_scores[1] + scores[2]).exp() +
                    (expected_forward_scores[1] + scores[3]).exp() +
                    (expected_forward_scores[1] + scores[4]).exp()).log()
                expected_forward_scores[3] = (
                    (expected_forward_scores[2] + scores[5]).exp() +
                    (expected_forward_scores[2] + scores[6]).exp()).log()
                expected_forward_scores[
                    4] = expected_forward_scores[3] + scores[7]

                assert torch.allclose(forward_scores, expected_forward_scores)
                (scale * forward_scores).sum().backward()
                (scale * expected_forward_scores).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

    def test_simple_fsa_vec(self):
        # combine case 1 and case 2
        s1 = '''
            0 1 1 0.0
            0 1 2 0.1
            0 2 3 2.2
            1 2 4 0.5
            1 2 5 0.6
            1 3 -1 3.0
            2 3 -1 0.8
            3
        '''

        s2 = '''
            0 1 1 0.1
            0 1 2 0.2
            1 2 3 0.3
            1 2 4 0.4
            1 2 5 0.5
            2 3 6 0.6
            2 3 7 0.7
            3 4 -1 0.8
            4
        '''
        for device in self.devices:
            for use_double_scores in [True, False]:
                fsa1 = k2.Fsa.from_str(s1).to(device).requires_grad_(True)
                fsa2 = k2.Fsa.from_str(s2).to(device).requires_grad_(True)
                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_scores1 = torch.tensor([0, 0.1, 2.2,
                                                 3.1]).to(forward_scores)
                expected_scores2 = torch.tensor([0, 0.2, 0.7, 1.4,
                                                 2.2]).to(forward_scores)
                expected_forward_scores = torch.cat(
                    [expected_scores1, expected_scores2])
                assert torch.allclose(forward_scores, expected_forward_scores)

                scale1 = torch.arange(expected_scores1.numel()).to(device)
                scale2 = torch.arange(expected_scores2.numel()).to(device)
                scale = torch.cat([scale1, scale2])
                (scale * forward_scores).sum().backward()

                expected_grad1 = torch.tensor([0, 4, 2, 0, 0, 3,
                                               0]).to(fsa1.grad)
                expected_grad2 = torch.tensor([0, 10, 0, 0, 9, 0, 7,
                                               4]).to(fsa2.grad)

                assert torch.allclose(fsa1.grad, expected_grad1)
                assert torch.allclose(fsa2.grad, expected_grad2)

                # now for log semiring
                fsa1.scores.grad = None
                fsa2.scores.grad = None

                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)

                scores1 = fsa1.scores.detach().clone().requires_grad_(True)
                scores2 = fsa2.scores.detach().clone().requires_grad_(True)

                expected_scores1 = torch.empty_like(forward_scores[:4])
                expected_scores2 = torch.empty_like(forward_scores[4:])

                expected_scores1[0] = 0
                expected_scores1[1] = scores1[:2].exp().sum().log()
                expected_scores1[2] = (
                    scores1[2].exp() +
                    (expected_scores1[1] + scores1[3]).exp() +
                    (expected_scores1[1] + scores1[4]).exp()).log()
                expected_scores1[3] = (
                    (expected_scores1[1] + scores1[5]).exp() +
                    (expected_scores1[2] + scores1[6]).exp()).log()

                expected_scores2[0] = 0
                expected_scores2[1] = scores2[:2].exp().sum().log()
                expected_scores2[2] = (
                    (expected_scores2[1] + scores2[2]).exp() +
                    (expected_scores2[1] + scores2[3]).exp() +
                    (expected_scores2[1] + scores2[4]).exp()).log()
                expected_scores2[3] = (
                    (expected_scores2[2] + scores2[5]).exp() +
                    (expected_scores2[2] + scores2[6]).exp()).log()
                expected_scores2[4] = expected_scores2[3] + scores2[7]

                expected_forward_scores = torch.cat(
                    [expected_scores1, expected_scores2])
                assert torch.allclose(forward_scores,
                                      expected_forward_scores,
                                      atol=1e-4)

                (scale * forward_scores).sum().backward()
                (scale * expected_forward_scores).sum().backward()
                assert torch.allclose(fsa1.grad, scores1.grad)
                assert torch.allclose(fsa2.grad, scores2.grad)


if __name__ == '__main__':
    unittest.main()
