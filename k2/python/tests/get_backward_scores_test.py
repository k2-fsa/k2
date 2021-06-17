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
#  ctest --verbose -R get_backward_scores_test_py

import unittest

import k2
import torch


class TestGetBackwardScores(unittest.TestCase):

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
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_backward_scores = torch.empty_like(backward_scores)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_backward_scores[3] = 0
                # yapf:disable
                expected_backward_scores[2] = expected_backward_scores[3] + scores[6] # noqa
                expected_backward_scores[1] = expected_backward_scores[3] + scores[5] # noqa
                expected_backward_scores[0] = expected_backward_scores[1] + scores[1] # noqa
                # yapf:enable
                assert torch.allclose(backward_scores,
                                      expected_backward_scores)
                scale = torch.arange(backward_scores.numel()).to(device)
                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

                # now for log semiring
                fsa.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                expected_backward_scores = torch.zeros_like(backward_scores)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_backward_scores[3] = 0
                # yapf:disable
                expected_backward_scores[2] = expected_backward_scores[3] + scores[6] # noqa
                # yapf:enable
                expected_backward_scores[1] = (
                    (expected_backward_scores[2] + scores[3]).exp() +
                    (expected_backward_scores[2] + scores[4]).exp() +
                    (expected_backward_scores[3] + scores[5]).exp()).log()
                expected_backward_scores[0] = (
                    (expected_backward_scores[1] + scores[0]).exp() +
                    (expected_backward_scores[1] + scores[1]).exp() +
                    (expected_backward_scores[2] + scores[2]).exp()).log()

                assert torch.allclose(backward_scores,
                                      expected_backward_scores)
                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
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
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_backward_scores = torch.empty_like(backward_scores)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_backward_scores[4] = 0
                # yapf:disable
                expected_backward_scores[3] = expected_backward_scores[4] + scores[7] # noqa
                expected_backward_scores[2] = expected_backward_scores[3] + scores[6] # noqa
                expected_backward_scores[1] = expected_backward_scores[2] + scores[4] # noqa
                expected_backward_scores[0] = expected_backward_scores[1] + scores[1] # noqa
                # yapf:enable
                assert torch.allclose(backward_scores,
                                      expected_backward_scores)
                scale = torch.arange(backward_scores.numel()).to(device)
                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

                # now for log semiring
                fsa.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                expected_backward_scores = torch.zeros_like(backward_scores)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_backward_scores[4] = 0
                # yapf:disable
                expected_backward_scores[3] = expected_backward_scores[4] + scores[7] # noqa
                # yapf:enable
                expected_backward_scores[2] = (
                    (expected_backward_scores[3] + scores[5]).exp() +
                    (expected_backward_scores[3] + scores[6]).exp()).log()
                expected_backward_scores[1] = (
                    (expected_backward_scores[2] + scores[2]).exp() +
                    (expected_backward_scores[2] + scores[3]).exp() +
                    (expected_backward_scores[2] + scores[4]).exp()).log()
                expected_backward_scores[0] = (
                    (expected_backward_scores[1] + scores[0]).exp() +
                    (expected_backward_scores[1] + scores[1]).exp()).log()
                assert torch.allclose(backward_scores,
                                      expected_backward_scores)
                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
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
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                scores1 = fsa1.scores.detach().clone().requires_grad_(True)
                scores2 = fsa2.scores.detach().clone().requires_grad_(True)
                expected_backward_scores1 = torch.empty_like(
                    backward_scores[:4])
                expected_backward_scores2 = torch.empty_like(
                    backward_scores[4:])

                expected_backward_scores1[3] = 0
                # yapf:disable
                expected_backward_scores1[2] = expected_backward_scores1[3] + scores1[6] # noqa
                expected_backward_scores1[1] = expected_backward_scores1[3] + scores1[5] # noqa
                expected_backward_scores1[0] = expected_backward_scores1[1] + scores1[1] # noqa
                # yapf:enable

                expected_backward_scores2[4] = 0
                # yapf:disable
                expected_backward_scores2[3] = expected_backward_scores2[4] + scores2[7] # noqa
                expected_backward_scores2[2] = expected_backward_scores2[3] + scores2[6] # noqa
                expected_backward_scores2[1] = expected_backward_scores2[2] + scores2[4] # noqa
                expected_backward_scores2[0] = expected_backward_scores2[1] + scores2[1] # noqa
                # yapf:enable
                expected_backward_scores = torch.cat(
                    [expected_backward_scores1, expected_backward_scores2])
                assert torch.allclose(backward_scores,
                                      expected_backward_scores)

                scale = torch.arange(backward_scores.numel()).to(device)
                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
                assert torch.allclose(fsa1.grad, scores1.grad)
                assert torch.allclose(fsa2.grad, scores2.grad)

                # now for log semiring
                fsa1.scores.grad = None
                fsa2.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                scores1 = fsa1.scores.detach().clone().requires_grad_(True)
                scores2 = fsa2.scores.detach().clone().requires_grad_(True)
                expected_backward_scores1 = torch.empty_like(
                    backward_scores[:4])
                expected_backward_scores2 = torch.empty_like(
                    backward_scores[4:])

                expected_backward_scores1[3] = 0
                # yapf:disable
                expected_backward_scores1[2] = expected_backward_scores1[3] + scores1[6] # noqa
                # yapf:enable
                expected_backward_scores1[1] = (
                    (expected_backward_scores1[2] + scores1[3]).exp() +
                    (expected_backward_scores1[2] + scores1[4]).exp() +
                    (expected_backward_scores1[3] + scores1[5]).exp()).log()
                expected_backward_scores1[0] = (
                    (expected_backward_scores1[1] + scores1[0]).exp() +
                    (expected_backward_scores1[1] + scores1[1]).exp() +
                    (expected_backward_scores1[2] + scores1[2]).exp()).log()

                expected_backward_scores2[4] = 0
                # yapf:disable
                expected_backward_scores2[3] = expected_backward_scores2[4] + scores2[7] # noqa
                # yapf:enable
                expected_backward_scores2[2] = (
                    (expected_backward_scores2[3] + scores2[5]).exp() +
                    (expected_backward_scores2[3] + scores2[6]).exp()).log()
                expected_backward_scores2[1] = (
                    (expected_backward_scores2[2] + scores2[2]).exp() +
                    (expected_backward_scores2[2] + scores2[3]).exp() +
                    (expected_backward_scores2[2] + scores2[4]).exp()).log()
                expected_backward_scores2[0] = (
                    (expected_backward_scores2[1] + scores2[0]).exp() +
                    (expected_backward_scores2[1] + scores2[1]).exp()).log()

                expected_backward_scores = torch.cat(
                    [expected_backward_scores1, expected_backward_scores2])
                assert torch.allclose(backward_scores,
                                      expected_backward_scores)

                (scale * backward_scores).sum().backward()
                (scale * expected_backward_scores).sum().backward()
                assert torch.allclose(fsa1.grad, scores1.grad)
                assert torch.allclose(fsa2.grad, scores2.grad)


if __name__ == '__main__':
    unittest.main()
