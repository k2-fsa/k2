#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R get_backward_scores_test_py

import unittest

import k2
import torch


class TestGetBackwardScores(unittest.TestCase):

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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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
                expected_backward_scores[1] = expected_backward_scores[3] + scores[5] # noaq
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

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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


if __name__ == '__main__':
    unittest.main()
