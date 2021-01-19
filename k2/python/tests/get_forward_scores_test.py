#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R get_forward_scores_test_py

import unittest

import k2
import torch


class TestGetForwardScores(unittest.TestCase):

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
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_scores = torch.tensor([
                    0,  # start state
                    0.1,  # state 1, arc: 0 -> 1 (2/0.1)
                    2.2,  # state 2, arc: 0 -> 2 (3/2.2)
                    3.1,  # state 3, arc: 1 -> 3 (-1/3.0)
                ]).to(forward_scores)
                assert torch.allclose(forward_scores, expected_scores)
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
                expected_scores = torch.empty_like(forward_scores)
                expected_scores[0] = 0
                expected_scores[1] = scores[:2].exp().sum().log()
                expected_scores[2] = (
                    scores[2].exp() + (expected_scores[1] + scores[3]).exp() +
                    (expected_scores[1] + scores[4]).exp()).log()
                expected_scores[3] = (
                    (expected_scores[1] + scores[5]).exp() +
                    (expected_scores[2] + scores[6]).exp()).log()

                (scale * forward_scores).sum().backward()
                (scale * expected_scores).sum().backward()
                assert torch.allclose(forward_scores, expected_scores)
                assert torch.allclose(fsa.grad, scores.grad)


if __name__ == '__main__':
    unittest.main()
