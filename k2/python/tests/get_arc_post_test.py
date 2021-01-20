#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R get_arc_post_test_py

import unittest

import k2
import torch


class TestGetArcPost(unittest.TestCase):

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
                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=False)
                scores = fsa.scores.detach().clone().requires_grad_(True)

                forward_scores = torch.empty(4).to(arc_post)
                forward_scores[0] = 0
                forward_scores[1] = scores[1]
                forward_scores[2] = scores[2]
                forward_scores[3] = forward_scores[1] + scores[5]

                backward_scores = torch.empty(4).to(arc_post)
                backward_scores[3] = 0
                backward_scores[2] = backward_scores[3] + scores[6]
                backward_scores[1] = backward_scores[3] + scores[5]
                backward_scores[0] = backward_scores[1] + scores[1]

                expected_arc_post = torch.empty_like(arc_post)
                tot_score = forward_scores[-1] + backward_scores[0]

                # yapf:disable
                expected_arc_post[0] = forward_scores[0] + scores[0] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[1] = forward_scores[0] + scores[1] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[2] = forward_scores[0] + scores[2] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[3] = forward_scores[1] + scores[3] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[4] = forward_scores[1] + scores[4] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[5] = forward_scores[1] + scores[5] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[6] = forward_scores[2] + scores[6] + backward_scores[3] - 0.5 * tot_score # noqa
                # yapf:enable

                assert torch.allclose(arc_post, expected_arc_post)

                scale = torch.arange(arc_post.numel()).to(device)
                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

                # now for log semiring
                fsa.scores.grad = None
                scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])

                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=True)

                forward_scores = torch.empty(4).to(arc_post)
                forward_scores[0] = 0
                forward_scores[1] = scores[:2].exp().sum().log()
                forward_scores[2] = (
                    scores[2].exp() + (forward_scores[1] + scores[3]).exp() +
                    (forward_scores[1] + scores[4]).exp()).log()
                forward_scores[3] = (
                    (forward_scores[1] + scores[5]).exp() +
                    (forward_scores[2] + scores[6]).exp()).log()

                backward_scores = torch.empty(4).to(arc_post)
                backward_scores[3] = 0
                backward_scores[2] = backward_scores[3] + scores[6]
                backward_scores[1] = (
                    (backward_scores[2] + scores[3]).exp() +
                    (backward_scores[2] + scores[4]).exp() +
                    (backward_scores[3] + scores[5]).exp()).log()
                backward_scores[0] = (
                    (backward_scores[1] + scores[0]).exp() +
                    (backward_scores[1] + scores[1]).exp() +
                    (backward_scores[2] + scores[2]).exp()).log()

                expected_arc_post = torch.empty_like(arc_post)
                tot_score = forward_scores[-1] + backward_scores[0]

                # yapf:disable
                expected_arc_post[0] = forward_scores[0] + scores[0] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[1] = forward_scores[0] + scores[1] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[2] = forward_scores[0] + scores[2] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[3] = forward_scores[1] + scores[3] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[4] = forward_scores[1] + scores[4] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[5] = forward_scores[1] + scores[5] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[6] = forward_scores[2] + scores[6] + backward_scores[3] - 0.5 * tot_score # noqa
                # yapf:enable

                assert torch.allclose(arc_post, expected_arc_post)

                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad, atol=1e-6)

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
                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=False)
                scores = fsa.scores.detach().clone().requires_grad_(True)

                forward_scores = torch.empty(5).to(arc_post)
                forward_scores[0] = 0
                forward_scores[1] = forward_scores[0] + scores[1]
                forward_scores[2] = forward_scores[1] + scores[4]
                forward_scores[3] = forward_scores[2] + scores[6]
                forward_scores[4] = forward_scores[3] + scores[7]

                backward_scores = torch.empty(5).to(arc_post)

                backward_scores[4] = 0
                backward_scores[3] = backward_scores[4] + scores[7]
                backward_scores[2] = backward_scores[3] + scores[6]
                backward_scores[1] = backward_scores[2] + scores[4]
                backward_scores[0] = backward_scores[1] + scores[1]

                expected_arc_post = torch.empty_like(arc_post)
                tot_score = forward_scores[-1] + backward_scores[0]
                # yapf:disable
                expected_arc_post[0] = forward_scores[0] + scores[0] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[1] = forward_scores[0] + scores[1] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[2] = forward_scores[1] + scores[2] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[3] = forward_scores[1] + scores[3] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[4] = forward_scores[1] + scores[4] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[5] = forward_scores[2] + scores[5] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[6] = forward_scores[2] + scores[6] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[7] = forward_scores[3] + scores[7] + backward_scores[4] - 0.5 * tot_score # noqa
                # yapf:enable

                assert torch.allclose(arc_post, expected_arc_post)

                scale = torch.arange(arc_post.numel()).to(device)
                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)

                # now for log semiring
                fsa.scores.grad = None
                scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa])

                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=True)

                forward_scores = torch.empty(5).to(arc_post)
                forward_scores[0] = 0
                forward_scores[1] = scores[:2].exp().sum().log()
                forward_scores[2] = (
                    (forward_scores[1] + scores[2]).exp() +
                    (forward_scores[1] + scores[3]).exp() +
                    (forward_scores[1] + scores[4]).exp()).log()
                forward_scores[3] = (
                    (forward_scores[2] + scores[5]).exp() +
                    (forward_scores[2] + scores[6]).exp()).log()
                forward_scores[4] = forward_scores[3] + scores[7]

                backward_scores = torch.empty(5).to(arc_post)
                backward_scores[4] = 0
                backward_scores[3] = backward_scores[4] + scores[7]
                backward_scores[2] = (
                    (backward_scores[3] + scores[5]).exp() +
                    (backward_scores[3] + scores[6]).exp()).log()
                backward_scores[1] = (
                    (backward_scores[2] + scores[2]).exp() +
                    (backward_scores[2] + scores[3]).exp() +
                    (backward_scores[2] + scores[4]).exp()).log()
                backward_scores[0] = (
                    (backward_scores[1] + scores[0]).exp() +
                    (backward_scores[1] + scores[1]).exp()).log()

                expected_arc_post = torch.empty_like(arc_post)
                tot_score = forward_scores[-1] + backward_scores[0]

                # yapf:disable
                expected_arc_post[0] = forward_scores[0] + scores[0] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[1] = forward_scores[0] + scores[1] + backward_scores[1] - 0.5 * tot_score # noqa
                expected_arc_post[2] = forward_scores[1] + scores[2] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[3] = forward_scores[1] + scores[3] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[4] = forward_scores[1] + scores[4] + backward_scores[2] - 0.5 * tot_score # noqa
                expected_arc_post[5] = forward_scores[2] + scores[5] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[6] = forward_scores[2] + scores[6] + backward_scores[3] - 0.5 * tot_score # noqa
                expected_arc_post[7] = forward_scores[3] + scores[7] + backward_scores[4] - 0.5 * tot_score # noqa
                # yapf:enable

                assert torch.allclose(arc_post, expected_arc_post, atol=1e-6)

                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad, atol=1e-5)

    def test_simple_fsa_vec(self):
        # combine case 1 and case 2
        #
        # see https://git.io/JtttZ
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

        # see https://git.io/Jttmm
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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            for use_double_scores in [True, False]:
                fsa1 = k2.Fsa.from_str(s1).to(device).requires_grad_(True)
                fsa2 = k2.Fsa.from_str(s2).to(device).requires_grad_(True)
                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=False)

                scores1 = fsa1.scores.detach().clone().requires_grad_(True)
                scores2 = fsa2.scores.detach().clone().requires_grad_(True)

                forward_scores1 = torch.empty(4).to(arc_post)
                forward_scores1[0] = 0
                forward_scores1[1] = scores1[1]
                forward_scores1[2] = scores1[2]
                forward_scores1[3] = forward_scores1[1] + scores1[5]

                backward_scores1 = torch.empty(4).to(arc_post)
                backward_scores1[3] = 0
                backward_scores1[2] = backward_scores1[3] + scores1[6]
                backward_scores1[1] = backward_scores1[3] + scores1[5]
                backward_scores1[0] = backward_scores1[1] + scores1[1]

                expected_arc_post1 = torch.empty_like(arc_post[:7])
                tot_score1 = forward_scores1[-1] + backward_scores1[0]

                # yapf:disable
                expected_arc_post1[0] = forward_scores1[0] + scores1[0] + backward_scores1[1] - 0.5 * tot_score1 # noqa
                expected_arc_post1[1] = forward_scores1[0] + scores1[1] + backward_scores1[1] - 0.5 * tot_score1 # noqa
                expected_arc_post1[2] = forward_scores1[0] + scores1[2] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[3] = forward_scores1[1] + scores1[3] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[4] = forward_scores1[1] + scores1[4] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[5] = forward_scores1[1] + scores1[5] + backward_scores1[3] - 0.5 * tot_score1 # noqa
                expected_arc_post1[6] = forward_scores1[2] + scores1[6] + backward_scores1[3] - 0.5 * tot_score1 # noqa
                # yapf:enable

                forward_scores2 = torch.empty(5).to(arc_post)
                forward_scores2[0] = 0
                forward_scores2[1] = forward_scores2[0] + scores2[1]
                forward_scores2[2] = forward_scores2[1] + scores2[4]
                forward_scores2[3] = forward_scores2[2] + scores2[6]
                forward_scores2[4] = forward_scores2[3] + scores2[7]

                backward_scores2 = torch.empty(5).to(arc_post)
                backward_scores2[4] = 0
                backward_scores2[3] = backward_scores2[4] + scores2[7]
                backward_scores2[2] = backward_scores2[3] + scores2[6]
                backward_scores2[1] = backward_scores2[2] + scores2[4]
                backward_scores2[0] = backward_scores2[1] + scores2[1]

                expected_arc_post2 = torch.empty_like(arc_post[7:])
                tot_score2 = forward_scores2[-1] + backward_scores2[0]

                # yapf:disable
                expected_arc_post2[0] = forward_scores2[0] + scores2[0] + backward_scores2[1] - 0.5 * tot_score2 # noqa
                expected_arc_post2[1] = forward_scores2[0] + scores2[1] + backward_scores2[1] - 0.5 * tot_score2 # noqa
                expected_arc_post2[2] = forward_scores2[1] + scores2[2] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[3] = forward_scores2[1] + scores2[3] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[4] = forward_scores2[1] + scores2[4] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[5] = forward_scores2[2] + scores2[5] + backward_scores2[3] - 0.5 * tot_score2 # noqa
                expected_arc_post2[6] = forward_scores2[2] + scores2[6] + backward_scores2[3] - 0.5 * tot_score2 # noqa
                expected_arc_post2[7] = forward_scores2[3] + scores2[7] + backward_scores2[4] - 0.5 * tot_score2 # noqa
                # yapf:enable

                expected_arc_post = torch.cat(
                    [expected_arc_post1, expected_arc_post2])
                assert torch.allclose(arc_post, expected_arc_post)

                scale = torch.arange(arc_post.numel()).to(device)
                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa1.grad, scores1.grad)
                assert torch.allclose(fsa2.grad, scores2.grad)

                # now for log semiring
                fsa1.scores.grad = None
                fsa2.scores.grad = None
                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
                arc_post = fsa_vec.get_arc_post(
                    use_double_scores=use_double_scores, log_semiring=True)

                scores1.grad = None
                scores2.grad = None

                forward_scores1 = torch.empty(4).to(arc_post)
                forward_scores1[0] = 0
                forward_scores1[1] = scores1[:2].exp().sum().log()
                forward_scores1[2] = (
                    scores1[2].exp() +
                    (forward_scores1[1] + scores1[3]).exp() +
                    (forward_scores1[1] + scores1[4]).exp()).log()
                forward_scores1[3] = (
                    (forward_scores1[1] + scores1[5]).exp() +
                    (forward_scores1[2] + scores1[6]).exp()).log()

                backward_scores1 = torch.empty(4).to(arc_post)
                backward_scores1[3] = 0
                backward_scores1[2] = backward_scores1[3] + scores1[6]
                backward_scores1[1] = (
                    (backward_scores1[2] + scores1[3]).exp() +
                    (backward_scores1[2] + scores1[4]).exp() +
                    (backward_scores1[3] + scores1[5]).exp()).log()
                backward_scores1[0] = (
                    (backward_scores1[1] + scores1[0]).exp() +
                    (backward_scores1[1] + scores1[1]).exp() +
                    (backward_scores1[2] + scores1[2]).exp()).log()

                expected_arc_post1 = torch.empty_like(arc_post[:7])
                tot_score1 = forward_scores1[-1] + backward_scores1[0]
                # yapf:disable
                expected_arc_post1[0] = forward_scores1[0] + scores1[0] + backward_scores1[1] - 0.5 * tot_score1 # noqa
                expected_arc_post1[1] = forward_scores1[0] + scores1[1] + backward_scores1[1] - 0.5 * tot_score1 # noqa
                expected_arc_post1[2] = forward_scores1[0] + scores1[2] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[3] = forward_scores1[1] + scores1[3] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[4] = forward_scores1[1] + scores1[4] + backward_scores1[2] - 0.5 * tot_score1 # noqa
                expected_arc_post1[5] = forward_scores1[1] + scores1[5] + backward_scores1[3] - 0.5 * tot_score1 # noqa
                expected_arc_post1[6] = forward_scores1[2] + scores1[6] + backward_scores1[3] - 0.5 * tot_score1 # noqa
                # yapf:enable

                forward_scores2 = torch.empty(5).to(arc_post)
                forward_scores2[0] = 0
                forward_scores2[1] = scores2[:2].exp().sum().log()
                forward_scores2[2] = (
                    (forward_scores2[1] + scores2[2]).exp() +
                    (forward_scores2[1] + scores2[3]).exp() +
                    (forward_scores2[1] + scores2[4]).exp()).log()
                forward_scores2[3] = (
                    (forward_scores2[2] + scores2[5]).exp() +
                    (forward_scores2[2] + scores2[6]).exp()).log()
                forward_scores2[4] = forward_scores2[3] + scores2[7]

                backward_scores2 = torch.empty(5).to(arc_post)
                backward_scores2[4] = 0
                backward_scores2[3] = backward_scores2[4] + scores2[7]
                backward_scores2[2] = (
                    (backward_scores2[3] + scores2[5]).exp() +
                    (backward_scores2[3] + scores2[6]).exp()).log()
                backward_scores2[1] = (
                    (backward_scores2[2] + scores2[2]).exp() +
                    (backward_scores2[2] + scores2[3]).exp() +
                    (backward_scores2[2] + scores2[4]).exp()).log()
                backward_scores2[0] = (
                    (backward_scores2[1] + scores2[0]).exp() +
                    (backward_scores2[1] + scores2[1]).exp()).log()

                expected_arc_post2 = torch.empty_like(arc_post[7:])
                tot_score2 = forward_scores2[-1] + backward_scores2[0]
                # yapf:disable
                expected_arc_post2[0] = forward_scores2[0] + scores2[0] + backward_scores2[1] - 0.5 * tot_score2 # noqa
                expected_arc_post2[1] = forward_scores2[0] + scores2[1] + backward_scores2[1] - 0.5 * tot_score2 # noqa
                expected_arc_post2[2] = forward_scores2[1] + scores2[2] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[3] = forward_scores2[1] + scores2[3] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[4] = forward_scores2[1] + scores2[4] + backward_scores2[2] - 0.5 * tot_score2 # noqa
                expected_arc_post2[5] = forward_scores2[2] + scores2[5] + backward_scores2[3] - 0.5 * tot_score2 # noqa
                expected_arc_post2[6] = forward_scores2[2] + scores2[6] + backward_scores2[3] - 0.5 * tot_score2 # noqa
                expected_arc_post2[7] = forward_scores2[3] + scores2[7] + backward_scores2[4] - 0.5 * tot_score2 # noqa
                # yapf:enable

                expected_arc_post = torch.cat(
                    [expected_arc_post1, expected_arc_post2])
                assert torch.allclose(arc_post, expected_arc_post, atol=1e-6)

                (scale * arc_post).sum().backward()
                (scale * expected_arc_post).sum().backward()
                assert torch.allclose(fsa1.grad, scores1.grad, atol=1e-5)
                assert torch.allclose(fsa2.grad, scores2.grad, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
