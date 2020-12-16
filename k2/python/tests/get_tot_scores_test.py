#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R get_tot_scores_test_py

import unittest

import k2
import torch


class TestGetTotScores(unittest.TestCase):

    def test_tropical_single_fsa(self):
        # best path arc indexes are: 1, 3, 5, 10
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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa = k2.create_fsa_vec([fsa])
            fsa.requires_grad_(True)
            log_like = k2.get_tot_scores(fsa,
                                         log_semiring=False,
                                         use_double_scores=False)

            assert log_like == 14
            assert log_like.dtype == torch.float32

            scale = -10

            (scale * log_like).sum().backward()
            expected = torch.zeros(len(fsa.scores)).to(device)
            expected[torch.tensor([1, 3, 5, 10])] = 1
            assert torch.allclose(fsa.scores.grad, scale * expected)

            # now for double
            fsa.scores.grad = None
            log_like = k2.get_tot_scores(fsa,
                                         log_semiring=False,
                                         use_double_scores=True)
            assert log_like == 14
            assert log_like.dtype == torch.float64

            scale = -1.25
            (scale * log_like).sum().backward()
            assert torch.allclose(fsa.scores.grad, scale * expected)

    def test_tropical_multiple_fsas(self):
        # best path:
        #  states: 0 -> 1 -> 3 -> 7 -> 9
        #  arcs:     1 -> 3 -> 5 -> 10
        #  scores: 1 + 3 + 5 + 5 = 14
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
        #   scores: 6 + 4 + 3 + 0 = 13
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
        #   scores: 100 + 5.5 = 105.5
        s3 = '''
            0 1 1 10
            0 2 2 100
            1 3 -1 3.5
            2 3 -1 5.5
            3
        '''

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)
            fsa3 = k2.Fsa.from_str(s3).to(device)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)
            fsa3.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2, fsa3])

            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=False,
                                         use_double_scores=False)
            assert log_like.dtype == torch.float32
            expected_log_like = torch.tensor([14, 13, 105.5]).to(device)
            assert torch.allclose(log_like, expected_log_like)

            scale = torch.tensor([-10, -20, -30.]).to(log_like)
            (log_like * scale).sum().backward()

            fsa1_best_arc_indexes = torch.tensor([1, 3, 5, 10]).to(device)
            assert torch.allclose(
                fsa1.scores.grad[fsa1_best_arc_indexes],
                scale[0] * torch.ones(4, dtype=torch.float32).to(device))
            assert fsa1.scores.grad.sum() == 4 * scale[0]

            fsa2_best_arc_indexes = torch.tensor([1, 4, 5, 7]).to(device)
            assert torch.allclose(
                fsa2.scores.grad[fsa2_best_arc_indexes],
                scale[1] * torch.ones(4, dtype=torch.float32).to(device))
            assert fsa2.scores.grad.sum() == 4 * scale[1]

            fsa3_best_arc_indexes = torch.tensor([1, 3]).to(device)
            assert torch.allclose(
                fsa3.scores.grad[fsa3_best_arc_indexes],
                scale[2] * torch.ones(2, dtype=torch.float32).to(device))
            assert fsa3.scores.grad.sum() == 2 * scale[2]

            # now for double
            fsa1.scores.grad = None
            fsa2.scores.grad = None
            fsa3.scores.grad = None
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=False,
                                         use_double_scores=True)

            assert log_like.dtype == torch.float64
            expected_log_like = expected_log_like.to(torch.float64)
            assert torch.allclose(log_like, expected_log_like)

            scale = torch.tensor([-1.25, -2.5, 3.5]).to(log_like)
            (scale * log_like).sum().backward()

            assert torch.allclose(
                fsa1.scores.grad[fsa1_best_arc_indexes],
                scale[0] * torch.ones(4, dtype=torch.float32).to(device))
            assert fsa1.scores.grad.sum() == 4 * scale[0]

            assert torch.allclose(
                fsa2.scores.grad[fsa2_best_arc_indexes],
                scale[1] * torch.ones(4, dtype=torch.float32).to(device))
            assert fsa2.scores.grad.sum() == 4 * scale[1]

            assert torch.allclose(
                fsa3.scores.grad[fsa3_best_arc_indexes],
                scale[2] * torch.ones(2, dtype=torch.float32).to(device))
            assert fsa3.scores.grad.sum() == 2 * scale[2]

    def test_log_single_fsa(self):
        s = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            1 3 4 0.4
            2 3 5 0.5
            3 4 -1 0
            4
        '''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            fsa_vec = k2.create_fsa_vec([fsa])
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=True,
                                         use_double_scores=False)
            assert log_like.dtype == torch.float32
            # The expected_log_like is computed using gtn.
            # See https://bit.ly/3oUiRx9
            expected_log_like = torch.tensor([1.8119014501571655]).to(device)
            assert torch.allclose(log_like, expected_log_like)

            # The expected_grad is computed using gtn.
            # See https://bit.ly/3oUiRx9
            expected_grad = torch.tensor([
                0.6710670590400696, 0.32893291115760803, 0.4017595648765564,
                0.2693074941635132, 0.7306925058364868, 1.0
            ]).to(device)

            scale = -1.75
            (scale * log_like).sum().backward()
            assert torch.allclose(fsa.scores.grad, scale * expected_grad)

            # now for double
            fsa.scores.grad = None
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=True,
                                         use_double_scores=True)
            assert log_like.dtype == torch.float64
            expected_log_like = expected_log_like.to(torch.float64)
            assert torch.allclose(log_like, expected_log_like)

            scale = 10.25
            (scale * log_like).sum().backward()
            assert torch.allclose(fsa.scores.grad, scale * expected_grad)

    def test_log_multiple_fsas(self):
        s1 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            1 3 4 0.4
            2 3 5 0.5
            3 4 -1 0
            4
        '''

        s2 = '''
            0 3 3 0.1
            0 1 1 0.2
            0 2 2 0.3
            1 2 2 0.4
            1 3 3 0.5
            2 3 3 0.6
            2 4 4 0.7
            3 4 4 0.8
            3 5 -1 0.9
            4 5 -1 1.0
            5
        '''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=True,
                                         use_double_scores=False)
            assert log_like.dtype == torch.float32
            # The expected_log_likes are computed using gtn.
            # See https://bit.ly/3oUiRx9
            expected_log_like = torch.tensor(
                [1.8119014501571655, 4.533502578735352]).to(device)
            assert torch.allclose(log_like, expected_log_like)

            scale = torch.tensor([1.25, -5.25]).to(log_like)
            (scale * log_like).sum().backward()

            # The expected_grads are computed using gtn.
            # See https://bit.ly/3oUiRx9
            expected_grad_fsa1 = torch.tensor([
                0.6710670590400696, 0.32893291115760803, 0.4017595648765564,
                0.2693074941635132, 0.7306925058364868, 1.0
            ]).to(device)

            expected_grad_fsa2 = torch.tensor([
                0.10102888941764832, 0.5947467088699341, 0.3042244613170624,
                0.410660058259964, 0.1840866357088089, 0.5283515453338623,
                0.18653297424316406, 0.5783339142799377, 0.2351330667734146,
                0.764866828918457
            ]).to(device)

            assert torch.allclose(fsa1.scores.grad,
                                  scale[0] * expected_grad_fsa1)
            assert torch.allclose(fsa2.scores.grad,
                                  scale[1] * expected_grad_fsa2)

            # now for double
            fsa1.scores.grad = None
            fsa2.scores.grad = None

            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=True,
                                         use_double_scores=True)
            assert log_like.dtype == torch.float64
            expected_log_like = expected_log_like.to(torch.float64)
            assert torch.allclose(log_like, expected_log_like)

            scale = torch.tensor([-10.25, 8.25]).to(log_like)
            (scale * log_like).sum().backward()

            assert torch.allclose(fsa1.scores.grad,
                                  scale[0] * expected_grad_fsa1)
            assert torch.allclose(fsa2.scores.grad,
                                  scale[1] * expected_grad_fsa2)


if __name__ == '__main__':
    unittest.main()
