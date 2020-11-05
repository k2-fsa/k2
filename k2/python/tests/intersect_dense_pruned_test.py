#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R intersect_dense_pruned_test_py

import unittest

import _k2
import k2
import torch


class TestIntersectDensePruned(unittest.TestCase):

    def test_simple(self):
        s = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        fsa_vec = k2.create_fsa_vec([fsa])
        log_prob = torch.tensor([[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06]]],
                                dtype=torch.float32,
                                requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 2]], dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                            dense_fsa_vec,
                                            beam=100000,
                                            max_active_states=10000,
                                            min_active_states=0)

        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_float_scores=True)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        expected_scores_out_fsa = torch.tensor([1.2, 2.06, 3.0])
        expected_grad_fsa = torch.tensor([1.0, 0.0, 1.0, 1.0])
        expected_grad_log_prob = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0,
                                               1.0]).reshape_as(log_prob)
        assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa, fsa.scores.grad)
        assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_dense(self):
        s = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        fsa_vec = k2.create_fsa_vec([fsa])
        log_prob = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
             [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 2], [1, 0, 3]],
                                            dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                            dense_fsa_vec,
                                            beam=100000,
                                            max_active_states=10000,
                                            min_active_states=0)
        assert out_fsa.shape == (2, None, None), 'There should be two FSAs!'

        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_float_scores=True)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        expected_scores_out_fsa = torch.tensor(
            [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0])
        expected_grad_fsa = torch.tensor([2.0, 1.0, 2.0, 2.0])
        expected_grad_log_prob = torch.tensor([
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0
        ]).reshape_as(log_prob)

        assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa, fsa.scores.grad)
        assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_fsas(self):
        s1 = '''
            0 1 1 1.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        s2 = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)

        fsa1.requires_grad_(True)
        fsa2.requires_grad_(True)

        fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

        log_prob = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
             [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 2], [1, 0, 3]],
                                            dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                            dense_fsa_vec,
                                            beam=100000,
                                            max_active_states=10000,
                                            min_active_states=0)
        assert out_fsa.shape == (2, None, None), 'There should be two FSAs!'

        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_float_scores=True)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        expected_scores_out_fsa = torch.tensor(
            [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0])

        expected_grad_fsa1 = torch.tensor([1.0, 1.0, 1.0])
        expected_grad_fsa2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        expected_grad_log_prob = torch.tensor([
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0
        ]).reshape_as(log_prob)

        assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa1, fsa1.scores.grad)
        assert torch.allclose(expected_grad_fsa2, fsa2.scores.grad)
        assert torch.allclose(expected_grad_log_prob, log_prob.grad)


if __name__ == '__main__':
    unittest.main()
