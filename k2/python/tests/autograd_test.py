#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R autograd_test_py

import unittest

import _k2
import k2
import torch


class TestGetForwardLogLike(unittest.TestCase):

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
        fsa = k2.Fsa.from_str(s)
        fsa = k2.create_fsa_vec([fsa])
        fsa.scores.requires_grad_(True)
        log_like = k2.get_forward_log_like(fsa,
                                           log_semiring=False,
                                           use_float_scores=True)

        assert log_like == 14
        assert log_like.dtype == torch.float32

        log_like.sum().backward()
        expected = torch.zeros(len(fsa.scores))
        expected[torch.tensor([1, 3, 5, 10])] = 1
        assert torch.allclose(fsa.scores.grad, expected)

        # now for double
        fsa.scores.grad = None
        log_like = k2.get_forward_log_like(fsa,
                                           log_semiring=False,
                                           use_float_scores=False)
        assert log_like == 14
        assert log_like.dtype == torch.float64
        log_like.sum().backward()
        assert torch.allclose(fsa.scores.grad, expected)

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

        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)
        fsa3 = k2.Fsa.from_str(s3)

        fsa1.scores.requires_grad_(True)
        fsa2.scores.requires_grad_(True)
        fsa3.scores.requires_grad_(True)

        fsa_vec = k2.create_fsa_vec([fsa1, fsa2, fsa3])

        log_like = k2.get_forward_log_like(fsa_vec,
                                           log_semiring=False,
                                           use_float_scores=True)
        expected_log_like = torch.tensor([14, 13, 105.5])
        assert torch.allclose(log_like, expected_log_like)

        log_like.sum().backward()

        fsa1_best_arc_indexes = torch.tensor([1, 3, 5, 10])
        assert torch.allclose(fsa1.scores.grad[fsa1_best_arc_indexes],
                              torch.ones(4, dtype=torch.float32))
        assert fsa1.scores.grad.sum() == 4

        fsa2_best_arc_indexes = torch.tensor([1, 4, 5, 7])
        assert torch.allclose(fsa2.scores.grad[fsa2_best_arc_indexes],
                              torch.ones(4, dtype=torch.float32))
        assert fsa2.scores.grad.sum() == 4

        fsa3_best_arc_indexes = torch.tensor([1, 3])
        assert torch.allclose(fsa3.scores.grad[fsa3_best_arc_indexes],
                              torch.ones(2, dtype=torch.float32))
        assert fsa3.scores.grad.sum() == 2

        # now for double
        fsa1.scores.grad = None
        fsa2.scores.grad = None
        fsa3.scores.grad = None
        log_like = k2.get_forward_log_like(fsa_vec,
                                           log_semiring=False,
                                           use_float_scores=False)

        expected_log_like = expected_log_like.to(torch.float64)
        assert torch.allclose(log_like, expected_log_like)

        log_like.sum().backward()

        assert torch.allclose(fsa1.scores.grad[fsa1_best_arc_indexes],
                              torch.ones(4, dtype=torch.float32))
        assert fsa1.scores.grad.sum() == 4

        assert torch.allclose(fsa2.scores.grad[fsa2_best_arc_indexes],
                              torch.ones(4, dtype=torch.float32))
        assert fsa2.scores.grad.sum() == 4

        assert torch.allclose(fsa3.scores.grad[fsa3_best_arc_indexes],
                              torch.ones(2, dtype=torch.float32))
        assert fsa3.scores.grad.sum() == 2


if __name__ == '__main__':
    unittest.main()
