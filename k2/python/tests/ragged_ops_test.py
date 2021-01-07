#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R ragged_ops_test_py

import unittest

import _k2
import k2
import numpy as np
import torch


class TestRaggedOps(unittest.TestCase):

    def test_remove_axis(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 0 ] [ 3 0 ] [ 2 ] ]')

        ans = k2.ragged.remove_axis(src, 1)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] ]')

    def test_to_list(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [ 3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(k2.ragged.to_list(ans), [[1, 2], [0], [3, 0], [2]])

    def test_remove_values_leq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_leq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 6)
        self.assertEqual(str(ans), '[ [ ] [ ] [ 8 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 8)
        self.assertEqual(str(ans), '[ [ ] [ ] [ ] [ ] ]')

    def test_remove_values_eq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_eq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_eq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 0 ] [ 3 0 2 ] [ 0 8 0 6 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 6)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 8 0 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 8)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 0 6 0 ] [ 0 ] ]')

    def test_normalize_scores_non_zero_stride(self):
        s = '''
            [ [1 -1 0] [2 10] [] [3] [5 8] ]
        '''
        src = k2.ragged.RaggedFloat(s)
        saved = src.values.clone().detach()
        saved.requires_grad_(True)
        src.requires_grad_(True)

        ans = k2.ragged.normalize_scores(src)

        scale = torch.arange(ans.values.numel())

        # the stride of grad is not 0
        (ans.values * scale).sum().backward()

        expected = saved.new_zeros(*ans.values.shape)

        normalizer = saved[:3].exp().sum().log()
        expected[:3] = saved[:3] - normalizer

        normalizer = saved[3:5].exp().sum().log()
        expected[3:5] = saved[3:5] - normalizer

        expected[5] = 0  # it has only one entry

        normalizer = saved[6:8].exp().sum().log()
        expected[6:8] = saved[6:8] - normalizer

        self.assertTrue(torch.allclose(expected, ans.values))
        (expected * scale).sum().backward()

        self.assertTrue(torch.allclose(saved.grad, src.grad))

    def test_normalize_scores_zero_stride(self):
        s = '''
            [ [1 3 5] [2 -1] [] [3] [5 2] ]
        '''
        src = k2.ragged.RaggedFloat(s)
        saved = src.values.clone().detach()
        saved.requires_grad_(True)
        src.requires_grad_(True)

        ans = k2.ragged.normalize_scores(src)

        # the stride of grad is 0
        ans.values.sum().backward()

        expected = saved.new_zeros(*ans.values.shape)

        normalizer = saved[:3].exp().sum().log()
        expected[:3] = saved[:3] - normalizer

        normalizer = saved[3:5].exp().sum().log()
        expected[3:5] = saved[3:5] - normalizer

        expected[5] = 0  # it has only one entry

        normalizer = saved[6:8].exp().sum().log()
        expected[6:8] = saved[6:8] - normalizer

        self.assertTrue(torch.allclose(expected, ans.values))
        expected.sum().backward()

        self.assertTrue(torch.allclose(saved.grad, src.grad))

    def test_normalize_scores_from_shape(self):
        s = '''
            0 1 1 0.
            0 1 2 0.
            0 1 3 0.
            1 2 4 0.
            1 2 5 0.
            2 3 -1 0.
            3
        '''
        fsa = k2.Fsa.from_str(s)
        scores = torch.arange(fsa.scores.numel(), dtype=torch.float32)
        scores.requires_grad_(True)

        ragged_scores = k2.ragged.RaggedFloat(fsa.arcs.shape(), scores)
        assert ragged_scores.requires_grad is True

        normalized_scores = k2.ragged.normalize_scores(ragged_scores)
        assert normalized_scores.requires_grad is True

        fsa.scores = normalized_scores.values
        assert fsa.scores.requires_grad is True

        # arcs leaving state 0
        self.assertAlmostEqual(fsa.scores[:3].exp().sum().item(),
                               1.0,
                               places=6)

        # arcs leaving state 1
        self.assertAlmostEqual(fsa.scores[3:5].exp().sum().item(),
                               1.0,
                               places=6)

        # arcs leaving state 2
        self.assertAlmostEqual(fsa.scores[5].exp().sum().item(), 1.0, places=6)

    def test_sum_per_sublist(self):
        s = '''
            0 1 1 0.
            0 1 2 0.
            0 1 3 0.
            1 2 4 0.
            1 2 5 0.
            2 3 -1 0.
            3
        '''
        fsa = k2.Fsa.from_str(s)
        scores = torch.randn_like(fsa.scores)
        fsa.set_scores_stochastic_(scores)
        normalized_scores = k2.ragged.sum_per_sublist(
            _k2.RaggedFloat(fsa.arcs.shape(), fsa.scores.exp()))
        assert normalized_scores.numel() == fsa.arcs.dim0()

        assert torch.allclose(normalized_scores[:-1],
                              torch.ones(normalized_scores.numel() - 1))

        # the final state has no leaving arcs
        assert normalized_scores[-1].item() == 0


if __name__ == '__main__':
    unittest.main()
