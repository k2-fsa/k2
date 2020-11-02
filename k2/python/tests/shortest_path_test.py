#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R shortest_path_test_py

import unittest

import k2
import torch


class TestShortestPath(unittest.TestCase):

    def test_single_fsa(self):
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
        fsa.scores.requires_grad_(True)
        best_path, _ = k2.shortest_path(fsa)

        # we recompute the total_scores for backprop
        total_scores = best_path.scores.sum()

        assert total_scores == 14
        expected = torch.zeros(12)
        expected[torch.tensor([1, 3, 5, 10])] = 1
        total_scores.backward()
        assert torch.allclose(fsa.scores.grad, expected)


if __name__ == '__main__':
    unittest.main()
