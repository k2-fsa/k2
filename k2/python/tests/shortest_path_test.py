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

    def test(self):
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
        fsa.score.requires_grad_(True)
        best_path = k2.shortest_path(fsa)
        distance = best_path.score.sum()
        assert distance == 14
        expected = torch.zeros(12)
        expected[torch.tensor([1, 3, 5, 10])] = 1
        distance.backward()
        assert torch.allclose(fsa.score.grad, expected)


if __name__ == '__main__':
    unittest.main()
