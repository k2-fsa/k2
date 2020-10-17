#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R linear_fsa_test_py

import unittest

import k2
import torch


class TestLinearFsa(unittest.TestCase):

    def test_single_fsa(self):
        symbols = [2, 5, 8]
        fsa = k2.linear_fsa(symbols)
        assert len(fsa.shape) == 2
        assert fsa.shape[0] == len(symbols) + 2, 'There should be 5 states'
        assert torch.allclose(
            fsa.score, torch.zeros(len(symbols) + 1, dtype=torch.float32))
        assert torch.allclose(
            fsa.arcs.values()[:, :-1],  # skip the last field `score`
            torch.tensor([[0, 1, 2], [1, 2, 5], [2, 3, 8], [3, 4, -1]],
                         dtype=torch.int32))

    def test_fsa_vec(self):
        symbols = [
            [1, 3, 5],
            [2, 6],
            [8, 7, 9],
        ]
        num_symbols = sum([len(s) for s in symbols])
        fsa = k2.linear_fsa(symbols)
        assert len(fsa.shape) == 3
        assert fsa.shape[0] == 3, 'There should be 3 FSAs'
        expected_arcs = [
            # fsa 0
            [0, 1, 1],
            [1, 2, 3],
            [2, 3, 5],
            [3, 4, -1],
            # fsa 1
            [0, 1, 2],
            [1, 2, 6],
            [2, 3, -1],
            # fsa 2
            [0, 1, 8],
            [1, 2, 7],
            [2, 3, 9],
            [3, 4, -1]
        ]
        print(fsa.arcs.values()[:, :-1])
        assert torch.allclose(
            fsa.arcs.values()[:, :-1],  # skip the last field `score`
            torch.tensor(expected_arcs, dtype=torch.int32))
        assert torch.allclose(
            fsa.score,
            torch.zeros(num_symbols + len(symbols), dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
