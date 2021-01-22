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
        labels = [2, 5, 8]
        fsa = k2.linear_fsa(labels)
        assert len(fsa.shape) == 2
        assert fsa.shape[0] == len(labels) + 2, 'There should be 5 states'
        assert torch.allclose(
            fsa.scores, torch.zeros(len(labels) + 1, dtype=torch.float32))
        assert torch.allclose(
            fsa.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor([[0, 1, 2], [1, 2, 5], [2, 3, 8], [3, 4, -1]],
                         dtype=torch.int32))

    def test_fsa_vec(self):
        labels = [
            [1, 3, 5],
            [2, 6],
            [8, 7, 9],
        ]
        num_labels = sum([len(s) for s in labels])
        fsa = k2.linear_fsa(labels)
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
            fsa.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor(expected_arcs, dtype=torch.int32))
        assert torch.allclose(
            fsa.scores,
            torch.zeros(num_labels + len(labels), dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
