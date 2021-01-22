#!/usr/bin/env python3
#
# Copyright (c)  2021  Mobvoi Inc.        (authors: Yaguang Hu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R linear_fst_test_py

import unittest

import k2
import torch


class TestLinearFst(unittest.TestCase):

    def test_single_fst(self):
        labels = [2, 5, 8]
        aux_labels = [3, 6, 9]
        fst = k2.linear_fst(labels, aux_labels)
        assert len(fst.shape) == 2
        assert fst.shape[0] == len(labels) + 2, 'There should be 5 states'
        assert fst.aux_labels.shape[0] == len(aux_labels)
        assert torch.allclose(fst.scores,
                              torch.zeros(len(labels) + 1, dtype=torch.float32))
        assert torch.allclose(
            fst.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor([[0, 1, 2], [1, 2, 5], [2, 3, 8], [3, 4, -1]],
                         dtype=torch.int32))
        assert torch.allclose(fst.aux_labels, torch.IntTensor(aux_labels))

    def test_fst_vec(self):
        labels = [
            [1, 3, 5],
            [2, 6],
            [8, 7, 9],
        ]
        aux_labels = [
            [2, 4, 6],
            [6, 2],
            [8, 7, 9],
        ]
        num_labels = sum([len(s) for s in labels])
        fst = k2.linear_fst(labels, aux_labels)
        assert len(fst.shape) == 3
        assert fst.shape[0] == 3, 'There should be 3 FSTs'
        assert torch.allclose(
            fst.scores,
            torch.zeros(num_labels + len(labels), dtype=torch.float32))
        expected_aux_labels = [2, 4, 6, -1, 6, 2, -1, 8, 7, 9, -1]
        assert torch.allclose(fst.aux_labels,
                              torch.IntTensor(expected_aux_labels))


if __name__ == '__main__':
    unittest.main()
