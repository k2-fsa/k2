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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            labels = [2, 5, 8]
            fsa = k2.linear_fsa(labels, device)
            assert fsa.device == device
            assert len(fsa.shape) == 2
            assert fsa.shape[0] == len(labels) + 2, 'There should be 5 states'

            assert torch.all(torch.eq(fsa.scores,
                                      torch.zeros_like(fsa.scores)))

            assert torch.all(
                torch.eq(
                    fsa.arcs.values()[:, :-1],  # skip the last field `scores`
                    torch.tensor([[0, 1, 2], [1, 2, 5], [2, 3, 8], [3, 4, -1]],
                                 dtype=torch.int32,
                                 device=device)))

    def test_fsa_vec(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            labels = [
                [1, 3, 5],
                [2, 6],
                [8, 7, 9],
            ]
            fsa = k2.linear_fsa(labels, device)
            assert len(fsa.shape) == 3
            assert fsa.device == device
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
            assert torch.all(
                torch.eq(
                    fsa.arcs.values()[:, :-1],  # skip the last field `scores`
                    torch.tensor(expected_arcs,
                                 dtype=torch.int32,
                                 device=device)))

            assert torch.all(torch.eq(fsa.scores,
                                      torch.zeros_like(fsa.scores)))

    def test_from_ragged_int_single_fsa(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            ragged_int = k2.RaggedInt('[ [10 20] ]').to(device)
            fsa = k2.linear_fsa(ragged_int)
            assert fsa.shape == (1, None, None)
            assert fsa.device == device
            expected_arcs = torch.tensor([[0, 1, 10], [1, 2, 20], [2, 3, -1]],
                                         dtype=torch.int32,
                                         device=device)
            assert torch.all(
                torch.eq(
                    fsa.arcs.values()[:, :-1],  # skip the last field `scores`
                    expected_arcs))

            assert torch.all(torch.eq(fsa.scores,
                                      torch.zeros_like(fsa.scores)))

    def test_from_ragged_int_two_fsas(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            ragged_int = k2.RaggedInt('[ [10 20] [100 200 300] ]').to(device)
            fsa = k2.linear_fsa(ragged_int)
            assert fsa.shape == (2, None, None)
            assert fsa.device == device
            expected_arcs = torch.tensor(
                [[0, 1, 10], [1, 2, 20], [2, 3, -1], [0, 1, 100], [1, 2, 200],
                 [2, 3, 300], [3, 4, -1]],
                dtype=torch.int32,
                device=device)
            assert torch.all(
                torch.eq(
                    fsa.arcs.values()[:, :-1],  # skip the last field `scores`
                    expected_arcs))

            assert torch.all(torch.eq(fsa.scores,
                                      torch.zeros_like(fsa.scores)))


if __name__ == '__main__':
    unittest.main()
