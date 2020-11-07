#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R add_epsilon_self_loops_test_py

import unittest

import k2
import torch


class TestAddEpsilonSelfLoops(unittest.TestCase):

    def test_single_fsa(self):
        s = '''
            0 1 1 0.1
            0 2 1 0.2
            1 3 2 0.3
            2 3 3 0.4
            3 4 -1 0.5
            4
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        new_fsa = k2.add_epsilon_self_loops(fsa)
        assert torch.allclose(
            new_fsa.arcs.values()[:, :3],
            torch.tensor([
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 1],
                [1, 1, 0],
                [1, 3, 2],
                [2, 2, 0],
                [2, 3, 3],
                [3, 3, 0],
                [3, 4, -1],
            ]).to(torch.int32))

        assert torch.allclose(
            new_fsa.scores, torch.tensor([0, 0.1, 0.2, 0, 0.3, 0, 0.4, 0,
                                          0.5]))
        scale = torch.arange(new_fsa.scores.numel())
        (new_fsa.scores * scale).sum().backward()
        assert torch.allclose(fsa.scores.grad,
                              torch.tensor([1., 2., 4., 6., 8.]))

    def test_two_fsas(self):
        s1 = '''
            0 1 1 0.1
            0 2 1 0.2
            1 3 2 0.3
            2 3 3 0.4
            3 4 -1 0.5
            4
        '''
        s2 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            2 3 -1 0.4
            3
        '''

        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)

        fsa1.requires_grad_(True)
        fsa2.requires_grad_(True)

        fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
        new_fsa_vec = k2.add_epsilon_self_loops(fsa_vec)
        assert torch.allclose(
            new_fsa_vec.arcs.values()[:, :3],
            torch.tensor([
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 1],
                [1, 1, 0],
                [1, 3, 2],
                [2, 2, 0],
                [2, 3, 3],
                [3, 3, 0],
                [3, 4, -1],
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 2],
                [1, 1, 0],
                [1, 2, 3],
                [2, 2, 0],
                [2, 3, -1],
            ]).to(torch.int32))

        assert torch.allclose(
            new_fsa_vec.scores,
            torch.tensor([
                0, 0.1, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0, 0.1, 0.2, 0, 0.3, 0,
                0.4
            ]))

        scale = torch.arange(new_fsa_vec.scores.numel())
        (new_fsa_vec.scores * scale).sum().backward()

        assert torch.allclose(fsa1.scores.grad,
                              torch.tensor([1., 2., 4., 6., 8.]))

        assert torch.allclose(fsa2.scores.grad,
                              torch.tensor([10., 11., 13., 15.]))


if __name__ == '__main__':
    unittest.main()
