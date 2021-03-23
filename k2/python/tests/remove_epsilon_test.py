#!/usr/bin/env python3
#
# Copyright (c)  2020-2021  Xiaomi Corporation (authors: Haowen Qiu
#                                                        Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R remove_epsilon_test_py

import unittest

import torch
import k2


class TestRemoveEpsilonHost(unittest.TestCase):

    def test1(self):
        s = '''
            0 4 1 1
            0 1 1 1
            1 2 0 2
            1 3 0 3
            1 4 0 2
            2 7 0 4
            3 7 0 5
            4 6 1 2
            4 6 0 3
            4 8 1 3
            4 9 -1 2
            5 9 -1 4
            6 9 -1 3
            7 9 -1 5
            8 9 -1 6
            9
        '''
        fsa = k2.Fsa.from_str(s)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))

    def test_autograd(self):
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa = k2.Fsa.from_str(s).requires_grad_(True)
        ans = k2.remove_epsilon(fsa)
        # arc map is [[1] [0 2] [2]]
        scale = torch.tensor([10, 20, 30])

        (ans.scores * scale).sum().backward()
        expected_grad = torch.empty_like(fsa.scores)
        expected_grad[0] = scale[1]
        expected_grad[1] = scale[0]
        expected_grad[2] = scale[1] + scale[2]
        assert torch.all(torch.eq(fsa.grad, expected_grad))


class TestRemoveEpsilonDevice(unittest.TestCase):

    def test1(self):
        if not torch.cuda.is_available():
            return
        device = torch.device('cuda', 0)
        s = '''
            0 1 0 1 1
            1 2 0 2 1
            2 3 0 3 1
            3 4 4 4 1
            3 5 -1 5 1
            4 5 -1 6 1
            5
        '''
        fsa = k2.Fsa.from_str(s, num_aux_labels=1).to(device)
        print(fsa.aux_labels)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))

        # just make sure that it runs.
        dest2 = k2.remove_epsilon_and_add_self_loops(fsa)
        dest3 = k2.remove_epsilon(dest2)

        self.assertTrue(
            k2.is_rand_equivalent(dest,
                                  dest3,
                                  log_semiring,
                                  treat_epsilons_specially=False))
        self.assertFalse(
            k2.is_rand_equivalent(dest,
                                  dest2,
                                  log_semiring,
                                  treat_epsilons_specially=False,
                                  npath=10000))
        self.assertTrue(
            k2.is_rand_equivalent(dest,
                                  dest2,
                                  log_semiring,
                                  treat_epsilons_specially=True))

    def test_autograd(self):
        if not torch.cuda.is_available():
            return
        device = torch.device('cuda', 0)
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
        ans = k2.remove_epsilon(fsa)
        # arc map is [[0 2] [1] [2]]
        scale = torch.tensor([10, 20, 30]).to(device)

        (ans.scores * scale).sum().backward()
        expected_grad = torch.empty_like(fsa.scores)
        expected_grad[0] = scale[0]
        expected_grad[1] = scale[1]
        expected_grad[2] = scale[0] + scale[2]
        assert torch.all(torch.eq(fsa.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
