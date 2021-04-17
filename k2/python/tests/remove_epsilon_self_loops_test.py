#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R remove_epsilon_self_loops_test_py

import unittest

import k2
import torch


class TestRemoveEpsilonSelfLoops(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))

    def test_single_fsa(self):
        for device in self.devices:
            # See https://git.io/JY7r4
            s = '''
                0 1 0 0.1
                0 2 0 0.2
                0 0 0 0.3
                1 1 0 0.4
                1 2 0 0.5
                2 3 -1 0.6
                3
            '''
            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)

            ans = k2.remove_epsilon_self_loops(fsa)

            # See https://git.io/JY7oC
            expected_fsa = k2.Fsa.from_str('''
                0 1 0 0.1
                0 2 0 0.2
                1 2 0 0.5
                2 3 -1 0.6
                3
            ''')
            assert str(ans) == str(expected_fsa)
            (ans.scores.sum() * 2).backward()
            expected_grad = torch.tensor([2, 2, 0, 0, 2,
                                          2.]).to(fsa.scores.grad)
            assert torch.all(torch.eq(fsa.scores.grad, expected_grad))

    def test_fsa_vec(self):
        for device in self.devices:
            # See https://git.io/JY7r4
            s = '''
                0 1 0 0.1
                0 2 0 0.2
                0 0 0 0.3
                1 1 0 0.4
                1 2 0 0.5
                2 3 -1 0.6
                3
            '''
            fsa1 = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa2 = k2.Fsa.from_str(s).to(device).requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

            ans = k2.remove_epsilon_self_loops(fsa_vec)

            # See https://git.io/JY7oC
            expected_fsa = k2.Fsa.from_str('''
                0 1 0 0.1
                0 2 0 0.2
                1 2 0 0.5
                2 3 -1 0.6
                3
            ''')
            assert str(ans[0]) == str(expected_fsa)
            assert str(ans[1]) == str(expected_fsa)
            (ans.scores.sum() * 2).backward()
            expected_grad = torch.tensor([2, 2, 0, 0, 2,
                                          2.]).to(fsa1.scores.grad)
            assert torch.all(torch.eq(fsa1.scores.grad, expected_grad))
            assert torch.all(torch.eq(fsa2.scores.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
