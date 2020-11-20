#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R remove_epsilon_test_py

import unittest

import k2
import torch


class TestRemoveEpsilon(unittest.TestCase):

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
        prop = k2.get_properties(fsa)
        self.assertFalse(k2.is_epsilon_free(prop))
        dest = k2.remove_epsilon(fsa)
        prop = k2.get_properties(dest)
        self.assertTrue(k2.is_epsilon_free(prop))
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))


if __name__ == '__main__':
    unittest.main()
