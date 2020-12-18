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
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))


class TestRemoveEpsilonsIterativeTropical(unittest.TestCase):

    def test1(self):
        s = '''
            0 1 0 1 1
            1 2 0 2 1
            2 3 0 3 1
            3 4 4 4 1
            3 5 -1 5 1
            4 5 -1 6 1
            5
        '''
        fsa = k2.Fsa.from_str(s)
        print(fsa.aux_labels)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilons_iterative_tropical(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))


if __name__ == '__main__':
    unittest.main()
