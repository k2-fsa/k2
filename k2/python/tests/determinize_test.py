#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R determinize_test.py

import unittest

import k2
import torch


class TestDeterminize(unittest.TestCase):

    def test1(self):
        s = '''
            0 4 1 1
            0 1 1 1
            1 2 2 2
            1 3 3 3
            2 7 1 4
            3 7 1 5
            4 6 1 2
            4 6 1 3
            4 5 1 3
            4 8 -1 2
            5 8 -1 4
            6 8 -1 3
            7 8 -1 5
            8
        '''
        fsa = k2.Fsa.from_str(s)
        prop = k2.get_properties(fsa)
        self.assertFalse(k2.is_arc_sorted_and_deterministic(prop))
        dest = k2.determinize(fsa)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))
        arc_sorted = k2.arc_sort(dest)
        prop = k2.get_properties(arc_sorted)
        self.assertTrue(k2.is_arc_sorted_and_deterministic(prop))


if __name__ == '__main__':
    unittest.main()
