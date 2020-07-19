#!/usr/bin/env python3
#
# Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_test_py
#

import unittest

import k2


class TestFsa(unittest.TestCase):

    def test_arc(self):
        arc = k2.Arc(1, 2, 3)
        self.assertEqual(arc.src_state, 1)
        self.assertEqual(arc.dest_state, 2)
        self.assertEqual(arc.label, 3)

    def test_fsa(self):
        s = r'''
        0 1 1
        0 2 2
        1 3 3
        2 3 3
        3 4 -1
        4
        '''

        fsa = k2.str_to_fsa(s)
        self.assertEqual(fsa.num_states(), 5)
        self.assertEqual(fsa.final_state(), 4)
        self.assertFalse(fsa.empty())
        self.assertIsInstance(fsa, k2.Fsa)
        self.assertIsInstance(fsa.data(0), k2.Arc);
        self.assertEqual(fsa.data(0).src_state, 0)
        self.assertEqual(fsa.data(0).dest_state, 1)
        self.assertEqual(fsa.data(0).label, 1)


if __name__ == '__main__':
    unittest.main()
