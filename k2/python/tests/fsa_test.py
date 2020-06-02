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

        fsa = k2.string_to_fsa(s)
        self.assertEqual(fsa.num_states(), 5)
        self.assertEqual(fsa.final_state(), 4)
        self.assertIsInstance(fsa.arc_indexes, list)
        self.assertEqual(fsa.arc_indexes, [0, 2, 3, 4, 5, 5])

        arcs = fsa.arcs
        self.assertIsInstance(arcs, k2.ArcVec)
        self.assertEqual(len(arcs), 5)
        if False:
            # the following is for demonstration purpose only.
            # we can iterate through the arcs vector.
            for i, arc in enumerate(arcs):
                print('arc {} -> '.format(i), arc)

    def test_fsa_vec(self):
        fsa_vec = k2.FsaVec()
        self.assertEqual(len(fsa_vec), 0)

        s1 = r'''
        0 1 1
        1 2 -1
        2
        '''
        fsa1 = k2.string_to_fsa(s1)

        fsa_vec.push_back(fsa1)
        self.assertEqual(len(fsa_vec), 1)

        s2 = r'''
        0 1 1
        0 2 2
        1 3 -1
        2 3 -1
        3
        '''
        fsa2 = k2.string_to_fsa(s2)

        fsa_vec.push_back(fsa2)
        self.assertEqual(len(fsa_vec), 2)

        for i, fsa in enumerate(fsa_vec):
            if i == 0:
                self.assertEqual(fsa.num_states(), fsa1.num_states())
                self.assertEqual(fsa.final_state(), fsa1.final_state())
            elif i == 1:
                self.assertEqual(fsa.num_states(), fsa2.num_states())
                self.assertEqual(fsa.final_state(), fsa2.final_state())

        fsa_vec.clear()
        self.assertEqual(len(fsa_vec), 0)

    def test_cfsa(self):
        s = r'''
        0 1 1
        0 2 2
        1 3 -1
        2 3 -1
        3
        '''
        fsa = k2.string_to_fsa(s)
        cfsa = k2.Cfsa(fsa)
        self.assertEqual(cfsa.num_states(), fsa.num_states())
        self.assertEqual(cfsa.num_arcs(), len(fsa.arcs))

        if False:
            # the following is for demonstration purpose only
            for i in range(cfsa.num_states()):
                for arc in cfsa.arc(i):
                    print(arc)


if __name__ == '__main__':
    unittest.main()
