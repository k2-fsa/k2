#!/usr/bin/env python3
#
# Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_utl_test_py
#

import unittest

import k2


class TestFsaUtil(unittest.TestCase):

    def test_str_to_fsa(self):
        s = '''
        0 1 1
        0 2 2
        0 3 3
        1 3 3
        3 10 -1
        10
        '''
        fsa = k2.str_to_fsa(s)
        self.assertEqual(fsa.num_states(), 11)
        self.assertEqual(fsa.final_state(), 10)
        self.assertEqual(len(fsa.arcs), fsa.final_state() + 1)

        for i in range(len(fsa.arcs)):
            for a in fsa.arcs[i]:
                print(a)


if __name__ == '__main__':
    unittest.main()
