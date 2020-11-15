#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R union_test_py

import unittest

import k2
import torch


class TestUnion(unittest.TestCase):

    def test(self):
        s0 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            1 3 -1 0.4
            2 3 -1 0.5
            2 1 5 0.55
            3
        '''
        s1 = '''
            0 1 -1 0.6
            1
        '''
        s2 = '''
            0 1 6 0.7
            1 0 7 0.8
            1 0 8 0.9
            1 2 -1 1.0
            2
        '''

        fsa0 = k2.Fsa.from_str(s0)
        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)

        fsa_vec = k2.create_fsa_vec([fsa0, fsa1, fsa2])

        fsa = k2.union(fsa_vec)
        dot = k2.to_dot(fsa)
        dot.render('/tmp/fsa', format='pdf')
        # the fsa is saved to /tmp/fsa.pdf
        print(fsa)


if __name__ == '__main__':
    unittest.main()
