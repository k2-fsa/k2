#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R ragged_ops_test_py

import unittest

import k2


class TestRaggedOps(unittest.TestCase):

    def test_remove_axis(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 0 ] [ 3 0 ] [ 2 ] ]')

        ans = k2.ragged.remove_axis(src, 1)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] ]')

    def test_to_list(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [ 3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(k2.ragged.to_list(ans), [[1, 2], [0], [3, 0], [2]])

    def test_remove_values_leq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_leq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 6)
        self.assertEqual(str(ans), '[ [ ] [ ] [ 8 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 8)
        self.assertEqual(str(ans), '[ [ ] [ ] [ ] [ ] ]')

    def test_remove_values_eq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_eq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_eq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 0 ] [ 3 0 2 ] [ 0 8 0 6 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 6)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 8 0 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 8)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 0 6 0 ] [ 0 ] ]')


if __name__ == '__main__':
    unittest.main()
