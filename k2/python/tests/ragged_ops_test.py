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

    def test_remove_values_leq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_leq(src, 0)
        assert str(ans) == '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]'

        ans = k2.ragged.remove_values_leq(src, 1)
        assert str(ans) == '[ [ 2 ] [ 3 2 ] [ 8 6 ] [ ] ]'

        ans = k2.ragged.remove_values_leq(src, 6)
        assert str(ans) == '[ [ ] [ ] [ 8 ] [ ] ]'

        ans = k2.ragged.remove_values_leq(src, 8)
        assert str(ans) == '[ [ ] [ ] [ ] [ ] ]'

    def test_remove_values_equal(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_equal(src, 0)
        assert str(ans) == '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]'

        ans = k2.ragged.remove_values_equal(src, 1)
        assert str(ans) == '[ [ 2 0 ] [ 3 0 2 ] [ 0 8 0 6 0 ] [ 0 ] ]'

        ans = k2.ragged.remove_values_equal(src, 6)
        assert str(ans) == '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 8 0 0 ] [ 0 ] ]'

        ans = k2.ragged.remove_values_equal(src, 8)
        assert str(ans) == '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 0 6 0 ] [ 0 ] ]'


if __name__ == '__main__':
    unittest.main()
