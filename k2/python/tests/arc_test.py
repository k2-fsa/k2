#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R arc_test_py

import unittest

import k2


class TestArc(unittest.TestCase):

    def test_constructor(self):
        arc = k2.Arc(1, 2, 3, 1.5)
        assert arc.src_state == 1
        assert arc.dest_state == 2
        assert arc.symbol == 3
        assert arc.score == 1.5


if __name__ == '__main__':
    unittest.main()
