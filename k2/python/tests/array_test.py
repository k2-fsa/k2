#!/usr/bin/env python3
#
# Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R array_test_py

import unittest

import k2
import torch


class TestArray(unittest.TestCase):

    def test_int_array1(self):
        a = torch.arange(5).to(torch.int32)
        arr1 = k2.IntArray1(a)
        self.assertEqual(len(arr1), a.numel())

        for i, v in enumerate(arr1):
            self.assertEqual(i, v)

        # the underlying memory is shared between k2 and torch;
        # so change one will change another
        arr1[0] = 10
        self.assertEqual(a[0], 10)

        a[1] = 100
        self.assertEqual(arr1[1], 100)

        del a
        # the array in k2 is still accessible
        self.assertEqual(arr1[1], 100)

    def test_float_array1(self):
        a = torch.arange(5).to(torch.float)
        arr1 = k2.FloatArray1(a)
        self.assertEqual(len(arr1), a.numel())

        for i, v in enumerate(arr1):
            self.assertEqual(i, v)

        # the underlying memory is shared between k2 and torch;
        # so change one will change another
        arr1[0] = 1.25
        self.assertEqual(a[0], 1.25)

        a[1] = 0.5
        self.assertEqual(arr1[1], 0.5)


if __name__ == '__main__':
    unittest.main()
