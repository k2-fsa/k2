#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R array_test_py
#

from struct import pack, unpack
import unittest

import torch

import k2


class TestArray(unittest.TestCase):

    def test_int_array1(self):
        data = torch.arange(10).to(torch.int32)

        array = k2.IntArray1(data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2.IntArray1)
        self.assertEqual(data.numel(), array.size)
        self.assertEqual(array.data[9], 9)

        # the underlying memory is shared between k2 and torch;
        # so change one will change another
        data[0] = 100
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)

        del data
        # the array in k2 is still accessible
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)

    def test_int_array2(self):
        data = torch.arange(10).to(torch.int32)
        indexes = torch.tensor([0, 2, 5, 6, 10]).to(torch.int32)
        self.assertEqual(data.numel(), indexes[-1].item())

        array = k2.IntArray2(indexes, data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2.IntArray2)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data.numel(), array.size2)
        self.assertEqual(array.data[9], 9)

        # the underlying memory is shared between k2 and torch;
        # so change one will change another
        data[0] = 100
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)
        indexes[1] = 3
        self.assertEqual(array.indexes[1], 3)
        self.assertEqual(array.get_indexes(1), 3)

        del data
        del indexes
        # the array in k2 is still accessible
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)
        self.assertEqual(array.indexes[1], 3)
        self.assertEqual(array.get_indexes(1), 3)

    def test_logsum_arc_derivs(self):
        indexes = torch.tensor([0, 2, 3, 5]).to(torch.int32)
        data1 = torch.tensor([0, 1, 2, 5, 3]).to(torch.int32)
        data2 = torch.tensor([0, 1.2, 2, 5, 3.8]).to(torch.float)
        self.assertEqual(data1.shape[0], indexes[-1].item())

        array = k2.LogSumArcDerivs(indexes, data1, data2)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2.LogSumArcDerivs)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data1.shape[0], array.size2)
        self.assertEqual(data2.shape[0], array.size2)


if __name__ == '__main__':
    unittest.main()
