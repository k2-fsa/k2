#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R array_test_py
#

import unittest

import torch

import k2


class TestArray(unittest.TestCase):

    def test_int_array2(self):
        data = torch.arange(10).to(torch.int32)
        indexes = torch.tensor([0, 2, 5, 6, 10]).to(torch.int32)
        self.assertEqual(data.numel(),indexes[-1].item())

        array = k2.IntArray2(indexes, data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2.IntArray2)

        # test iterator
        for i, v in enumerate(array):
            self.assertEqual(i, v)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data.numel(), array.size2)

        # the underlying memory is shared between k2 and torch;
        # so change one will change another
        data[0] = 100
        self.assertEqual(array.data(0), 100)

        del data
        # the array in k2 is still accessible
        self.assertEqual(array.data(0), 100)


    def test_logsum_arc_derivs(self):
        data = torch.arange(10).reshape(5,2).to(torch.float)
        indexes = torch.tensor([0, 2, 3, 5]).to(torch.int32)
        self.assertEqual(data.shape[0],indexes[-1].item())

        array = k2.LogSumArcDerivs(indexes, data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2.LogSumArcDerivs)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data.shape[0], array.size2)

        self.assertEqual(array.data(0), (0,1.0))


if __name__ == '__main__':
    unittest.main()
