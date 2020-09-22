#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_array_test_py
#

from struct import pack, unpack
import unittest

import torch

import k2host


class TestArray(unittest.TestCase):

    def test_int_array1(self):
        data = torch.arange(10).to(torch.int32)

        array = k2host.IntArray1(data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2host.IntArray1)
        self.assertEqual(data.numel(), array.size)
        self.assertEqual(array.data[9], 9)

        # the underlying memory is shared between k2host and torch;
        # so change one will change another
        data[0] = 100
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)

        del data
        # the array in k2host is still accessible
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)

    def test_int_array2(self):
        data = torch.arange(10).to(torch.int32)
        indexes = torch.tensor([0, 2, 5, 6, 10]).to(torch.int32)
        self.assertEqual(data.numel(), indexes[-1].item())

        array = k2host.IntArray2(indexes, data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2host.IntArray2)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data.numel(), array.size2)
        self.assertEqual(array.data[9], 9)

        # the underlying memory is shared between k2host and torch;
        # so change one will change another
        data[0] = 100
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)
        indexes[1] = 3
        self.assertEqual(array.indexes[1], 3)
        self.assertEqual(array.get_indexes(1), 3)

        del data
        del indexes
        # the array in k2host is still accessible
        self.assertEqual(array.data[0], 100)
        self.assertEqual(array.get_data(0), 100)
        self.assertEqual(array.indexes[1], 3)
        self.assertEqual(array.get_indexes(1), 3)

    def test_logsum_arc_derivs(self):
        data = torch.arange(10).reshape(5, 2).to(torch.float)
        indexes = torch.tensor([0, 2, 3, 5]).to(torch.int32)
        self.assertEqual(data.shape[0], indexes[-1].item())

        array = k2host.LogSumArcDerivs(indexes, data)
        self.assertFalse(array.empty())
        self.assertIsInstance(array, k2host.LogSumArcDerivs)

        self.assertEqual(indexes.numel(), array.size1 + 1)
        self.assertEqual(data.shape[0], array.size2)
        self.assertTrue(torch.equal(array.data[1], torch.FloatTensor([2, 3])))

        # convert arc-ids in arc-derivs to IntArray
        arc_ids = k2host.StridedIntArray1.from_float_tensor(array.data[:, 0])
        # the underlying memory is shared between k2host and torch;
        # so change one will change another
        data[1] = torch.FloatTensor([100, 200])
        self.assertTrue(
            torch.equal(array.data[1], torch.FloatTensor([100, 200])))
        self.assertEqual(array.get_data(1)[1], 200)
        self.assertEqual(arc_ids.data[1], 100)
        # we need pack and then unpack here to interpret arc_id (int) as a float,
        # this is only for test purpose as users would usually never call
        # `array.get_data` to retrieve data. Instead, it is supposed to call
        # `array.data` to retrieve or update data in the array object.
        arc_id = pack('i', array.get_data(1)[0])
        self.assertEqual(unpack('f', arc_id)[0], 100)

        del data
        # the array in k2host is still accessible
        self.assertEqual(array.get_data(1)[1], 200)


if __name__ == '__main__':
    unittest.main()
