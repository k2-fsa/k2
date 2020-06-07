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

    def test_int_array1_with_stride_3(self):
        a = torch.arange(15).reshape(5, 3).to(torch.int32)
        b = a[:, 0]
        self.assertEqual(b.size(), (5,))
        self.assertEqual(b.stride(), (3,))

        arr1 = k2.IntArray1(b)
        self.assertEqual(len(arr1), b.numel())

        for i, v in enumerate(arr1):
            self.assertEqual(v, b[i])

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

    def test_arc_array1(self):
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).to(torch.int32)
        arr1 = k2.ArcArray1(tensor)
        self.assertEqual(len(arr1), 3)
        self.assertEqual(arr1[0], k2.Arc(1, 2, 3))
        self.assertEqual(arr1[1], k2.Arc(4, 5, 6))
        self.assertEqual(arr1[2], k2.Arc(7, 8, 9))

        # memory is shared between torch.Tensor and k2.ArcArray1
        arr1[0] = k2.Arc(100, 200, 300)
        self.assertEqual(tensor[0], 100)

        tensor += 1
        self.assertEqual(arr1[0], k2.Arc(101, 201, 301))

    def test_arc_array2(self):
        data = torch.tensor([
            [0, 1, 1],
            [0, 2, 2],
            [1, 3, 3],
            [1, 4, -1],
            [1, 2, 2],
            [2, 3, 3],
            [3, 4, -1],
        ]).to(torch.int32)
        indexes = torch.tensor([0, 2, 5, 6, 7]).to(torch.int32)

        arr2 = k2.ArcArray2(indexes, data)
        self.assertEqual(len(arr2), 4)
        k = 0
        for i in range(len(arr2)):
            for arc in arr2[i]:
                self.assertEqual(arc, k2.Arc(*list(data[k].numpy())))
                k += 1

        # memory is shared between torch.Tensor and k2.ArcArray2
        data[0][0] = 10
        self.assertEqual(arr2[0][0].src_state, 10)

        arr2[0][0] = k2.Arc(100, 200, 300)
        self.assertTrue(all(data[0] == torch.tensor([100, 200, 300])))


if __name__ == '__main__':
    unittest.main()
