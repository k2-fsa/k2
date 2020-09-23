#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_arcsort_test_py
#

import unittest

import torch

import k2host
from k2host.fsa_util import float_to_int


class TestArcSort(unittest.TestCase):

    def test_empty_fsa(self):
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(fsa.size2)
        k2host.arc_sort(fsa, arc_map)
        self.assertTrue(k2host.is_empty(fsa))
        self.assertTrue(arc_map.empty())

        # test without arc_map
        k2host.arc_sort(fsa)
        self.assertTrue(k2host.is_empty(fsa))

    def test_arc_sort(self):
        s = r'''
        0 1 2 1
        0 4 0 2
        0 2 0 3
        1 2 1 4
        1 3 0 5
        2 1 0 6
        4
        '''

        fsa = k2host.str_to_fsa(s)
        arc_map = k2host.IntArray1.create_array_with_size(fsa.size2)
        k2host.arc_sort(fsa, arc_map)
        expected_arc_indexes = torch.IntTensor([0, 3, 5, 6, 6, 6])
        expected_arcs = torch.IntTensor([[0, 2, 0, float_to_int(3)],
                                         [0, 4, 0, float_to_int(2)],
                                         [0, 1, 2, float_to_int(1)],
                                         [1, 3, 0, float_to_int(5)],
                                         [1, 2, 1, float_to_int(4)],
                                         [2, 1, 0, float_to_int(6)]])
        expected_arc_map = torch.IntTensor([2, 1, 0, 4, 3, 5])
        self.assertTrue(torch.equal(fsa.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))


class TestArcSorter(unittest.TestCase):

    def test_empty_fsa(self):
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        sorter = k2host.ArcSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        sorter.get_output(fsa_out, arc_map)
        self.assertTrue(k2host.is_empty(fsa))

        # test without arc_map
        sorter.get_output(fsa_out)
        self.assertTrue(k2host.is_empty(fsa_out))

    def test_arc_sort(self):
        s = r'''
        0 1 2 1
        0 4 0 2
        0 2 0 3
        1 2 1 4
        1 3 0 5
        2 1 0 6
        4
        '''

        fsa = k2host.str_to_fsa(s)
        sorter = k2host.ArcSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        sorter.get_output(fsa_out, arc_map)
        expected_arc_indexes = torch.IntTensor([0, 3, 5, 6, 6, 6])
        expected_arcs = torch.IntTensor([[0, 2, 0, float_to_int(3)],
                                         [0, 4, 0, float_to_int(2)],
                                         [0, 1, 2, float_to_int(1)],
                                         [1, 3, 0, float_to_int(5)],
                                         [1, 2, 1, float_to_int(4)],
                                         [2, 1, 0, float_to_int(6)]])
        expected_arc_map = torch.IntTensor([2, 1, 0, 4, 3, 5])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))


if __name__ == '__main__':
    unittest.main()
