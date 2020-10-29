#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_topsort_test_py
#

import unittest

import torch

import k2host


class TestTopSorter(unittest.TestCase):

    def test_case_1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        sorter = k2host.TopSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = sorter.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        self.assertTrue(k2host.is_empty(fsa_out))
        self.assertTrue(arc_map.empty())

        # test without arc_map
        sorter.get_output(fsa_out)
        self.assertTrue(k2host.is_empty(fsa_out))

    def test_case_2(self):
        # non-connected fsa (not co-accessible)
        s = r'''
        0 2 -1 0
        1 2 -1 0
        1 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        sorter = k2host.TopSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = sorter.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        self.assertFalse(k2host.is_empty(fsa_out))
        expected_arc_map = torch.IntTensor([0])
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))

    def test_case_3(self):
        # non-connected fsa (not accessible)
        s = r'''
        0 2 -1 0
        1 0 1 0
        1 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        sorter = k2host.TopSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = sorter.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        self.assertFalse(k2host.is_empty(fsa_out))
        expected_arc_map = torch.IntTensor([0])
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))

    def test_case_4(self):
        # connected fsa
        s = r'''
        0 4 40 0
        0 2 20 0
        1 6 -1 0
        2 3 30 0
        3 6 -1 0
        3 1 10 0
        4 5 50 0
        5 2 8 0
        6
        '''
        fsa = k2host.str_to_fsa(s)
        sorter = k2host.TopSorter(fsa)
        array_size = k2host.IntArray2Size()
        sorter.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = sorter.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        expected_arc_indexes = torch.IntTensor([0, 2, 3, 4, 5, 7, 8, 8])
        expected_arcs = torch.IntTensor([[0, 1, 40, 0], [0, 3, 20, 0],
                                         [1, 2, 50, 0], [2, 3, 8, 0],
                                         [3, 4, 30, 0], [4, 6, -1, 0],
                                         [4, 5, 10, 0], [5, 6, -1, 0]])
        expected_arc_map = torch.IntTensor([0, 1, 6, 7, 3, 4, 5, 2])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))


if __name__ == '__main__':
    unittest.main()
