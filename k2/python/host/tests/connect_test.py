#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_connect_test_py
#

import unittest

import torch

import k2host


class TestConnection(unittest.TestCase):

    def test_case_1(self):
        # a non-connected, non-topsorted, acyclic input fsa;
        # the output fsa is topsorted.
        s = r'''
        0 1 1 0
        0 2 2 0
        1 3 3 0
        1 6 -1 0
        2 4 2 0
        2 6 -1 0
        2 1 1 0
        5 0 1 0
        6
        '''
        fsa = k2host.str_to_fsa(s)
        connection = k2host.Connection(fsa)
        array_size = k2host.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = connection.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        expected_arc_indexes = torch.IntTensor([0, 2, 4, 5, 5])
        expected_arcs = torch.IntTensor([[0, 2, 1, 0], [0, 1, 2, 0],
                                         [1, 3, -1, 0], [1, 2, 1, 0],
                                         [2, 3, -1, 0]])
        expected_arc_map = torch.IntTensor([0, 1, 5, 6, 3])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))

    def test_case_2(self):
        # a cyclic input fsa
        # after trimming, the cycle is removed;
        # so the output fsa should be topsorted.
        s = r'''
        0 1 1 0
        0 2 2 1
        1 3 3 -2
        1 6 6 -3
        2 4 2 4
        2 6 3 5
        2 6 -1 6
        5 0 1 7
        5 7 -1 8
        7
        '''
        fsa = k2host.str_to_fsa(s)
        connection = k2host.Connection(fsa)
        array_size = k2host.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = connection.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        self.assertTrue(k2host.is_empty(fsa_out))
        self.assertTrue(arc_map.empty())

    def test_case_3(self):
        # a non-connected, non-topsorted, acyclic input fsa;
        # the output fsa is topsorted.
        s = r'''
        0 3 3 1
        0 5 5 2
        1 2 2 3
        2 1 1 4
        3 5 5 5
        3 2 2 -6
        3 4 4 7
        3 6 -1 8
        4 5 5 9
        4 6 -1 10
        5 6 -1 11
        6
        '''
        fsa = k2host.str_to_fsa(s)
        connection = k2host.Connection(fsa)
        array_size = k2host.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        connection.get_output(fsa_out)
        self.assertTrue(k2host.is_top_sorted(fsa_out))

    def test_case_4(self):
        # a cyclic input fsa
        # after trimming, the cycle remains (it is not a self-loop);
        # so the output fsa is NOT topsorted.
        s = r'''
        0 3 3 1
        0 2 2 2
        1 0 1 3
        2 6 -1 4
        3 5 5 5
        3 2 2 6
        3 5 5 7
        4 4 4 8
        5 3 3 9
        5 4 4 10
        6
        '''
        fsa = k2host.str_to_fsa(s)
        connection = k2host.Connection(fsa)
        array_size = k2host.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(array_size)
        status = connection.get_output(fsa_out)
        self.assertFalse(status)
        self.assertFalse(k2host.is_top_sorted(fsa_out))


if __name__ == '__main__':
    unittest.main()
