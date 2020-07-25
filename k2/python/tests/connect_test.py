#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R connect_test_py
#

import unittest

import torch

import k2


class TestConnection(unittest.TestCase):

    def test_case_1(self):
        # a non-connected, non-topsorted, acyclic input fsa;
        # the output fsa is topsorted.
        s = r'''
        0 1 1
        0 2 2
        1 3 3
        1 6 -1
        2 4 2
        2 6 -1
        2 1 1
        5 0 1
        6
        '''
        fsa = k2.str_to_fsa(s)
        connection = k2.Connection(fsa)
        array_size = k2.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2.Fsa.create_fsa_with_size(array_size)
        arc_map = k2.IntArray1.create_array_with_size(array_size.size2)
        status = connection.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        expected_arc_indexes = torch.IntTensor([0, 2, 4, 5, 5])
        expected_arcs = torch.IntTensor([[0, 2, 1], [0, 1, 2], [1, 3, -1],
                                         [1, 2, 1], [2, 3, -1]])
        expected_arc_map = torch.IntTensor([0, 1, 5, 6, 3])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))

    def test_case_2(self):
        # a cyclic input fsa
        # after trimming, the cycle is removed;
        # so the output fsa should be topsorted.
        s = r'''
        0 1 1
        0 2 2
        1 3 3
        1 6 6
        2 4 2
        2 6 3
        2 6 -1
        5 0 1
        5 7 -1
        7
        '''
        fsa = k2.str_to_fsa(s)
        connection = k2.Connection(fsa)
        array_size = k2.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2.Fsa.create_fsa_with_size(array_size)
        arc_map = k2.IntArray1.create_array_with_size(array_size.size2)
        status = connection.get_output(fsa_out, arc_map)
        self.assertTrue(status)
        self.assertTrue(k2.is_empty(fsa_out))
        self.assertTrue(arc_map.empty())

    def test_case_3(self):
        # a non-connected, non-topsorted, acyclic input fsa;
        # the output fsa is topsorted.
        s = r'''
        0 3 3
        0 5 5
        1 2 2
        2 1 1
        3 5 5
        3 2 2
        3 4 4
        3 6 -1
        4 5 5
        4 6 -1
        5 6 -1
        6
        '''
        fsa = k2.str_to_fsa(s)
        connection = k2.Connection(fsa)
        array_size = k2.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2.Fsa.create_fsa_with_size(array_size)
        connection.get_output(fsa_out)
        self.assertTrue(k2.is_top_sorted(fsa_out))

    def test_case_4(self):
        # a cyclic input fsa
        # after trimming, the cycle remains (it is not a self-loop);
        # so the output fsa is NOT topsorted.
        s = r'''
        0 3 3
        0 2 2
        1 0 1
        2 6 -1
        3 5 5
        3 2 2
        3 5 5
        4 4 4
        5 3 3
        5 4 4
        6
        '''
        fsa = k2.str_to_fsa(s)
        connection = k2.Connection(fsa)
        array_size = k2.IntArray2Size()
        connection.get_sizes(array_size)
        fsa_out = k2.Fsa.create_fsa_with_size(array_size)
        status = connection.get_output(fsa_out)
        self.assertFalse(status)
        self.assertFalse(k2.is_top_sorted(fsa_out))


if __name__ == '__main__':
    unittest.main()
