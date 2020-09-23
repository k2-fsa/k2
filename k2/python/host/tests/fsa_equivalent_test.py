#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_fsa_equivalent_test_py
#

import unittest

import torch

import k2host


class TestIsFsaEquivalent(unittest.TestCase):

    def test_bad_case_1(self):
        # just set arc.weight as 0 since we won't use it here
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        1 3 4 0
        2 3 5 0
        3
        '''
        fsa_a = k2host.str_to_fsa(s_a)
        s_b = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        3
        '''
        fsa_b = k2host.str_to_fsa(s_b)
        self.assertFalse(k2host.is_rand_equivalent(fsa_a, fsa_b))

    def test_bad_case_2(self):
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        1 3 4 0
        2 3 5 0
        3
        '''
        fsa_a = k2host.str_to_fsa(s_a)
        s_b = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        1 3 4 0
        2 3 6 0
        3
        '''
        fsa_b = k2host.str_to_fsa(s_b)
        self.assertFalse(k2host.is_rand_equivalent(fsa_a, fsa_b, 100))

    def test_good_case_1(self):
        # both fsas will be empty after triming
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        3
        '''
        fsa_a = k2host.str_to_fsa(s_a)
        s_b = r'''
        0 1 1 0
        0 2 2 0
        3
        '''
        fsa_b = k2host.str_to_fsa(s_b)
        self.assertTrue(k2host.is_rand_equivalent(fsa_a, fsa_b))

    def test_good_case_2(self):
        # same fsas
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        1 3 4 0
        2 3 5 0
        3
        '''
        fsa_a = k2host.str_to_fsa(s_a)
        self.assertTrue(k2host.is_rand_equivalent(fsa_a, fsa_a))

    def test_bad_case_2(self):
        s_a = r'''
        0 1 1 0
        0 2 2 0
        0 3 8 0
        1 4 4 0
        2 4 5 0
        4
        '''
        fsa_a = k2host.str_to_fsa(s_a)
        s_b = r'''
        0 2 1 0
        0 1 2 0
        0 3 9 0
        1 4 5 0
        2 4 4 0
        4
        '''
        fsa_b = k2host.str_to_fsa(s_b)
        self.assertTrue(k2host.is_rand_equivalent(fsa_a, fsa_b))


class TestIsWfsaRandEquivalent(unittest.TestCase):

    def setUp(self):
        s_a = r'''
        0 1 1 2
        0 1 2 2
        0 1 3 3
        0 2 4 3
        0 2 5 1
        1 3 5 3
        1 3 6 2
        2 4 5 5
        2 4 6 4
        3 5 -1 1
        4 5 -1 3
        5
        '''
        self.fsa_a = k2host.str_to_fsa(s_a)
        s_b = r'''
        0 1 1 5
        0 1 2 5
        0 1 3 6
        0 1 4 10
        0 1 5 8
        1 2 5 1
        1 2 6 0
        2 3 -1 0
        3
        '''
        self.fsa_b = k2host.str_to_fsa(s_b)
        s_c = r'''
        0 1 1 5
        0 1 2 5
        0 1 3 6
        0 1 4 10
        0 1 5 9
        1 2 5 1
        1 2 6 0
        2 3 -1 0
        3
        '''
        self.fsa_c = k2host.str_to_fsa(s_c)

    def test_max_weight(self):
        self.assertTrue(
            k2host.is_rand_equivalent_max_weight(self.fsa_a, self.fsa_b))
        self.assertFalse(
            k2host.is_rand_equivalent_max_weight(self.fsa_a, self.fsa_c))

    def test_logsum_weight(self):
        self.assertTrue(
            k2host.is_rand_equivalent_logsum_weight(self.fsa_a, self.fsa_b))
        self.assertFalse(
            k2host.is_rand_equivalent_logsum_weight(self.fsa_a, self.fsa_c))

    def test_with_beam(self):
        self.assertTrue(
            k2host.is_rand_equivalent_max_weight(self.fsa_a, self.fsa_b, 4.0))
        self.assertFalse(
            k2host.is_rand_equivalent_max_weight(self.fsa_a, self.fsa_c, 6.0))


class TestRandPath(unittest.TestCase):

    def test_bad_case_1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        rand_path = k2host.RandPath(fsa, False)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = rand_path.get_output(path, arc_map)
        self.assertFalse(status)
        self.assertTrue(k2host.is_empty(path))
        self.assertTrue(arc_map.empty())

    def test_bad_case_2(self):
        # non-connected fsa
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 3 4 0
        3
        '''
        fsa = k2host.str_to_fsa(s_a)
        rand_path = k2host.RandPath(fsa, False)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = rand_path.get_output(path, arc_map)
        self.assertFalse(status)
        self.assertTrue(k2host.is_empty(path))
        self.assertTrue(arc_map.empty())

    def test_good_case_1(self):
        s_a = r'''
        0 1 1 0
        0 2 2 0
        1 2 3 0
        2 3 4 0
        2 4 5 0
        3 4 7 0
        4 5 9 0
        5
        '''
        fsa = k2host.str_to_fsa(s_a)
        rand_path = k2host.RandPath(fsa, False)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        status = rand_path.get_output(path)
        self.assertTrue(status)
        self.assertFalse(k2host.is_empty(path))

    def test_good_case_2(self):
        s_a = r'''
        0 1 1 0
        1 2 3 0
        2 3 4 0
        3
        '''
        fsa = k2host.str_to_fsa(s_a)
        rand_path = k2host.RandPath(fsa, False)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = rand_path.get_output(path, arc_map)
        self.assertTrue(status)
        self.assertFalse(k2host.is_empty(path))
        self.assertFalse(arc_map.empty())
        expected_arc_indexes = torch.IntTensor([0, 1, 2, 3, 3])
        expected_arcs = torch.IntTensor([[0, 1, 1, 0], [1, 2, 3, 0],
                                         [2, 3, 4, 0]])
        expected_arc_map = torch.IntTensor([0, 1, 2])
        self.assertTrue(torch.equal(path.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(path.data, expected_arcs))
        self.assertTrue(torch.equal(arc_map.data, expected_arc_map))

    def test_eps_arc_1(self):
        s_a = r'''
        0 1 1 0
        0 2 0 0
        1 2 3 0
        2 3 0 0
        2 4 5 0
        3 4 7 0
        4 5 9 0
        5
        '''
        fsa = k2host.str_to_fsa(s_a)
        rand_path = k2host.RandPath(fsa, True)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = rand_path.get_output(path, arc_map)
        self.assertTrue(status)
        self.assertFalse(k2host.is_empty(path))
        self.assertFalse(arc_map.empty())

    def test_eps_arc_2(self):
        # there is no epsilon-free path
        s_a = r'''
        0 1 1 0
        0 2 0 0
        1 2 3 0
        2 3 0 0
        3 5 7 0
        3 4 8 0
        4 5 9 0
        5
        '''
        fsa = k2host.str_to_fsa(s_a)
        rand_path = k2host.RandPath(fsa, True)
        array_size = k2host.IntArray2Size()
        rand_path.get_sizes(array_size)
        path = k2host.Fsa.create_fsa_with_size(array_size)
        arc_map = k2host.IntArray1.create_array_with_size(array_size.size2)
        status = rand_path.get_output(path, arc_map)
        self.assertFalse(status)
        self.assertTrue(k2host.is_empty(path))
        self.assertTrue(arc_map.empty())


if __name__ == '__main__':
    unittest.main()
