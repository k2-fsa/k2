#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_properties_test_py
#

import unittest

import torch

import k2host


class TestIsValid(unittest.TestCase):

    def test_bad_case1(self):
        # fsa should contain at least two states
        array_size = k2host.IntArray2Size(1, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertFalse(k2host.is_valid(fsa))

    def test_bad_case2(self):
        # only kFinalSymbol arcs enter the final state
        s = r'''
        0 1 0 0
        0 2 1 0
        1 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_valid(fsa))

    def test_bad_case3(self):
        # `arc_indexes` and `arcs` in this state are not consistent
        arc_indexes = torch.IntTensor([0, 2, 2, 2])
        arcs = torch.IntTensor([[0, 1, 0, 0], [0, 2, 1, 0], [1, 2, 0, 0]])
        fsa = k2host.Fsa(arc_indexes, arcs)
        self.assertFalse(k2host.is_valid(fsa))

    def test_good_cases1(self):
        # empty fsa is valid
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_valid(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        2 3 -1 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_valid(fsa))

    def test_good_case3(self):
        s = r'''
        0 1 0 0
        0 2 -1 0
        1 2 -1 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_valid(fsa))


class TestIsTopSorted(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        2 1 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_top_sorted(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_top_sorted(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        1 2 0 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_top_sorted(fsa))


class TestIsArcSorted(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 1 0
        0 2 2 0
        1 2 2 0
        1 3 1 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_arc_sorted(fsa))

    def test_bad_cases2(self):
        # same label on two arcs
        s = r'''
        0 2 0 0
        0 1 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_arc_sorted(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_arc_sorted(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        1 2 1 0
        1 3 2 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_arc_sorted(fsa))


class TestHasSelfLoops(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        1 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.has_self_loops(fsa))

    def test_bad_cases2(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertFalse(k2host.has_self_loops(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 0 0
        1 2 0 0
        1 1 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.has_self_loops(fsa))


class TestIsDeterministic(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 2 0
        1 2 0 0
        1 3 0 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_deterministic(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_deterministic(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 2 0
        1 2 0 0
        1 3 2 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_deterministic(fsa))


class TestIsEpsilonFree(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 2 0
        0 2 0 0
        1 2 1 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_epsilon_free(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_epsilon_free(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 2 0
        0 2 1 0
        1 2 1 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_epsilon_free(fsa))


class TestIsConnected(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_connected(fsa))

    def test_bad_cases2(self):
        s = r'''
        0 1 0 0
        0 2 0 0
        2
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_connected(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_connected(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 0 0
        0 3 0 0
        1 2 0 0
        2 3 0 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_connected(fsa))

    def test_good_case3(self):
        s = r'''
        0 3 0 0
        1 2 0 0
        2 3 0 0
        2 3 0 0
        2 4 0 0
        3 1 0 0
        4
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_connected(fsa))


class TestIsAcyclic(unittest.TestCase):

    def test_bad_cases1(self):
        s = r'''
        0 1 2 0
        0 4 0 0
        0 2 0 0
        1 2 1 0
        1 3 0 0
        2 1 0 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_acyclic(fsa))

    def test_good_cases1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_acyclic(fsa))

    def test_good_case2(self):
        s = r'''
        0 1 2 0
        0 2 1 0
        1 2 0 0
        1 3 5 0
        2 3 6 0
        3
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertTrue(k2host.is_acyclic(fsa))


class TestIsEmpty(unittest.TestCase):

    def test_good_cases1(self):
        array_size = k2host.IntArray2Size(0, 0)
        fsa = k2host.Fsa.create_fsa_with_size(array_size)
        self.assertTrue(k2host.is_empty(fsa))

    def test_bad_case1(self):
        s = r'''
        0 1 2 0
        1
        '''
        fsa = k2host.str_to_fsa(s)
        self.assertFalse(k2host.is_empty(fsa))


if __name__ == '__main__':
    unittest.main()
