#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_aux_labels_test_py
#

import unittest

import torch

import k2host


class TestAuxLabelsMapper(unittest.TestCase):

    def setUp(self):
        indexes = torch.IntTensor([0, 1, 3, 6, 7])
        data = torch.IntTensor([1, 2, 3, 4, 5, 6, 7])
        self.aux_labels_in = k2host.AuxLabels(indexes, data)

    def test_mapper1_case_1(self):
        # empty arc map
        arc_map = k2host.IntArray1.create_array_with_size(0)
        mapper = k2host.AuxLabels1Mapper(self.aux_labels_in, arc_map)
        aux_size = k2host.IntArray2Size()
        mapper.get_sizes(aux_size)
        self.assertEqual(aux_size.size1, 0)
        self.assertEqual(aux_size.size2, 0)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        mapper.get_output(labels_out)
        self.assertTrue(labels_out.empty())

    def test_mapper1_case_2(self):
        arc_map = k2host.IntArray1(torch.IntTensor([2, 0, 3]))
        mapper = k2host.AuxLabels1Mapper(self.aux_labels_in, arc_map)
        aux_size = k2host.IntArray2Size()
        mapper.get_sizes(aux_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        mapper.get_output(labels_out)
        self.assertEqual(aux_size.size1, 3)
        self.assertEqual(aux_size.size2, 5)
        expected_indexes = torch.IntTensor([0, 3, 4, 5])
        expected_data = torch.IntTensor([4, 5, 6, 1, 7])
        self.assertTrue(torch.equal(labels_out.indexes, expected_indexes))
        self.assertTrue(torch.equal(labels_out.data, expected_data))

    def test_mapper1_case_3(self):
        # all arcs in the input fsa remain.
        arc_map = k2host.IntArray1(torch.IntTensor([2, 0, 3, 1]))
        mapper = k2host.AuxLabels1Mapper(self.aux_labels_in, arc_map)
        aux_size = k2host.IntArray2Size()
        mapper.get_sizes(aux_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        mapper.get_output(labels_out)
        self.assertEqual(aux_size.size1, 4)
        self.assertEqual(aux_size.size2, 7)
        expected_indexes = torch.IntTensor([0, 3, 4, 5, 7])
        expected_data = torch.IntTensor([4, 5, 6, 1, 7, 2, 3])
        self.assertTrue(torch.equal(labels_out.indexes, expected_indexes))
        self.assertTrue(torch.equal(labels_out.data, expected_data))

    def test_mapper2_case_1(self):
        # empty arc map
        array_size = k2host.IntArray2Size(0, 0)
        arc_map = k2host.IntArray2.create_array_with_size(array_size)
        mapper = k2host.AuxLabels2Mapper(self.aux_labels_in, arc_map)
        aux_size = k2host.IntArray2Size()
        mapper.get_sizes(aux_size)
        self.assertEqual(aux_size.size1, 0)
        self.assertEqual(aux_size.size2, 0)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        mapper.get_output(labels_out)
        self.assertTrue(labels_out.empty())

    def test_mapper2_case_2(self):
        indexes = torch.IntTensor([0, 2, 4, 5, 6])
        data = torch.IntTensor([2, 3, 0, 1, 0, 2])
        arc_map = k2host.IntArray2(indexes, data)
        mapper = k2host.AuxLabels2Mapper(self.aux_labels_in, arc_map)
        aux_size = k2host.IntArray2Size()
        mapper.get_sizes(aux_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        mapper.get_output(labels_out)
        self.assertEqual(aux_size.size1, 4)
        self.assertEqual(aux_size.size2, 11)
        expected_indexes = torch.IntTensor([0, 4, 7, 8, 11])
        expected_data = torch.IntTensor([4, 5, 6, 7, 1, 2, 3, 1, 4, 5, 6])
        self.assertTrue(torch.equal(labels_out.indexes, expected_indexes))
        self.assertTrue(torch.equal(labels_out.data, expected_data))


class TestFstInverter(unittest.TestCase):

    def test_case_1(self):
        # empty fsa
        array_size = k2host.IntArray2Size(0, 0)
        fsa_in = k2host.Fsa.create_fsa_with_size(array_size)
        indexes = torch.IntTensor([0, 1, 3, 6, 7])
        data = torch.IntTensor([1, 2, 3, 4, 5, 6, 7])
        labels_in = k2host.AuxLabels(indexes, data)
        inverter = k2host.FstInverter(fsa_in, labels_in)
        fsa_size = k2host.IntArray2Size()
        aux_size = k2host.IntArray2Size()
        inverter.get_sizes(fsa_size, aux_size)
        self.assertEqual(aux_size.size1, 0)
        self.assertEqual(aux_size.size2, 0)
        fsa_out = k2host.Fsa.create_fsa_with_size(fsa_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        inverter.get_output(fsa_out, labels_out)
        self.assertTrue(k2host.is_empty(fsa_out))
        self.assertTrue(labels_out.empty())

    def test_case_2(self):
        # top-sorted input FSA
        s = r'''
        0 1 1 0
        0 1 0 0
        0 3 2 0
        1 2 3 0
        1 3 4 0
        1 5 -1 0
        2 3 0 0
        2 5 -1 0
        4 5 -1 0
        5
        '''

        fsa_in = k2host.str_to_fsa(s)
        indexes = torch.IntTensor([0, 2, 3, 3, 6, 6, 7, 7, 8, 9])
        data = torch.IntTensor([1, 2, 3, 5, 6, 7, -1, -1, -1])
        labels_in = k2host.AuxLabels(indexes, data)
        inverter = k2host.FstInverter(fsa_in, labels_in)
        fsa_size = k2host.IntArray2Size()
        aux_size = k2host.IntArray2Size()
        inverter.get_sizes(fsa_size, aux_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(fsa_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        inverter.get_output(fsa_out, labels_out)
        expected_arc_indexes = torch.IntTensor(
            [0, 3, 4, 7, 8, 9, 11, 11, 12, 12])
        expected_arcs = torch.IntTensor([[0, 1, 1, 0], [0, 2, 3, 0],
                                         [0, 6, 0, 0], [1, 2, 2, 0],
                                         [2, 3, 5, 0], [2, 6, 0, 0],
                                         [2, 8, -1, 0], [3, 4, 6, 0],
                                         [4, 5, 7, 0], [5, 6, 0, 0],
                                         [5, 8, -1, 0], [7, 8, -1, 0]])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        expected_label_indexes = torch.IntTensor(
            [0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7])
        expected_labels = torch.IntTensor([2, 1, 4, -1, 3, -1, -1])
        self.assertTrue(torch.equal(labels_out.indexes,
                                    expected_label_indexes))
        self.assertTrue(torch.equal(labels_out.data, expected_labels))

    def test_case_3(self):
        # non-top-sorted input FSA
        s = r'''
        0 1 1 0
        0 1 0 0
        0 3 2 0
        1 2 3 0
        1 3 4 0
        2 1 5 0
        2 5 -1 0
        3 1 6 0
        4 5 -1 0
        5
        '''

        fsa_in = k2host.str_to_fsa(s)
        indexes = torch.IntTensor([0, 2, 3, 3, 6, 6, 7, 8, 10, 11])
        data = torch.IntTensor([1, 2, 3, 5, 6, 7, 8, -1, 9, 10, -1])
        labels_in = k2host.AuxLabels(indexes, data)
        inverter = k2host.FstInverter(fsa_in, labels_in)
        fsa_size = k2host.IntArray2Size()
        aux_size = k2host.IntArray2Size()
        inverter.get_sizes(fsa_size, aux_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(fsa_size)
        labels_out = k2host.AuxLabels.create_array_with_size(aux_size)
        inverter.get_output(fsa_out, labels_out)
        expected_arc_indexes = torch.IntTensor(
            [0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 13])
        expected_arcs = torch.IntTensor([[0, 1, 1, 0], [0, 3, 3, 0],
                                         [0, 7, 0, 0], [1, 3, 2, 0],
                                         [2, 3, 10, 0], [3, 4, 5, 0],
                                         [3, 7, 0, 0], [4, 5, 6, 0],
                                         [5, 6, 7, 0], [6, 3, 8, 0],
                                         [6, 9, -1, 0], [7, 2, 9, 0],
                                         [8, 9, -1, 0]])
        self.assertTrue(torch.equal(fsa_out.indexes, expected_arc_indexes))
        self.assertTrue(torch.equal(fsa_out.data, expected_arcs))
        expected_label_indexes = torch.IntTensor(
            [0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8])
        expected_labels = torch.IntTensor([2, 1, 6, 4, 3, 5, -1, -1])
        self.assertTrue(torch.equal(labels_out.indexes,
                                    expected_label_indexes))
        self.assertTrue(torch.equal(labels_out.data, expected_labels))


if __name__ == '__main__':
    unittest.main()
