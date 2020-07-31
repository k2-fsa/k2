#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R rmepsilon_test_py
#

from struct import pack, unpack
import unittest

import torch

import k2


class TestRmEpsilon(unittest.TestCase):

    def setUp(self):
        s = r'''
        0 4 1
        0 1 1
        1 2 0
        1 3 0
        1 4 0
        2 7 0
        3 7 0
        4 6 1
        4 6 0
        4 8 1
        4 9 -1
        5 9 -1
        6 9 -1
        7 9 -1
        8 9 -1
        9
        '''
        self.fsa = k2.str_to_fsa(s)
        self.num_states = self.fsa.num_states()
        weights = torch.FloatTensor(
            [1, 1, 2, 3, 2, 4, 5, 2, 3, 3, 2, 4, 3, 5, 6])
        self.weights = k2.FloatArray1(weights)

    def test_max_weight(self):
        forward_max_weights = k2.DoubleArray1.create_array_with_size(
            self.num_states)
        backward_max_weights = k2.DoubleArray1.create_array_with_size(
            self.num_states)
        wfsa = k2.WfsaWithFbWeights(self.fsa, self.weights,
                                    k2.FbWeightType.kMaxWeight,
                                    forward_max_weights, backward_max_weights)
        beam = 8.0
        remover = k2.EpsilonsRemoverMax(wfsa, beam)
        fsa_size = k2.IntArray2Size()
        arc_derivs_size = k2.IntArray2Size()
        remover.get_sizes(fsa_size, arc_derivs_size)
        fsa_out = k2.Fsa.create_fsa_with_size(fsa_size)
        arc_derivs = k2.IntArray2.create_array_with_size(arc_derivs_size)
        arc_weights_out = k2.FloatArray1.create_array_with_size(fsa_size.size2)
        remover.get_output(fsa_out, arc_weights_out, arc_derivs)
        self.assertTrue(k2.is_epsilon_free(fsa_out))
        self.assertEqual(fsa_out.size1, 6)
        self.assertEqual(fsa_out.size2, 11)
        self.assertEqual(arc_derivs.size1, 11)
        self.assertEqual(arc_derivs.size2, 18)
        self.assertTrue(
            k2.is_rand_equivalent_max_weight(self.fsa, self.weights, fsa_out,
                                             arc_weights_out, beam))

    def test_logsum_weight(self):
        forward_logsum_weights = k2.DoubleArray1.create_array_with_size(
            self.num_states)
        backward_logsum_weights = k2.DoubleArray1.create_array_with_size(
            self.num_states)
        wfsa = k2.WfsaWithFbWeights(self.fsa, self.weights,
                                    k2.FbWeightType.kLogSumWeight,
                                    forward_logsum_weights,
                                    backward_logsum_weights)
        beam = 8.0
        remover = k2.EpsilonsRemoverLogSum(wfsa, beam)
        fsa_size = k2.IntArray2Size()
        arc_derivs_size = k2.IntArray2Size()
        remover.get_sizes(fsa_size, arc_derivs_size)
        fsa_out = k2.Fsa.create_fsa_with_size(fsa_size)
        arc_derivs = k2.LogSumArcDerivs.create_arc_derivs_with_size(
            arc_derivs_size)
        arc_weights_out = k2.FloatArray1.create_array_with_size(fsa_size.size2)
        remover.get_output(fsa_out, arc_weights_out, arc_derivs)
        self.assertTrue(k2.is_epsilon_free(fsa_out))
        self.assertEqual(fsa_out.size1, 6)
        self.assertEqual(fsa_out.size2, 11)
        self.assertEqual(arc_derivs.size1, 11)
        self.assertEqual(arc_derivs.size2, 20)
        self.assertTrue(
            k2.is_rand_equivalent_after_rmeps_pruned_logsum(
                self.fsa, self.weights, fsa_out, arc_weights_out, beam))


if __name__ == '__main__':
    unittest.main()
