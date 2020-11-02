#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_rmepsilon_test_py
#

from struct import pack, unpack
import unittest

import torch

import k2host


class TestRmEpsilon(unittest.TestCase):

    def setUp(self):
        s = r'''
        0 4 1 1
        0 1 1 1
        1 2 0 2
        1 3 0 3
        1 4 0 2
        2 7 0 4
        3 7 0 5
        4 6 1 2
        4 6 0 3
        4 8 1 3
        4 9 -1 2
        5 9 -1 4
        6 9 -1 3
        7 9 -1 5
        8 9 -1 6
        9
        '''
        self.fsa = k2host.str_to_fsa(s)
        self.num_states = self.fsa.num_states()

    def test_max_weight(self):
        forward_max_weights = k2host.DoubleArray1.create_array_with_size(
            self.num_states)
        backward_max_weights = k2host.DoubleArray1.create_array_with_size(
            self.num_states)
        wfsa = k2host.WfsaWithFbWeights(self.fsa,
                                        k2host.FbWeightType.kMaxWeight,
                                        forward_max_weights,
                                        backward_max_weights)
        beam = 8.0
        remover = k2host.EpsilonsRemoverMax(wfsa, beam)
        fsa_size = k2host.IntArray2Size()
        arc_derivs_size = k2host.IntArray2Size()
        remover.get_sizes(fsa_size, arc_derivs_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(fsa_size)
        arc_derivs = k2host.IntArray2.create_array_with_size(arc_derivs_size)
        arc_weights_out = k2host.FloatArray1.create_array_with_size(
            fsa_size.size2)
        remover.get_output(fsa_out, arc_derivs)
        self.assertTrue(k2host.is_epsilon_free(fsa_out))
        self.assertEqual(fsa_out.size1, 6)
        self.assertEqual(fsa_out.size2, 11)  # TODO: fix this
        self.assertEqual(arc_derivs.size1, 11)  # TODO: fix this
        self.assertEqual(arc_derivs.size2, 18)  # TODO: fix this
        self.assertTrue(
            k2host.is_rand_equivalent_max_weight(self.fsa, fsa_out, beam))

    def test_logsum_weight(self):
        forward_logsum_weights = k2host.DoubleArray1.create_array_with_size(
            self.num_states)
        backward_logsum_weights = k2host.DoubleArray1.create_array_with_size(
            self.num_states)
        wfsa = k2host.WfsaWithFbWeights(self.fsa,
                                        k2host.FbWeightType.kLogSumWeight,
                                        forward_logsum_weights,
                                        backward_logsum_weights)
        beam = 8.0
        remover = k2host.EpsilonsRemoverLogSum(wfsa, beam)
        fsa_size = k2host.IntArray2Size()
        arc_derivs_size = k2host.IntArray2Size()
        remover.get_sizes(fsa_size, arc_derivs_size)
        fsa_out = k2host.Fsa.create_fsa_with_size(fsa_size)
        arc_derivs = k2host.LogSumArcDerivs.create_arc_derivs_with_size(
            arc_derivs_size)
        arc_weights_out = k2host.FloatArray1.create_array_with_size(
            fsa_size.size2)
        remover.get_output(fsa_out, arc_derivs)
        self.assertTrue(k2host.is_epsilon_free(fsa_out))
        self.assertEqual(fsa_out.size1, 6)
        self.assertEqual(fsa_out.size2, 11)  # TODO: fix this
        self.assertEqual(arc_derivs.size1, 11)  # TODO: fix this
        self.assertEqual(arc_derivs.size2, 20)  # TODO: fix this
        # TODO(haowen): uncomment this after re-implementing
        # IsRandEquivalentAfterRmEpsPrunedLogSum
        #self.assertTrue(
        #    k2host.is_rand_equivalent_after_rmeps_pruned_logsum(
        #        self.fsa, fsa_out, beam))
        # cast float to int
        arc_ids = k2host.StridedIntArray1.from_float_tensor(arc_derivs.data[:,
                                                                            0])
        # we may get different value of `arc_ids.get_data(1)`
        # with different STL implementations as we use
        # `std::unordered_map` in implementation of rmepsilon,
        # thus below assertion may fail on some platforms.
        self.assertEqual(arc_ids.get_data(1), 1)


if __name__ == '__main__':
    unittest.main()
