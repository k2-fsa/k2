#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R host_fsa_test_py
#

import unittest

import torch

import k2host
from k2host.fsa_util import float_to_int


class TestFsa(unittest.TestCase):

    def test_arc(self):
        # construct arc
        arc = k2host.Arc(1, 2, 3, 1.5)
        self.assertEqual(arc.src_state, 1)
        self.assertEqual(arc.dest_state, 2)
        self.assertEqual(arc.label, 3)
        self.assertEqual(arc.weight, 1.5)

        # test from_tensor
        arc_tensor = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
        arc = k2host.Arc.from_tensor(arc_tensor)
        self.assertEqual(arc.src_state, 1)
        self.assertEqual(arc.dest_state, 2)
        self.assertEqual(arc.label, 3)
        self.assertEqual(arc.weight, 0)

        # test to_tensor
        arc.src_state = 2
        arc_tensor = arc.to_tensor()
        arc_tensor_target = torch.tensor([2, 2, 3, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(arc_tensor, arc_tensor_target))

    def test_fsa(self):
        s = r'''
        0 1 1 1.25
        0 2 2 1.5
        1 3 3 1.75
        2 3 3 2.25
        3 4 -1 2.5
        4
        '''

        fsa = k2host.str_to_fsa(s)
        self.assertEqual(fsa.num_states(), 5)
        self.assertEqual(fsa.final_state(), 4)
        self.assertFalse(fsa.empty())
        self.assertIsInstance(fsa, k2host.Fsa)
        # test get_data
        self.assertEqual(fsa.get_data(0).src_state, 0)
        self.assertEqual(fsa.get_data(0).dest_state, 1)
        self.assertEqual(fsa.get_data(0).label, 1)
        self.assertEqual(fsa.get_data(0).weight, 1.25)
        self.assertEqual(fsa.get_data(1).weight, 1.5)
        self.assertEqual(fsa.get_data(2).weight, 1.75)
        self.assertEqual(fsa.get_data(3).weight, 2.25)
        self.assertEqual(fsa.get_data(4).weight, 2.5)
        # fsa.data and the corresponding k2host::Fsa object are sharing memory
        fsa.data[0] = torch.IntTensor([5, 1, 6, 1])
        self.assertEqual(fsa.get_data(0).src_state, 5)


if __name__ == '__main__':
    unittest.main()
