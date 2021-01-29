#!/usr/bin/env python3
#
# Copyright (c)  2020   Xiaomi Corporation (authors: Fangjun Kuang)
#                2021   Mobvoi Inc.        (authors: Yaguang Hu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  ragged_test_py

import unittest

import k2
import torch


class TestRagged(unittest.TestCase):

    def test_ragged_int_from_str(self):
        s = '''
        [ [1 2] [3] ]
        '''
        ragged_int = k2.RaggedInt(s)
        print(ragged_int)
        assert torch.all(torch.eq(ragged_int.values(), torch.tensor([1, 2, 3])))
        assert ragged_int.dim0() == 2
        assert torch.all(
            torch.eq(ragged_int.row_splits(1), torch.tensor([0, 2, 3])))

        self.assertEqual([2, 3], ragged_int.tot_sizes())

    def test_ragged_int_from_list(self):
        lst = [[7, 9], [12, 13], []]
        ragged_int = k2.RaggedInt(lst)
        print(ragged_int)
        assert torch.all(
            torch.eq(ragged_int.values(), torch.tensor([7, 9, 12, 13])))
        assert ragged_int.dim0() == 3
        assert torch.all(
            torch.eq(ragged_int.row_splits(1), torch.tensor([0, 2, 4, 4])))

        self.assertEqual([3, 4], ragged_int.tot_sizes())


if __name__ == '__main__':
    unittest.main()
