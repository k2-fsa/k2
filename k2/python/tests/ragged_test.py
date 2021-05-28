#!/usr/bin/env python3
#
# Copyright (c)  2020   Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  ragged_test_py

import unittest

import k2
import torch


class TestRagged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_ragged_int_from_str(self):
        s = '''
        [ [1 2] [3] ]
        '''
        for device in self.devices:
            ragged_int = k2.RaggedInt(s).to(device)
            print(ragged_int)
            assert torch.all(
                torch.eq(ragged_int.values(),
                         torch.tensor([1, 2, 3], device=device)))
            assert ragged_int.dim0() == 2
            assert torch.all(
                torch.eq(ragged_int.row_splits(1),
                         torch.tensor([0, 2, 3], device=device)))

            self.assertEqual([2, 3], ragged_int.tot_sizes())


if __name__ == '__main__':
    unittest.main()
