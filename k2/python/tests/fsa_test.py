#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_test_py

import unittest

import k2
import torch


class TestFsa(unittest.TestCase):

    def test_fsa_from_str(self):
        s = '''0 1 2 -1.2
        0 2  10 -2.2
        1 3  3  -3.2
        1 6 -1  -4.2
        2 6 -1  -5.2
        2 4  2  -6.2
        3 6 -1  -7.2
        5 0  1  -8.2
        6
        '''

        fsa = k2.str_to_fsa(s)
        expected_str = '''0 1 2 -1.2
0 2 10 -2.2
1 3 3 -3.2
1 6 -1 -4.2
2 6 -1 -5.2
2 4 2 -6.2
3 6 -1 -7.2
5 0 1 -8.2
6
'''
        assert expected_str == k2.fsa_to_str(fsa)

        arcs = fsa.arcs()
        assert isinstance(arcs, torch.Tensor)
        assert arcs.shape == (8, 4), 'there should be 8 arcs'
        assert arcs[0][0] == 0
        assert arcs[0][1] == 1
        assert arcs[0][2] == 2
        assert arcs[0][3] == k2.float_as_int(-1.2)
        assert arcs.device.type == 'cpu'

        fsa = fsa.cuda(gpu_id=0)
        arcs[0][0] += 10
        assert arcs[0][0] == 10, 'arcs should still be accessible'

        arcs = fsa.arcs()
        assert arcs.device.type == 'cuda'
        assert arcs.device.index == 0
        assert arcs[0][0] == 0
        assert arcs[0][1] == 1
        assert arcs[0][2] == 2
        assert arcs[0][3] == k2.float_as_int(-1.2)

        fsa = fsa.cpu()
        arcs = fsa.arcs()
        assert arcs.device.type == 'cpu'
        assert arcs[0][0] == 0
        assert arcs[0][1] == 1
        assert arcs[0][2] == 2
        assert arcs[0][3] == k2.float_as_int(-1.2)

        fsa2 = k2.Fsa(arcs)  # construct an FSA from a tensor
        del fsa, arcs

        arcs = fsa2.arcs()
        assert arcs.device.type == 'cpu'
        assert arcs[0][0] == 0
        assert arcs[0][1] == 1
        assert arcs[0][2] == 2
        assert arcs[0][3] == k2.float_as_int(-1.2)


if __name__ == '__main__':
    unittest.main()
