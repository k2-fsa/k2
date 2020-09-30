#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_test_py -E host

import unittest

import k2
import torch


class TestFsa(unittest.TestCase):

    def test_acceptor_from_str(self):
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

        fsa, aux_labels = k2.fsa_from_str(s)
        assert aux_labels is None

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

        expected_str = '''0 1 2 1.2
0 2 10 2.2
1 3 3 3.2
1 6 -1 4.2
2 6 -1 5.2
2 4 2 6.2
3 6 -1 7.2
5 0 1 8.2
6
'''
        assert expected_str == k2.fsa_to_str(fsa, negate_scores=True)

        arcs = fsa.arcs()
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 4), 'there should be 8 arcs'
        assert arcs[0][0] == 0
        assert arcs[0][1] == 1
        assert arcs[0][2] == 2
        assert arcs[0][3] == k2.float_as_int(-1.2)

        weights = k2.int_as_float(arcs[:, -1])
        assert torch.allclose(
            weights,
            torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        arcs[0][3] = k2.float_as_int(-2020.0930)
        assert weights[0] == -2020.0930, \
                'Memory should be shared between weights and arcs!'

        fsa = fsa.cuda(gpu_id=0)
        arcs[0][0] += 10
        assert arcs[0][0] == 10, 'arcs should still be accessible'

        arcs = fsa.arcs()
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cuda'
        assert arcs.device.index == 0
        assert arcs.shape == (8, 4), 'there should be 8 arcs'
        assert arcs[1][0] == 0
        assert arcs[1][1] == 2
        assert arcs[1][2] == 10
        assert arcs[1][3] == k2.float_as_int(-2.2)

        fsa = fsa.cpu()
        arcs = fsa.arcs()
        assert arcs.device.type == 'cpu'
        assert arcs[2][0] == 1
        assert arcs[2][1] == 3
        assert arcs[2][2] == 3
        assert arcs[2][3] == k2.float_as_int(-3.2)

        fsa2 = k2.Fsa(arcs)  # construct an FSA from a tensor
        del fsa, arcs

        arcs = fsa2.arcs()
        assert arcs.device.type == 'cpu'
        assert arcs[3][0] == 1
        assert arcs[3][1] == 6
        assert arcs[3][2] == -1
        assert arcs[3][3] == k2.float_as_int(-4.2)

    def test_transducer_from_str(self):
        s = '''0 1 2 22  -1.2
            0 2  10 100 -2.2
            1 3  3  33  -3.2
            1 6 -1  16  -4.2
            2 6 -1  26  -5.2
            2 4  2  22  -6.2
            3 6 -1  36  -7.2
            5 0  1  50  -8.2
            6
        '''
        fsa, aux_labels = k2.fsa_from_str(s)
        assert isinstance(aux_labels, torch.Tensor)
        assert aux_labels.dtype == torch.int32
        assert aux_labels.device.type == 'cpu'
        assert torch.allclose(
            aux_labels,
            torch.tensor([22, 100, 33, 16, 26, 22, 36, 50]).to(torch.int32))

        expected_str = '''0 1 2 22 -1.2
0 2 10 100 -2.2
1 3 3 33 -3.2
1 6 -1 16 -4.2
2 6 -1 26 -5.2
2 4 2 22 -6.2
3 6 -1 36 -7.2
5 0 1 50 -8.2
6
'''
        assert expected_str == k2.fsa_to_str(fsa, aux_labels=aux_labels)

        expected_str = '''0 1 2 22 1.2
0 2 10 100 2.2
1 3 3 33 3.2
1 6 -1 16 4.2
2 6 -1 26 5.2
2 4 2 22 6.2
3 6 -1 36 7.2
5 0 1 50 8.2
6
'''
        assert expected_str == k2.fsa_to_str(fsa,
                                             negate_scores=True,
                                             aux_labels=aux_labels)


if __name__ == '__main__':
    unittest.main()
