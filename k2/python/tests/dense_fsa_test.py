#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R dense_fsa_test_py

import unittest

import k2
import torch


class TestDenseFsa(unittest.TestCase):

    def test_dense_fsa_cpu(self):
        log_prob = torch.arange(20).reshape(2, 5, 2).to(torch.float32)
        supervision_segments = torch.tensor([
            # seq_index, start_time, duration
            [0, 0, 3],
            [0, 1, 4],
            [1, 0, 2],
            [0, 2, 3],
            [1, 3, 2],
        ]).to(torch.int32)

        dense_fsa = k2.dense_fsa(log_prob, supervision_segments)
        print(dense_fsa.to_str())
        print(log_prob)
        # TODO(fangjun): Let the computer check the output
        '''
        num_axes: 2
        device_type: kCpu
        row_splits1: [ 0 4 9 12 16 19 ]
        row_ids1: [ 0 0 0 0 1 1 1 1 1 2 2 2 3 3 3 3 4 4 4 ]
        scores:
        [[ -inf 0 1 ]
        [ -inf 2 3 ]
        [ -inf 4 5 ]
        [ 0 -inf -inf ]
        [ -inf 2 3 ]
        [ -inf 4 5 ]
        [ -inf 6 7 ]
        [ -inf 8 9 ]
        [ 0 -inf -inf ]
        [ -inf 10 11 ]
        [ -inf 12 13 ]
        [ 0 -inf -inf ]
        [ -inf 4 5 ]
        [ -inf 6 7 ]
        [ -inf 8 9 ]
        [ 0 -inf -inf ]
        [ -inf 16 17 ]
        [ -inf 18 19 ]
        [ 0 -inf -inf ]
        ]

        tensor([[[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.],
                 [ 6.,  7.],
                 [ 8.,  9.]],

                [[10., 11.],
                 [12., 13.],
                 [14., 15.],
                 [16., 17.],
                 [18., 19.]]])
        '''

    def test_dense_fsa_cuda(self):
        device = torch.device('cuda', index=0)
        log_prob = torch.arange(20).reshape(2, 5, 2).to(torch.float32)
        supervision_segments = torch.tensor([
            [0, 0, 3],
            [0, 1, 4],
            [1, 0, 2],
            [0, 2, 3],
            [1, 3, 2],
        ]).to(torch.int32)

        log_prob = log_prob.to(device)

        dense_fsa = k2.dense_fsa(log_prob, supervision_segments)
        print(dense_fsa.to_str())
        print(log_prob)


if __name__ == '__main__':
    unittest.main()
