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

    def test_dense_fsa(self):
        log_prob = torch.arange(12).reshape(2, 3, 2).to(torch.float32)
        input_lengths = torch.tensor([3, 2], dtype=torch.int32)

        dense_fsa = k2.dense_fsa(log_prob, input_lengths)
        s = dense_fsa.to_str()

        # TODO(fangjun): let the computer check the output.
        print(s)
        '''
        num_axes:
        2
        row_splits1:
        [ 0 4 7 ]
        row_ids1:
        [ 0 0 0 0 1 1 1 ]
        scores:

        [[ -inf 0 1 ]
        [ -inf 2 3 ]
        [ -inf 4 5 ]
        [ 0 -inf -inf ]
        [ -inf 6 7 ]
        [ -inf 8 9 ]
        [ 0 -inf -inf ]
        ]
        '''


if __name__ == '__main__':
    unittest.main()
