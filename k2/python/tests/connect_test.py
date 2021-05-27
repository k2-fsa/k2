#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R connect_test_py

import unittest

import k2
import torch


class TestConnect(unittest.TestCase):
    s = '''
        0 1 1 0.1
        0 2 2 0.2
        1 4 -1 0.3
        3 4 -1 0.4
        4
    '''
    fsa = k2.Fsa.from_str(s)
    fsa.requires_grad_(True)
    expected_str = '\n'.join(['0 1 1 0.1', '1 2 -1 0.3', '2'])
    connected_fsa = k2.connect(fsa)
    actual_str = k2.to_str_simple(connected_fsa)
    assert actual_str.strip() == expected_str

    loss = connected_fsa.scores.sum()
    loss.backward()
    assert torch.allclose(fsa.scores.grad,
                          torch.tensor([1, 0, 1, 0], dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
