#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R intersect_test_py

import unittest

import k2
import torch


class TestIntersect(unittest.TestCase):

    def test(self):
        # for the symbol table
        # <eps> 0
        # a 0
        # b 1
        # c 2

        # an FSA that recognizes a+(b|c)
        s = '''
            0 1 1 0.1
            1 1 1 0.2
            1 2 2 0.3
            1 3 3 0.4
            2 4 -1 0.5
            3 4 -1 0.6
            5
        '''
        a_fsa = k2.Fsa.from_str(s)
        a_fsa.requires_grad_(True)

        # an FSA that recognizes ab
        s = '''
            0 1 1 10
            1 2 2 20
            2 3 -1 30
            3
        '''
        b_fsa = k2.Fsa.from_str(s)
        b_fsa.requires_grad_(True)

        fsa = k2.intersect(a_fsa, b_fsa)
        assert len(fsa.shape) == 2
        actual_str = k2.to_str(fsa)
        expected_str = '\n'.join(
            ['0 1 1 10.1', '1 2 2 20.3', '2 3 -1 30.5', '3'])
        assert actual_str.strip() == expected_str

        loss = fsa.scores.sum()
        loss.backward()
        # arc 0, 2, and 4 of a_fsa are kept in the final intersected FSA
        assert torch.allclose(
            a_fsa.scores.grad,
            torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32))

        assert torch.allclose(b_fsa.scores.grad,
                              torch.tensor([1, 1, 1], dtype=torch.float32))

        # if any of the input FSA is an FsaVec,
        # the outupt FSA is also an FsaVec.
        a_fsa.scores.grad = None
        b_fsa.scores.grad = None
        a_fsa = k2.create_fsa_vec([a_fsa])
        fsa = k2.intersect(a_fsa, b_fsa)
        assert len(fsa.shape) == 3


if __name__ == '__main__':
    unittest.main()
