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

    def test_treat_epsilon_specially_false(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            # a_fsa recognizes `(0|1)2*`
            s1 = '''
                0 1 0 0.1
                0 1 1 0.2
                1 1 2 0.3
                1 2 -1 0.4
                2
            '''
            a_fsa = k2.Fsa.from_str(s1).to(device)
            a_fsa.requires_grad_(True)

            # b_fsa recognizes `1|2`
            s2 = '''
                0 1 1 1
                0 1 2 2
                1 2 -1 3
                2
            '''
            b_fsa = k2.Fsa.from_str(s2).to(device)
            b_fsa.requires_grad_(True)

            # fsa recognizes `1`
            fsa = k2.intersect(a_fsa, b_fsa, treat_epsilons_specially=False)
            assert len(fsa.shape) == 2
            actual_str = k2.to_str(fsa)
            expected_str = '\n'.join(['0 1 1 1.2', '1 2 -1 3.4', '2'])
            assert actual_str.strip() == expected_str

            loss = fsa.scores.sum()
            (-loss).backward()
            # arc 1 and 3 of a_fsa are kept in the final intersected FSA
            assert torch.allclose(a_fsa.grad,
                                  torch.tensor([0, -1, 0, -1]).to(a_fsa.grad))

            # arc 0 and 2 of b_fsa are kept in the final intersected FSA
            assert torch.allclose(b_fsa.grad,
                                  torch.tensor([-1, 0, -1]).to(b_fsa.grad))

            # if any of the input FSA is an FsaVec,
            # the outupt FSA is also an FsaVec.
            a_fsa.scores.grad = None
            b_fsa.scores.grad = None
            a_fsa = k2.create_fsa_vec([a_fsa])
            fsa = k2.intersect(a_fsa, b_fsa, treat_epsilons_specially=False)
            assert len(fsa.shape) == 3

    def test_treat_epsilon_specially_true(self):
        # this version works only on CPU and requires
        # arc-sorted inputs
        # a_fsa recognizes `(1|3)?2*`
        s1 = '''
            0 1 3 0.0
            0 1 1 0.2
            0 1 0 0.1
            1 1 2 0.3
            1 2 -1 0.4
            2
        '''
        a_fsa = k2.Fsa.from_str(s1)
        a_fsa.requires_grad_(True)

        # b_fsa recognizes `1|2|5`
        s2 = '''
            0 1 5 0
            0 1 1 1
            0 1 2 2
            1 2 -1 3
            2
        '''
        b_fsa = k2.Fsa.from_str(s2)
        b_fsa.requires_grad_(True)

        # fsa recognizes 1|2
        fsa = k2.intersect(k2.arc_sort(a_fsa), k2.arc_sort(b_fsa))
        assert len(fsa.shape) == 2
        actual_str = k2.to_str(fsa)
        expected_str = '\n'.join(
            ['0 1 0 0.1', '0 2 1 1.2', '1 2 2 2.3', '2 3 -1 3.4', '3'])
        assert actual_str.strip() == expected_str

        loss = fsa.scores.sum()
        (-loss).backward()
        # arc 1, 2, 3, and 4 of a_fsa are kept in the final intersected FSA
        assert torch.allclose(a_fsa.grad,
                              torch.tensor([0, -1, -1, -1, -1]).to(a_fsa.grad))

        # arc 1, 2, and 3 of b_fsa are kept in the final intersected FSA
        assert torch.allclose(b_fsa.grad,
                              torch.tensor([0, -1, -1, -1]).to(b_fsa.grad))

        # if any of the input FSA is an FsaVec,
        # the outupt FSA is also an FsaVec.
        a_fsa.scores.grad = None
        b_fsa.scores.grad = None
        a_fsa = k2.create_fsa_vec([a_fsa])
        fsa = k2.intersect(k2.arc_sort(a_fsa), k2.arc_sort(b_fsa))
        assert len(fsa.shape) == 3


if __name__ == '__main__':
    unittest.main()
