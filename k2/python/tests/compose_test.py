#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R compose_test_py

import unittest

import _k2
import k2
import torch


class TestCompose(unittest.TestCase):

    def test_compose(self):
        s = '''
            0 1 11 1 1.0
            0 2 12 2 2.5
            1 3 -1 -1 0
            2 3 -1 -1 2.5
            3
        '''
        a_fsa = k2.Fsa.from_str(s).requires_grad_(True)

        s = '''
            0 1 1 1 1.0
            0 2 2 3 3.0
            1 2 3 2 2.5
            2 3 -1 -1 2.0
            3
        '''
        b_fsa = k2.Fsa.from_str(s).requires_grad_(True)

        ans = k2.compose(a_fsa, b_fsa, inner_labels='inner')
        ans = k2.connect(ans)

        ans = k2.create_fsa_vec([ans])

        scores = k2.get_tot_scores(ans,
                                   log_semiring=True,
                                   use_double_scores=False)
        # The reference values for `scores`, `a_fsa.grad` and `b_fsa.grad`
        # are computed using GTN.
        # See https://bit.ly/3heLAJq
        assert scores.item() == 10
        scores.backward()
        assert torch.allclose(a_fsa.grad, torch.tensor([0., 1., 0., 1.]))
        assert torch.allclose(b_fsa.grad, torch.tensor([0., 1., 0., 1.]))
        print(ans)


if __name__ == '__main__':
    unittest.main()
