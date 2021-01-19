#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R get_backward_scores_test_py

import unittest

import k2
import torch


class TestGetBackwardScores(unittest.TestCase):

    def test_simple_fsa_case_1(self):
        s = '''
            0 1 1 0.0
            0 1 2 0.1
            0 2 3 2.2
            1 2 4 0.5
            1 2 5 0.6
            1 3 -1 3.0
            2 3 -1 0.8
            3
        '''
        devices = [torch.device('cpu')]
        #  if torch.cuda.is_available():
        #      devices.append(torch.device('cuda'))

        for device in devices:
            for use_double_scores in [True, False]:
                fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
                fsa_vec = k2.create_fsa_vec([fsa])
                backward_scores = fsa_vec.get_backward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                print(backward_scores)


if __name__ == '__main__':
    unittest.main()
