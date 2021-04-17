#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R prune_on_arc_post_test_py

import unittest

import k2
import torch


class TestPruneOnArcPost(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))

    def test(self):
        for device in self.devices:
            s = '''
                0 1 1 0.2
                0 1 2 0.1
                1 2 2 0.1
                1 2 3 0.2
                2 3 -1 0
                3
            '''
            fsa = k2.Fsa.from_str(s)
            fsa_vec = k2.create_fsa_vec([fsa])
            threshold_prob = 0.5
            ans = k2.prune_on_arc_post(fsa_vec,
                                       threshold_prob,
                                       use_double_scores=True)
            expected = k2.Fsa.from_str('''
                0 1 1 0.2
                1 2 3 0.2
                2 3 -1 0
                3
            ''')
            assert str(ans[0]) == str(expected)


if __name__ == '__main__':
    unittest.main()
