#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R top_sort_test_py

import unittest

import k2
import torch


def _remove_leading_spaces(s: str) -> str:
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    return '\n'.join(lines)


class TestTopSort(unittest.TestCase):

    def test(self):
        # arc 0: 0 -> 1, weight 1
        # arc 1: 0 -> 2, weight 2
        # arc 2: 1 -> 3, weight 3
        # arc 3: 2 -> 1, weight 4
        # the shortest path is 0 -> 1 -> 3, weight is 4
        # That is, (arc 0) -> (arc 2)
        s = '''
            0 1 1 1
            0 2 2 2
            1 3 -1 3
            2 1 3 4
            1
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.score.requires_grad_(True)
        sorted_fsa = k2.top_sort(fsa)

        # the shortest path in the sorted fsa is (arc 0) -> (arc 3)
        loss = (sorted_fsa.score[0] + sorted_fsa.score[3]) / 2
        loss.backward()
        assert torch.allclose(
            fsa.score.grad, torch.tensor([0.5, 0, 0.5, 0],
                                         dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
