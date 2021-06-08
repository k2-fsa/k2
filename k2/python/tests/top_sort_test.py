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


class TestTopSort(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

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
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            sorted_fsa = k2.top_sort(fsa)

            # the shortest path in the sorted fsa is (arc 0) -> (arc 3)
            loss = (sorted_fsa.scores[0] + sorted_fsa.scores[3]) / 2
            loss.backward()
            assert torch.allclose(
                fsa.scores.grad,
                torch.tensor([0.5, 0, 0.5, 0],
                             dtype=torch.float32,
                             device=device))


if __name__ == '__main__':
    unittest.main()
