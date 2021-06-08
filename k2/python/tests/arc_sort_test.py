#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R arc_sort_test_py

import unittest

import k2
import torch


class TestArcSort(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        s = '''
            0 1 2 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s)
            fsa.requires_grad_(True)
            sorted_fsa = k2.arc_sort(fsa)

            actual_str = k2.to_str_simple(sorted_fsa)
            expected_str = '\n'.join(
                ['0 1 1 0.2', '0 1 2 0.1', '1 2 -1 0.3', '2'])
            assert actual_str.strip() == expected_str

            loss = (sorted_fsa.scores[1] + sorted_fsa.scores[2]) / 2
            loss.backward()
            assert torch.allclose(
                fsa.scores.grad,
                torch.tensor([0.5, 0, 0.5], dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
