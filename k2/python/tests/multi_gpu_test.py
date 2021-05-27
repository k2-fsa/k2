#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  multi_gpu_test_py

import unittest

import k2
import torch


class TestMultiGPU(unittest.TestCase):

    def test(self):
        if torch.cuda.is_available() is False:
            print('skip it since CUDA is not available')
            return
        if torch.cuda.device_count() < 2:
            print('skip it since number of GPUs is 1')
            return

        device0 = torch.device('cuda', 0)
        device1 = torch.device('cuda', 1)

        torch.cuda.set_device(device1)

        r0 = k2.RaggedInt('[ [[0] [1]] ]').to(device0)
        r1 = k2.RaggedInt('[ [[0] [1]] ]').to(device1)

        assert torch.cuda.current_device() == 1

        r0 = k2.ragged.remove_axis(r0, 0)
        r1 = k2.ragged.remove_axis(r1, 0)

        expected_r0 = k2.RaggedInt('[[0] [1]]').to(device0)
        expected_r1 = k2.RaggedInt('[[0] [1]]').to(device1)

        assert torch.all(torch.eq(r0.row_splits(1), expected_r0.row_splits(1)))
        assert torch.all(torch.eq(r1.row_splits(1), expected_r1.row_splits(1)))

        try:
            # will throw an exception because they two are not on the same device
            assert torch.all(
                torch.eq(r0.row_splits(1), expected_r1.row_splits(1)))
        except RuntimeError as e:
            print(e)

        assert torch.cuda.current_device() == 1


if __name__ == '__main__':
    unittest.main()
