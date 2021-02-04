#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R index_and_sum_test_py

import unittest

import k2
import torch


class TestIndexAndSum(unittest.TestCase):

    def test_without_negative_1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            src = torch.tensor([0, 1, 2, 3],
                               dtype=torch.float32,
                               requires_grad=True,
                               device=device)
            indexes = k2.RaggedInt('[ [1 2] [0 3] [0 2 3 1 3] [] ]').to(device)
            ans = k2.index_and_sum(src, indexes)
            expected = torch.tensor([1 + 2, 0 + 3, 0 + 2 + 3 + 1 + 3,
                                     0]).to(src)
            assert torch.all(torch.eq(ans, expected))

            # now for autograd
            scale = torch.tensor([10, 20, 30, 40]).to(device)
            (ans * scale).sum().backward()

            expected_grad = torch.empty_like(src.grad)
            expected_grad[0] = scale[1] + scale[2]
            expected_grad[1] = scale[0] + scale[2]
            expected_grad[2] = scale[0] + scale[2]
            expected_grad[3] = scale[1] + scale[2] * 2

            assert torch.all(torch.eq(src.grad, expected_grad))

    def test_with_negative_1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            src = torch.tensor([0, 1, 2, 3],
                               dtype=torch.float32,
                               requires_grad=True,
                               device=device)
            indexes = k2.RaggedInt(
                '[ [1 2 -1] [0 3] [-1] [0 2 3 1 3] [] ]').to(device)
            ans = k2.index_and_sum(src, indexes)
            expected = torch.tensor([1 + 2, 0 + 3, 0, 0 + 2 + 3 + 1 + 3,
                                     0]).to(src)
            assert torch.all(torch.eq(ans, expected))

            # now for autograd
            scale = torch.tensor([10, 20, 30, 40, 50]).to(device)
            (ans * scale).sum().backward()
            expected_grad = torch.empty_like(src.grad)
            expected_grad[0] = scale[1] + scale[3]
            expected_grad[1] = scale[0] + scale[3]
            expected_grad[2] = scale[0] + scale[3]
            expected_grad[3] = scale[1] + scale[3] * 2

            assert torch.all(torch.eq(src.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
