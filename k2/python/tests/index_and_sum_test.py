#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R index_and_sum_test_py

import unittest

import k2
import torch


class TestIndexAndSum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_without_negative_1(self):
        for device in self.devices:
            src = torch.tensor([0, 1, 2, 3],
                               dtype=torch.float32,
                               requires_grad=True,
                               device=device)
            indexes = k2.RaggedTensor([[1, 2], [0, 3], [0, 2, 3, 1, 3],
                                       []]).to(device)
            ans = k2.ragged.index_and_sum(src, indexes)
            expected = torch.tensor([1 + 2, 0 + 3, 0 + 2 + 3 + 1 + 3,
                                     0]).to(src)
            assert torch.all(torch.eq(ans, expected)), \
                    f'ans: {ans}, expected: {expected}'

            # now for autograd
            scale = torch.tensor([10, 20, 30, 40]).to(device)
            (ans * scale).sum().backward()

            expected_grad = torch.empty_like(src.grad)
            expected_grad[0] = scale[1] + scale[2]
            expected_grad[1] = scale[0] + scale[2]
            expected_grad[2] = scale[0] + scale[2]
            expected_grad[3] = scale[1] + scale[2] * 2

            assert torch.allclose(src.grad, expected_grad)

    def test_with_negative_1(self):
        for device in self.devices:
            src = torch.tensor([0, 1, 2, 3],
                               dtype=torch.float32,
                               requires_grad=True,
                               device=device)
            indexes = k2.RaggedTensor([[1, 2, -1], [0, 3], [-1],
                                       [0, 2, 3, 1, 3], []]).to(device)
            ans = k2.ragged.index_and_sum(src, indexes)
            expected = torch.tensor([1 + 2, 0 + 3, 0, 0 + 2 + 3 + 1 + 3,
                                     0]).to(src)
            assert torch.allclose(ans, expected)

            # now for autograd
            scale = torch.tensor([10, 20, 30, 40, 50]).to(device)
            (ans * scale).sum().backward()
            expected_grad = torch.empty_like(src.grad)
            expected_grad[0] = scale[1] + scale[3]
            expected_grad[1] = scale[0] + scale[3]
            expected_grad[2] = scale[0] + scale[3]
            expected_grad[3] = scale[1] + scale[3] * 2

            assert torch.allclose(src.grad, expected_grad)


if __name__ == '__main__':
    unittest.main()
