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
#  ctest --verbose -R  fsa_from_unary_function_tensor_test_py

import unittest

import k2
import torch
import _k2


class TestFsaFromUnaryFunctionTensor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        for device in self.devices:
            s = '''
                0 1 2 10
                0 1 1 20
                1 2 -1 30
                2
            '''
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            src.float_attr = torch.tensor([0.1, 0.2, 0.3],
                                          dtype=torch.float32,
                                          requires_grad=True,
                                          device=device)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[1, 2, 3], [5, 6],
                                               []]).to(device)
            src.attr1 = 'src'
            src.attr2 = 'fsa'

            ragged_arc, arc_map = _k2.arc_sort(src.arcs, need_arc_map=True)

            dest = k2.utils.fsa_from_unary_function_tensor(
                src, ragged_arc, arc_map)

            assert torch.allclose(
                dest.float_attr,
                torch.tensor([0.2, 0.1, 0.3],
                             dtype=torch.float32,
                             device=device))

            assert torch.all(
                torch.eq(
                    dest.scores,
                    torch.tensor([20, 10, 30],
                                 dtype=torch.float32,
                                 device=device)))

            assert torch.all(
                torch.eq(
                    dest.int_attr,
                    torch.tensor([2, 1, 3], dtype=torch.int32, device=device)))

            expected_ragged_attr = k2.RaggedTensor([[5, 6], [1, 2, 3],
                                                    []]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            # now for autograd
            scale = torch.tensor([10, 20, 30], device=device)
            (dest.float_attr * scale).sum().backward()
            (dest.scores * scale).sum().backward()

            expected_grad = torch.tensor([20, 10, 30],
                                         dtype=torch.float32,
                                         device=device)

            assert torch.all(torch.eq(src.float_attr.grad, expected_grad))

            assert torch.all(torch.eq(src.scores.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
