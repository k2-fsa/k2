#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang, Daniel Povey)
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
#  ctest --verbose -R  expand_ragged_attributes_tests_py

import unittest

import k2
import torch
import _k2


class TestExpandArcs(unittest.TestCase):

    def test(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            devices.append(torch.device('cuda', 0))

        for device in devices:
            for need_map in [True, False]:
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

                if need_map:
                    dest, arc_map = k2.expand_ragged_attributes(
                        src, ret_arc_map=True)
                else:
                    dest = k2.expand_ragged_attributes(src)

                assert torch.allclose(
                    dest.float_attr,
                    torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 0.3],
                                 dtype=torch.float32,
                                 device=device))

                assert torch.all(
                    torch.eq(
                        dest.scores,
                        torch.tensor([10, 20, 0, 0, 0, 30],
                                     dtype=torch.float32,
                                     device=device)))

                assert torch.all(
                    torch.eq(
                        dest.int_attr,
                        torch.tensor([1, 2, 0, 0, 0, 3],
                                     dtype=torch.int32,
                                     device=device)))

                assert torch.all(
                    torch.eq(
                        dest.ragged_attr,
                        torch.tensor([1, 5, 2, 3, 6, -1],
                                     dtype=torch.float32,
                                     device=device)))

                # non-tensor attributes...
                assert dest.attr1 == src.attr1
                assert dest.attr2 == src.attr2

                # now for autograd
                scale = torch.tensor([10, 20, 10, 10, 10, 30], device=device)
                (dest.float_attr * scale).sum().backward()
                (dest.scores * scale).sum().backward()

                expected_grad = torch.tensor([10, 20, 30],
                                             dtype=torch.float32,
                                             device=device)

                assert torch.all(torch.eq(src.float_attr.grad, expected_grad))

                assert torch.all(torch.eq(src.scores.grad, expected_grad))

    def test_final(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            devices.append(torch.device('cuda', 0))

        for device in devices:
            for need_map in [True, False]:
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
                                                   [1]]).to(device)

                src.attr1 = 'src'
                src.attr2 = 'fsa'

                if need_map:
                    dest, arc_map = k2.expand_ragged_attributes(
                        src, ret_arc_map=True)
                else:
                    dest = k2.expand_ragged_attributes(src)

                assert torch.allclose(
                    dest.float_attr,
                    torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0],
                                 dtype=torch.float32,
                                 device=device))
                assert torch.all(
                    torch.eq(
                        dest.scores,
                        torch.tensor([10, 20, 0, 0, 0, 30, 0],
                                     dtype=torch.float32,
                                     device=device)))
                assert torch.all(
                    torch.eq(
                        dest.int_attr,
                        torch.tensor([1, 2, 0, 0, 0, 3, 0],
                                     dtype=torch.int32,
                                     device=device)))
                _k2.fix_final_labels(dest.arcs, dest.int_attr)
                assert torch.all(
                    torch.eq(
                        dest.int_attr,
                        torch.tensor([1, 2, 0, 0, 0, 3, -1],
                                     dtype=torch.int32,
                                     device=device)))

                assert torch.all(
                    torch.eq(
                        dest.ragged_attr,
                        torch.tensor([1, 5, 2, 3, 6, 1, -1],
                                     dtype=torch.float32,
                                     device=device)))

                # non-tensor attributes...
                assert dest.attr1 == src.attr1
                assert dest.attr2 == src.attr2

                # now for autograd
                scale = torch.tensor([10, 20, 10, 10, 10, 30, 10],
                                     device=device)
                (dest.float_attr * scale).sum().backward()
                (dest.scores * scale).sum().backward()

                expected_grad = torch.tensor([10, 20, 30],
                                             dtype=torch.float32,
                                             device=device)

                assert torch.all(torch.eq(src.float_attr.grad, expected_grad))
                assert torch.all(torch.eq(src.scores.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
