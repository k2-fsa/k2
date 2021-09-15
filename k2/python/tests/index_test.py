#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
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
#  ctest --verbose -R index_test_py

import unittest

import k2
import torch


class TestIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        s0 = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            2 3 -1 0.4
            3
        '''
        s1 = '''
            0 1 -1 0.5
            1
        '''
        s2 = '''
            0 2 1 0.6
            0 1 2 0.7
            1 3 -1 0.8
            2 1 3 0.9
            3
        '''
        for device in self.devices:
            fsa0 = k2.Fsa.from_str(s0).to(device).requires_grad_(True)
            fsa1 = k2.Fsa.from_str(s1).to(device).requires_grad_(True)
            fsa2 = k2.Fsa.from_str(s2).to(device).requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa0, fsa1, fsa2])

            new_fsa21 = k2.index_fsa(
                fsa_vec, torch.tensor([2, 1], dtype=torch.int32,
                                      device=device))
            assert new_fsa21.shape == (2, None, None)
            assert torch.all(
                torch.eq(
                    new_fsa21.arcs.values()[:, :3],
                    torch.tensor([
                        # fsa 2
                        [0, 2, 1],
                        [0, 1, 2],
                        [1, 3, -1],
                        [2, 1, 3],
                        # fsa 1
                        [0, 1, -1]
                    ]).to(torch.int32).to(device)))

            scale = torch.arange(new_fsa21.scores.numel(), device=device)
            (new_fsa21.scores * scale).sum().backward()
            assert torch.allclose(fsa0.scores.grad,
                                  torch.tensor([0., 0, 0, 0], device=device))
            assert torch.allclose(fsa1.scores.grad,
                                  torch.tensor([4.], device=device))
            assert torch.allclose(
                fsa2.scores.grad, torch.tensor([0., 1., 2., 3.],
                                               device=device))

            # now select only a single FSA
            fsa0.scores.grad = None
            fsa1.scores.grad = None
            fsa2.scores.grad = None

            new_fsa0 = k2.index_fsa(
                fsa_vec, torch.tensor([0], dtype=torch.int32, device=device))
            assert new_fsa0.shape == (1, None, None)

            scale = torch.arange(new_fsa0.scores.numel(), device=device)
            (new_fsa0.scores * scale).sum().backward()
            assert torch.allclose(
                fsa0.scores.grad, torch.tensor([0., 1., 2., 3.],
                                               device=device))
            assert torch.allclose(fsa1.scores.grad,
                                  torch.tensor([0.], device=device))
            assert torch.allclose(
                fsa2.scores.grad, torch.tensor([0., 0., 0., 0.],
                                               device=device))


class TestIndexRaggedTensor(unittest.TestCase):

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
            src_row_splits = torch.tensor([0, 2, 3, 3, 6],
                                          dtype=torch.int32,
                                          device=device)
            src_shape = k2.ragged.create_ragged_shape2(src_row_splits, None, 6)
            src_values = torch.tensor([1, 2, 3, 4, 5, 6],
                                      dtype=torch.int32,
                                      device=device)
            src = k2.RaggedTensor(src_shape, src_values)

            # index with ragged int
            index_row_splits = torch.tensor([0, 2, 2, 3, 7],
                                            dtype=torch.int32,
                                            device=device)
            index_shape = k2.ragged.create_ragged_shape2(
                index_row_splits, None, 7)
            index_values = torch.tensor([0, 3, 2, 1, 2, 1, 0],
                                        dtype=torch.int32,
                                        device=device)
            ragged_index = k2.RaggedTensor(index_shape, index_values)
            ans = src.index(ragged_index)
            ans = ans.remove_axis(1)
            expected_row_splits = torch.tensor([0, 5, 5, 5, 9],
                                               dtype=torch.int32,
                                               device=device)
            self.assertTrue(
                torch.allclose(ans.shape.row_splits(1), expected_row_splits))
            expected_values = torch.tensor([1, 2, 4, 5, 6, 3, 3, 1, 2],
                                           dtype=torch.int32,
                                           device=device)
            self.assertTrue(torch.allclose(ans.values, expected_values))

            # index with tensor
            tensor_index = torch.tensor([0, 3, 2, 1, 2, 1],
                                        dtype=torch.int32,
                                        device=device)
            ans, _ = src.index(tensor_index, axis=0, need_value_indexes=False)
            expected_row_splits = torch.tensor([0, 2, 5, 5, 6, 6, 7],
                                               dtype=torch.int32,
                                               device=device)
            self.assertTrue(
                torch.allclose(ans.shape.row_splits(1), expected_row_splits))
            expected_values = torch.tensor([1, 2, 4, 5, 6, 3, 3],
                                           dtype=torch.int32,
                                           device=device)
            self.assertTrue(torch.allclose(ans.values, expected_values))


class TestIndexTensorWithRaggedInt(unittest.TestCase):

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
            src = torch.tensor([1, 2, 3, 4, 5, 6, 7],
                               dtype=torch.int32,
                               device=device)
            index_row_splits = torch.tensor([0, 2, 2, 3, 7],
                                            dtype=torch.int32,
                                            device=device)
            index_shape = k2.ragged.create_ragged_shape2(
                index_row_splits, None, 7)
            index_values = torch.tensor([0, 3, 2, 3, 5, 1, 3],
                                        dtype=torch.int32,
                                        device=device)
            ragged_index = k2.RaggedTensor(index_shape, index_values)

            ans = k2.ragged.index(src, ragged_index)
            self.assertTrue(
                torch.allclose(ans.shape.row_splits(1), index_row_splits))
            expected_values = torch.tensor([1, 4, 3, 4, 6, 2, 4],
                                           dtype=torch.int32,
                                           device=device)
            self.assertTrue(torch.allclose(ans.values, expected_values))


if __name__ == '__main__':
    unittest.main()
