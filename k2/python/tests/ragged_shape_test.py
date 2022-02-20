#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Liyong Guo)
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
#  ctest --verbose -R  ragged_shape_test_py

import unittest

import k2
import torch


class TestRaggedShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_ragged_shape(self):
        # test case reference:
        # https://github.com/k2-fsa/k2/blob/f79ce20ce2deeb8f4ed82a0ea028da34cb26e40e/k2/csrc/ragged_shape_test.cu#L60
        src = '''
         [
           [ [[ x x] [x]]  [[x x]] ]
           [ [[x x x]] [[x] [x x x]] [[x]] ]
           [ [[x x] [] [x]] ]
         ]
        '''
        for device in self.devices:
            shape = k2.RaggedShape(src)
            shape = shape.to(device)
            assert shape.num_axes == 4
            assert shape.dim0 == 3
            assert shape.tot_size(0) == 3
            assert shape.tot_size(1) == 6
            assert shape.tot_size(2) == 10
            assert shape.tot_size(3) == 16
            assert shape.numel() == shape.tot_size(3)

            assert shape.max_size(1) == 3
            assert shape.max_size(2) == 3
            assert shape.max_size(3) == 3

            assert torch.allclose(
                shape.row_splits(1),
                torch.tensor([0, 2, 5, 6], dtype=torch.int32).to(device))
            assert torch.allclose(
                shape.row_splits(2),
                torch.tensor([0, 2, 3, 4, 6, 7, 10],
                             dtype=torch.int32).to(device))
            assert torch.allclose(
                shape.row_splits(3),
                torch.tensor([0, 2, 3, 5, 8, 9, 12, 13, 15, 15, 16],
                             dtype=torch.int32).to(device))

            assert torch.allclose(
                shape.row_ids(1),
                torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32).to(device))
            assert torch.allclose(
                shape.row_ids(2),
                torch.tensor([0, 0, 1, 2, 3, 3, 4, 5, 5, 5],
                             dtype=torch.int32).to(device))
            assert torch.allclose(
                shape.row_ids(3),
                torch.tensor([0, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 9],
                             dtype=torch.int32).to(device))

    def test_random_ragged_shape(self):
        # test case reference:
        # https://github.com/k2-fsa/k2/blob/master/k2/csrc/ragged_shape_test.cu#L304
        shape = k2.ragged.random_ragged_shape(False, 2, 4, 0, 0)
        assert shape.num_axes >= 2
        assert shape.numel() == 0

        shape = k2.ragged.random_ragged_shape()
        assert shape.num_axes >= 2
        assert shape.numel() >= 0

        shape = k2.ragged.random_ragged_shape(False, 3, 5, 100)
        assert shape.num_axes >= 3
        assert shape.numel() >= 100

        shape = k2.ragged.random_ragged_shape(True, 3, 5, 100)
        assert shape.num_axes >= 3
        assert shape.numel() >= 100

    def test_compose_ragged_shape(self):
        for device in self.devices:
            a = k2.RaggedTensor('[ [ 0 ] [ 1 2 ] ]').to(device)
            b = k2.RaggedTensor('[ [ 3 ] [ 4 5 ] [ 6 7 ] ]').to(device)
            prod = k2.RaggedTensor('[ [ [ 3 ] ] [ [ 4 5 ] [ 6 7 ] ] ]').to(
                device)
            ashape = a.shape
            bshape = b.shape
            abshape = ashape.compose(bshape)
            # should also be available under k2.ragged.
            abshape2 = ashape.compose(bshape)
            assert abshape == prod.shape
            assert abshape2 == prod.shape
            prod2 = k2.RaggedTensor(abshape2, b.values)
            assert prod == prod2

    def test_create_ragged_shape2_with_row_splits(self):
        for device in self.devices:
            row_splits = torch.tensor([0, 1, 3],
                                      dtype=torch.int32,
                                      device=device)
            shape = k2.ragged.create_ragged_shape2(row_splits=row_splits)
            expected_shape = k2.RaggedShape('[[x] [x x]]').to(device)
            assert shape == expected_shape

    def test_create_ragged_shape2_with_row_ids(self):
        for device in self.devices:
            row_ids = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
            shape = k2.ragged.create_ragged_shape2(row_ids=row_ids)
            expected_shape = k2.RaggedShape('[[x] [x x]]').to(device)
            assert shape == expected_shape


if __name__ == '__main__':
    unittest.main()
