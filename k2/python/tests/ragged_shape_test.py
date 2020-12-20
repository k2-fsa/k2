#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Liyong Guo)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  ragged_shape_test_py

import unittest

import k2
import torch


class TestRaggedShape(unittest.TestCase):

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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            shape = k2.RaggedShape(src)
            shape = shape.to(device)
            assert shape.num_axes() == 4
            assert shape.dim0() == 3
            assert shape.tot_size(0) == 3
            assert shape.tot_size(1) == 6
            assert shape.tot_size(2) == 10
            assert shape.tot_size(3) == 16
            assert shape.num_elements() == shape.tot_size(3)

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
        assert shape.num_axes() >= 2
        assert shape.num_elements() == 0

        shape = k2.ragged.random_ragged_shape()
        assert shape.num_axes() >= 2
        assert shape.num_elements() >= 0

        shape = k2.ragged.random_ragged_shape(False, 3, 5, 100)
        assert shape.num_axes() >= 3
        assert shape.num_elements() >= 100

        shape = k2.ragged.random_ragged_shape(True, 3, 5, 100)
        assert shape.num_axes() >= 3
        assert shape.num_elements() >= 100

    def test_compose_ragged_shape(self):
        a = k2.RaggedInt('[ [ 0 ] [ 1 2 ] ]')
        b = k2.RaggedInt('[ [ 3 ] [ 4 5 ] [ 6 7 ] ]')
        prod = k2.RaggedInt('[ [ [ 3 ] ] [ [ 4 5 ] [ 6 7 ] ] ]')
        ashape = a.shape()
        bshape = b.shape()
        abshape = k2.ragged.compose_ragged_shapes(ashape, bshape)
        # should also be available under k2.ragged.
        abshape2 = k2.ragged.compose_ragged_shapes(ashape, bshape)
        self.assertEqual(str(abshape), str(prod.shape()))
        self.assertEqual(str(abshape2), str(prod.shape()))
        prod2 = k2.RaggedInt(abshape2, b.values())
        self.assertEqual(str(prod), str(prod2))


if __name__ == '__main__':
    unittest.main()
