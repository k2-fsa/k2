#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R array_test_py -E host

import unittest

import k2
import torch

import _k2  # for test only, users should not import it.


class TestArray(unittest.TestCase):

    def test_cpu_int_array1_to_tensor(self):
        _arr = _k2.get_cpu_int_array1()
        arr = k2.Array(_arr)

        tensor = arr.tensor()
        assert tensor.ndim == 1
        assert tensor.dtype == torch.int32
        assert tensor.device.type == 'cpu'
        assert tensor[0] == _arr.get(0)

        # now we change the tensor, `_arr` should also be changed
        # since they share the underlying memory

        tensor[0] += 100
        assert tensor[0] == _arr.get(0)

        val = tensor[0]

        del _arr, arr
        assert tensor[0] == val, 'tensor should still be accessible'
        del tensor

    def test_cpu_float_array1_from_tensor(self):
        gt_tensor = torch.tensor([1, 2, 3], dtype=torch.float)
        array = k2.Array(gt_tensor)
        actual_tensor = array.tensor()

        assert actual_tensor.dtype == gt_tensor.dtype
        assert actual_tensor.device == gt_tensor.device
        assert actual_tensor.ndim == gt_tensor.ndim

        assert torch.allclose(gt_tensor, actual_tensor)

        gt_tensor += 100
        assert torch.allclose(gt_tensor, actual_tensor), \
                'actual_tensor should share the same memory with gt_tensor'

        val = gt_tensor[0]
        del gt_tensor, array

        actual_tensor[0] += 1
        val += 1
        assert val == actual_tensor[0], \
                'actual_tensor[0] should still be accessible'
        del actual_tensor

    def test_cuda_float_array1_to_tensor(self):
        device_id = 0
        _arr = _k2.get_cuda_float_array1(device_id)
        arr = k2.Array(_arr)

        tensor = arr.tensor()
        assert tensor.ndim == 1
        assert tensor.dtype == torch.float
        assert tensor.device.type == 'cuda'
        assert tensor.device.index == device_id
        assert tensor[0] == _arr.get(0)

        # now we change the tensor, `_arr` should also be changed
        # since they share the underlying memory

        tensor[0] += 100
        assert tensor[0] == _arr.get(0)

        val = tensor[0]

        del _arr, arr
        tensor[0] += 1
        val += 1
        assert tensor[0] == val, 'tensor should still be accessible'
        del tensor

    def test_cuda_int_array1_from_tensor(self):
        device_id = 0
        device = torch.device('cuda', device_id)
        gt_tensor = torch.tensor([1, 2, 3], dtype=torch.int32).to(device)
        array = k2.Array(gt_tensor)
        actual_tensor = array.tensor()

        assert actual_tensor.dtype == gt_tensor.dtype
        assert actual_tensor.device == gt_tensor.device
        assert actual_tensor.ndim == gt_tensor.ndim

        assert torch.allclose(gt_tensor, actual_tensor)

        gt_tensor += 100
        assert torch.allclose(gt_tensor, actual_tensor), \
                'actual_tensor should share the same memory with gt_tensor'

        val = gt_tensor[0]
        del gt_tensor, array

        actual_tensor[0] += 1
        val += 1
        assert val == actual_tensor[0], \
                'actual_tensor[0] should still be accessible'
        del actual_tensor

    def test_cpu_arc_array1_from_tensor(self):
        gt_tensor = torch.tensor(
            [[1, 2, 3, _k2._float_as_int(1.5)],
             [10, 20, 30, _k2._float_as_int(-2.5)]],
            dtype=torch.int32).contiguous()
        arc_array = k2.Array(gt_tensor, True)

        arc0 = arc_array.data.get(0)
        assert arc0.src_state == 1
        assert arc0.dest_state == 2
        assert arc0.symbol == 3
        assert arc0.score == 1.5

        actual_tensor = arc_array.tensor()
        assert torch.allclose(gt_tensor, actual_tensor)
        gt_tensor[0] += 10  # also change actual_tensor

        assert torch.allclose(gt_tensor, actual_tensor)
        assert arc_array.data.get(0).src_state == 11
        del gt_tensor, actual_tensor

        # arc_array is still accessible
        assert arc_array.data.get(0).src_state == 11

    def test_cuda_arc_array1_to_tensor(self):
        device_id = 0
        _arc_array = _k2.get_cuda_arc_array1(device_id)
        arc_array = k2.Array(_arc_array)
        tensor = arc_array.tensor()
        assert tensor.ndim == 2
        assert tensor.shape == (2, 4)
        assert tensor.device.type == 'cuda'
        assert tensor.device.index == device_id
        assert tensor[0][0] == 1
        assert tensor[0][3] == _k2._float_as_int(_arc_array.get(0).score)

        tensor[0][0] = 10  # also change _arc_array
        assert _arc_array.get(0).src_state == 10

        del arc_array, _arc_array
        tensor[0][0] += 10
        assert tensor[0][0] == 20, 'tensor should still be accessible'

    def test_cpu_int_array2_to_tensor(self):
        _arr = _k2.get_cpu_int_array2()
        arr = k2.Array(_arr)
        tensor = arr.tensor()
        del _arr, arr
        assert torch.allclose(
            tensor, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))

    def test_cpu_float_array2_from_tensor(self):
        gt_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        arr = k2.Array(gt_tensor)
        actual_tensor = arr.tensor()

        actual_tensor[0, 0] = 1000
        assert torch.allclose(gt_tensor, actual_tensor)

        del gt_tensor, arr
        assert actual_tensor[0, 0] == 1000

    def test_cuda_float_array2_to_tensor(self):
        _arr = _k2.get_cuda_float_array2()
        arr = k2.Array(_arr)
        tensor = arr.tensor()
        assert torch.allclose(
            tensor,
            torch.tensor([[1, 2, 3], [4, 5, 6]],
                         dtype=torch.float,
                         device='cuda'))
        tensor[0, 0] = 100
        assert _arr.get(0).get(0) == 100

    def test_cuda_int32_array2_to_tensor(self):
        gt_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                                 dtype=torch.int32,
                                 device='cuda')

        arr = k2.Array(gt_tensor)
        actual_tensor = arr.tensor()

        assert torch.allclose(gt_tensor, actual_tensor)
        gt_tensor[0, 0] = 100
        del gt_tensor
        assert actual_tensor[0, 0] == 100


if __name__ == '__main__':
    unittest.main()
