#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R array_test_py

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


if __name__ == '__main__':
    unittest.main()
