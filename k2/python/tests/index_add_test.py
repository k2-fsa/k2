#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R index_add_test_py

import unittest

import k2
import torch


class TestIndexAdd(unittest.TestCase):

    def test_contiguous(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            num_elements = torch.randint(10, 1000, (1,)).item()
            src = torch.rand(num_elements, dtype=torch.float32).to(device)

            num_indexes = num_elements * torch.randint(2, 10, (1,)).item()
            index = torch.randint(-1,
                                  num_elements, (num_indexes,),
                                  dtype=torch.int32).to(device)

            value = torch.rand(num_indexes, dtype=torch.float32).to(device)

            saved = src.clone()
            k2.index_add(index, value, src)

            saved = torch.cat([torch.tensor([0]).to(saved), saved])

            saved.index_add_(0, index.to(torch.int64) + 1, value)
            assert torch.allclose(src, saved[1:])

    def test_non_contiguous(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            num_elements = torch.randint(100, 10000, (1,)).item()
            src = torch.rand(num_elements, dtype=torch.float32).to(device)
            src_stride = torch.randint(2, 8, (1,)).item()
            src = src[::src_stride]

            num_elements = src.numel()
            num_indexes = num_elements * torch.randint(2, 10, (1,)).item()
            index = torch.randint(0,
                                  num_elements, (num_indexes,),
                                  dtype=torch.int32).to(device)

            value_stride = torch.randint(2, 6, (1,)).item()
            value = torch.rand(num_indexes * value_stride,
                               dtype=torch.float32).to(device)

            value = value[::value_stride]

            assert src.is_contiguous() is False
            assert index.is_contiguous()
            assert value.is_contiguous() is False

            saved = src.clone()
            k2.index_add(index, value, src)

            saved = torch.cat([torch.tensor([0]).to(saved), saved])

            saved.index_add_(0, index.to(torch.int64) + 1, value)
            assert torch.allclose(src, saved[1:])


if __name__ == '__main__':
    unittest.main()
