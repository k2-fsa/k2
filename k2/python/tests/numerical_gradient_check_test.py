#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R numerical_gradient_check_test_py

import unittest

import k2
import torch


class TestNumericalGradientCheck(unittest.TestCase):

    def test_get_tot_scores(self):

        def my_func(scores: torch.Tensor,
                    switch: torch.Tensor) -> torch.Tensor:
            s = '''
                0 4 1 0
                0 1 1 0
                1 2 1 0
                1 3 1 0
                2 7 1 0
                3 7 1 0
                4 6 1 0
                4 8 1 0
                5 9 -1 0
                6 9 -1 0
                7 9 -1 0
                8 9 -1 0
                9
            '''
            fsa = k2.Fsa.from_str(s).to(scores.device)
            fsa_vec = k2.create_fsa_vec([fsa])
            assert scores.requires_grad is True
            fsa_vec.scores = scores.to(torch.float32)
            log_semiring = switch[0].item() == 1
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=log_semiring,
                                         use_double_scores=True)
            return -2 * log_like

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            s = torch.randn(12)
            s.requires_grad_(True)
            s = s.to(torch.float64)
            torch.autograd.gradcheck(my_func, (s, torch.tensor([1])), eps=1e-2)
            torch.autograd.gradcheck(my_func, (s, torch.tensor([0])), eps=1e-2)

    def test_get_tot_scores_multiple_fsas(self):

        def my_func(scores: torch.Tensor,
                    switch: torch.Tensor) -> torch.Tensor:
            s1 = '''
                0 4 1 0
                0 1 1 0
                1 2 1 0
                1 3 1 0
                2 7 1 0
                3 7 1 0
                4 6 1 0
                4 8 1 0
                5 9 -1 0
                6 9 -1 0
                7 9 -1 0
                8 9 -1 0
                9
            '''

            s2 = '''
                0 1 1 0
                0 2 2 0
                1 2 3 0
                1 3 4 0
                2 3 5 0
                3 4 6 0
                3 5 -1 0
                4 5 -1 0
                5
            '''

            s3 = '''
                0 1 1 0
                0 2 2 0
                1 3 -1 0
                2 3 -1 0
                3
            '''

            fsa1 = k2.Fsa.from_str(s1).to(scores.device)
            fsa2 = k2.Fsa.from_str(s2).to(scores.device)
            fsa3 = k2.Fsa.from_str(s3).to(scores.device)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2, fsa3])

            assert scores.requires_grad is True
            fsa_vec.scores = scores.to(torch.float32)
            log_semiring = switch[0].item() == 1
            log_like = k2.get_tot_scores(fsa_vec,
                                         log_semiring=log_semiring,
                                         use_double_scores=True)
            return -1.25 * log_like

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            s = torch.randn(24)
            s.requires_grad_(True)
            s = s.to(torch.float64)
            torch.autograd.gradcheck(my_func, (s, torch.tensor([1])), eps=1e-2)
            torch.autograd.gradcheck(my_func, (s, torch.tensor([0])), eps=1e-2)

    def test_index_add_contiguous(self):

        def my_func(index: torch.Tensor, value: torch.Tensor,
                    src: torch.Tensor) -> torch.Tensor:
            saved = torch.zeros_like(src).to(torch.float32)
            k2.index_add(index, value, saved)
            return src + saved

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            num_elements = torch.randint(10, 100, (1,)).item()
            src = torch.rand(num_elements, dtype=torch.float32).to(device)
            src.requires_grad_(True)
            src = src.to(torch.float64)

            num_indexes = num_elements * torch.randint(2, 10, (1,)).item()
            index = torch.randint(-1,
                                  num_elements, (num_indexes,),
                                  dtype=torch.int32).to(device)

            value = torch.rand(num_indexes, dtype=torch.float32).to(device)
            torch.autograd.gradcheck(my_func, (index, value, src),
                                     eps=1e-3,
                                     atol=1e-3,
                                     rtol=1e-3)

    def test_index_add_non_contiguous(self):

        def my_func(index: torch.Tensor, value: torch.Tensor,
                    src: torch.Tensor) -> torch.Tensor:
            saved = torch.zeros_like(src).to(torch.float32)
            k2.index_add(index, value, saved)
            return src + saved

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            num_elements = torch.randint(100, 10000, (1,)).item()
            src = torch.rand(num_elements, dtype=torch.float32).to(device)
            src.requires_grad_(True)
            src = src.to(torch.float64)
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

            torch.autograd.gradcheck(my_func, (index, value, src),
                                     eps=1e-2,
                                     atol=1e-2,
                                     rtol=1e-2)

    def test_index_select_1d(self):

        def my_func(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return k2.index_select(src.to(torch.float32), index)

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            num_elements = torch.randint(10, 100, (1,)).item()
            src = torch.rand(num_elements, dtype=torch.float32).to(device)
            src.requires_grad_(True)
            src = src.to(torch.float64)

            num_indexes = num_elements * torch.randint(2, 10, (1,)).item()
            index = torch.randint(-1,
                                  num_elements, (num_indexes,),
                                  dtype=torch.int32).to(device)
            torch.autograd.gradcheck(my_func, (src, index),
                                     eps=1e-3,
                                     atol=1e-3,
                                     rtol=1e-3)

    def test_index_select_2d(self):

        def my_func(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return k2.index_select(src.to(torch.float32), index)

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            num_rows = torch.randint(1, 20, size=(1,)).item()
            num_cols = torch.randint(1, 20, size=(1,)).item()
            a = torch.rand((num_rows, num_cols),
                           dtype=torch.float64,
                           device=device).contiguous()

            a.requires_grad_(True)
            b = torch.randint(-1,
                              num_rows,
                              size=(100,),
                              device=device,
                              dtype=torch.int32)
            assert a.is_contiguous()
            torch.autograd.gradcheck(my_func, (a, b),
                                     eps=1e-3,
                                     atol=1e-3,
                                     rtol=1e-3)


if __name__ == '__main__':
    torch.manual_seed(20210109)
    unittest.main()
