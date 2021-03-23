#!/usr/bin/env python3
#
# Copyright (c)  2020-2021  Xiaomi Corporation (authors: Haowen Qiu
#                                                        Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R invert_test_py

import unittest

import torch
import k2


class TestInvert(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))

    def test_aux_as_tensor(self):
        s = '''
            0 1 1 1 0
            0 1 0 2 0
            0 3 2 3 0
            1 2 3 4 0
            1 3 4 5 0
            2 1 5 6 0
            2 5 -1 -1 0
            3 1 6 7 0
            4 5 -1 -1 0
            5
        '''
        fsa = k2.Fsa.from_str(s, num_aux_labels=1)
        assert fsa.device.type == 'cpu'
        dest = k2.invert(fsa)
        print(dest)

    def test_aux_as_ragged(self):
        s = '''
            0 1 1 0
            0 1 0 0
            0 3 2 0
            1 2 3 0
            1 3 4 0
            2 1 5 0
            2 5 -1 0
            3 1 6 0
            4 5 -1 0
            5
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.device.type == 'cpu'
        aux_row_splits = torch.tensor([0, 2, 3, 3, 6, 6, 7, 8, 10, 11],
                                      dtype=torch.int32)
        aux_shape = k2.ragged.create_ragged_shape2(aux_row_splits, None, 11)
        aux_values = torch.tensor([1, 2, 3, 5, 6, 7, 8, -1, 9, 10, -1],
                                  dtype=torch.int32)
        fsa.aux_labels = k2.RaggedInt(aux_shape, aux_values)
        dest = k2.invert(fsa)
        print(dest)  # will print aux_labels as well
        # TODO(haowen): wrap C++ code to check equality for Ragged?

    def test_aux_ragged(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                0 2 2 0.2
                1 3 3 0.3
                2 3 4 0.6
                3 4 -1 0.7
                4
            '''
            # https://git.io/JqNiR
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.aux_labels = k2.RaggedInt('[[2 3] [3 4] [] [5] [-1]]').to(
                device)
            fsa.tensor_attr1 = torch.tensor([1, 2, 3, 4, 5]).to(device)

            # https://git.io/JqNiw
            ans = k2.invert(fsa)

            assert torch.all(
                torch.eq(ans.tensor_attr1,
                         torch.tensor([1, 2, 0, 0, 3, 4, 5], device=device)))

            assert str(ans.aux_labels) == str(
                k2.RaggedInt('[[1] [2] [] [] [3] [4] [-1]]'))

            assert torch.all(
                torch.eq(ans.labels,
                         torch.tensor([2, 3, 3, 4, 0, 5, -1], device=device)))


if __name__ == '__main__':
    unittest.main()
