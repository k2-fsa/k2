#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R create_sparse_test_py

import unittest

import k2
import torch


class TestCreaseSparse(unittest.TestCase):

    def test_create_sparse(self):
        s = '''
            0 1 10 0.1
            0 1 11 0.2
            1 2 20 0.3
            2 3 21 0.4
            2 3 24 0.5
            3 4 -1 0.6
            4
        '''

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.phones = torch.tensor([10, 11, 20, 21, 24, -1],
                                      dtype=torch.int32,
                                      device=device)
            fsa.seqframes = torch.tensor([0, 0, 1, 2, 2, 3],
                                         dtype=torch.int32,
                                         device=device)
            fsa.requires_grad_(True)

            tensor = k2.create_sparse(rows=fsa.seqframes,
                                      cols=fsa.phones,
                                      values=fsa.scores,
                                      size=(6, 25),
                                      min_col_index=0)
            assert tensor.device == device
            assert tensor.is_sparse
            assert torch.allclose(tensor._indices()[0],
                                  fsa.seqframes[:-1].to(torch.int64))
            assert torch.allclose(tensor._indices()[1],
                                  fsa.phones[:-1].to(torch.int64))
            assert torch.allclose(tensor._values(), fsa.scores[:-1])
            assert tensor.requires_grad == fsa.requires_grad
            assert tensor.dtype == fsa.scores.dtype


if __name__ == '__main__':
    unittest.main()
