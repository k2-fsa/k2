#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  replace_fsa_test_py

import unittest

import k2
import torch
import _k2


class TestReplaceFsa(unittest.TestCase):

    def test(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            devices.append(torch.device('cuda', 0))

        for device in devices:
            s1 = '''
            0 1 11 11
            0 2 12 12
            0 3 13 13
            1 4 -1 0
            2 4 -1 0
            3 4 -1 0
            4
            '''
            fsa1 = k2.Fsa.from_str(s1)
            s2 = '''
            0 1 21 21
            0 2 22 22
            1 2 23 23
            1 3 -1 0
            2 3 -1 0
            3
            '''
            fsa2 = k2.Fsa.from_str(s2)
            s3 = '''
            0 1 31 31
            1 2 32 32
            1 3 33 33
            2 4 -1 0
            3 4 -1 0
            4
            '''
            fsa3 = k2.Fsa.from_str(s3)
            src = k2.create_fsa_vec([fsa1, fsa2, fsa3]).to(device)

            s0 = '''
            0 1 1 1
            0 2 2 2
            1 3 4 4
            2 3 3 3 
            2 4 -1 0
            3 4 -1 0
            4
            '''
            index = k2.Fsa.from_str(s0).to(device)

            index.aux_label = torch.tensor([1, 2, 3, 4, 5, 6],
                                        dtype=torch.int32,
                                        device=device)

            dest = k2.replace_fsa(src, index, 1) 

            actual_str = k2.to_str_simple(dest)
            expected_str = '\n'.join(
                ['0 1 0 1', '0 5 0 2', '1 2 11 11', '1 3 12 12', '1 4 13 13',
                 '2 8 0 0', '3 8 0 0', '4 8 0 0', '5 6 21 21', '5 7 22 22',
                 '6 7 23 23', '6 9 0 0', '7 9 0 0', '8 14 4 4', '9 10 0 3',
                 '9 15 -1 0', '10 11 31 31', '11 12 32 32', '11 13 33 33',
                 '12 14 0 0', '13 14 0 0', '14 15 -1 0', '15']);

            assert actual_str.strip() == expected_str

            assert torch.all(
                torch.eq(
                    dest.aux_label,
                    torch.tensor([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  3, 4, 5, 0, 0, 0, 0, 0, 6],
                                 dtype=torch.int32,
                                 device=device)))


if __name__ == '__main__':
    unittest.main()
