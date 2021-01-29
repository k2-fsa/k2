#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R random_paths_test_py

import unittest

import k2.sparse
import torch


class TestRandomPaths(unittest.TestCase):

    def test_single_fsa_case1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            for use_double_scores in (True, False):
                s = '''
                    0 1 1 0.1
                    1 2 -1 0.2
                    2
                '''
                fsa = k2.Fsa.from_str(s).to(device)
                fsa_vec = k2.create_fsa_vec([fsa])
                path = k2.random_paths(fsa_vec,
                                       use_double_scores=use_double_scores,
                                       num_paths=2)
                assert path.num_axes() == 3
                self.assertEqual(str(path), '[ [ [ 0 1 ] [ 0 1 ] ] ]')

    def test_single_fsa_case2(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            for use_double_scores in (True,):
                s = '''
                    0 1 1 1
                    0 1 2 1
                    1 2 3 1
                    1 2 4 1
                    2 3 -1 1
                    2 3 -1 1
                    3
                '''
                fsa = k2.Fsa.from_str(s).to(device)
                fsa_vec = k2.create_fsa_vec([fsa])

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=use_double_scores,
                                       num_paths=1)
                assert path.num_axes() == 3
                # iter 0, p is 0.5, select the last leaving arc of state 0
                # iter 1, p is 0, select the first leaving arc of state 1
                # iter 2, p is 0, select the first leaving arc of state 2
                self.assertEqual(str(path), '[ [ [ 1 2 4 ] ] ]')

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=use_double_scores,
                                       num_paths=2)
                # path 0
                #  iter 0, p is 0.25, select the first leaving arc of state 0
                #  iter 1, p is 0.25/0.5 = 0.5, select the second leaving arc
                #          of state 1
                #  iter 2, p is (0.5 - 0.5) / (1 - 0.5) = 0, select the first
                #          leaving arc of state 2
                # path 1
                #  iter 0, p is 0.75, select the second leaving arc of state 0
                #  iter 1, p is (0.75 - 0.5) / (1 - 0.5) = 0.5, select the
                #          second leaving arc of state 1
                #  iter 2, p is (0.5 - 0.5) / (1 - 0.5) = 0, select the
                #          first leaving arc of state 1
                assert path.num_axes() == 3
                self.assertEqual(str(path), '[ [ [ 0 3 4 ] [ 1 3 4 ] ] ]')

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=True,
                                       num_paths=4)
                # path 0
                #  iter 0, p is 0.125, select the first leaving arc of state 0
                #  iter 1, p is 0.125/0.5=0.25, select the first leaving arc of
                #          state 1
                #  iter 2, p is 0.25/0.5=0.5, select the second leaving arc of
                #          state 2
                # path 1
                #  iter 0, p is 0.375, select the first leaving arc of state 0
                #  iter 1, p is 0.375/0.5=0.75, select the second leaving arc
                #          of state 1
                #  iter 2, p is 0.25/0.5=0.5, select the second leaving arc
                #          of state 2
                # path 2
                #  iter 0, p is 0.625, select the second leaving arc of state 0
                #  iter 1, p is 0.125/0.5=0.25, select the first leaving arc of
                #          state 1
                #  iter 2, p is 0.25/0.5=0.5, select the second leaving arc of
                #          state 2
                # path 3
                #  iter 0, p is 0.875, select the second leaving arc of state 0
                #  iter 1, p is 0.375/0.5=0.75, select the second leaving arc of
                #          state 1
                #  iter 2, p is 0.25/0.5=0.5, select the second leaving arc of
                #          state 2
                assert path.num_axes() == 3
                self.assertEqual(
                    str(path),
                    '[ [ [ 0 2 5 ] [ 0 3 5 ] [ 1 2 5 ] [ 1 3 5 ] ] ]')

    def test_fsa_vec(self):
        s1 = '''
            0 1 1 1
            0 1 2 1
            0 1 3 1
            0 1 4 1
            1 2 -1 1
            2
        '''

        s2 = '''
            0 1 1 1
            0 1 2 1
            0 1 3 1
            1 2 -1 1
            2
        '''
        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)
        fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
        path = k2.random_paths(fsa_vec, use_double_scores=True, num_paths=1)
        # p is 0.5
        #  Select the third leaving arc of state 0 for fsa1
        #  Select the second leaving arc of state 0 for fsa2
        assert path.num_axes() == 3
        self.assertEqual(str(path), '[ [ [ 2 4 ] ] [ [ 6 8 ] ] ]')

        path = k2.random_paths(fsa_vec, use_double_scores=True, num_paths=2)
        # p is [0.25, 0.75]
        # The first path (p == 0.25):
        #  Select the second leaving arc of state 0 for fsa1
        #  Select the first leaving arc of state 0 for fsa2
        # The second path (p == 0.75)
        #  Select the last leaving arc of state 0 for fsa1
        #  Select the last leaving arc of state 0 for fsa2
        assert path.num_axes() == 3
        self.assertEqual(str(path),
                         '[ [ [ 1 4 ] [ 3 4 ] ] [ [ 5 8 ] [ 7 8 ] ] ]')


if __name__ == '__main__':
    unittest.main()
