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
            for use_double_scores in (True, False):
                s = '''
                    0 1 1 1
                    0 1 2 1.1
                    1 2 3 1
                    1 2 4 1.1
                    2 3 -1 1
                    2 3 -1 1.1
                    3
                '''
                fsa = k2.Fsa.from_str(s).to(device)
                fsa_vec = k2.create_fsa_vec([fsa])

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=use_double_scores,
                                       num_paths=1)
                assert path.num_axes() == 3
                # p is 0.5, it should select the last leaving arc of each state
                self.assertEqual(str(path), '[ [ [ 1 3 5 ] ] ]')

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=use_double_scores,
                                       num_paths=2)
                # p is [0.25, 0.75], the first path selects the first leaving arc
                # and the second path selects the second leaving arc of each state
                assert path.num_axes() == 3
                self.assertEqual(str(path), '[ [ [ 0 2 4 ] [ 1 3 5 ] ] ]')

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=True,
                                       num_paths=4)
                # p is [0.125, 0.375, 0.6250, 0.8750],
                # the first path and second path select the first leaving arc
                # the third and the fourth select the second leaving arc of each state
                assert path.num_axes() == 3
                self.assertEqual(
                    str(path),
                    '[ [ [ 0 2 4 ] [ 0 2 4 ] [ 1 3 5 ] [ 1 3 5 ] ] ]')

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
