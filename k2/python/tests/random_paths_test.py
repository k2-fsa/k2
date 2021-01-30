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
                # iter 0, p is 0.5, select the second leaving arc of state 0
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
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            for use_double_scores in (True, False):
                s1 = '''
                    0 1 1 1
                    0 1 2 1
                    1 2 3 1
                    1 2 4 1
                    2 3 -1 1
                    2 3 -1 1
                    3
                '''

                s2 = '''
                    0 1 1 1
                    0 1 2 1
                    0 1 3 1
                    0 1 4 1
                    1 2 1 1
                    1 2 2 1
                    1 2 3 1
                    2 3 1 1
                    2 3 2 1
                    3 4 -1 1
                    4
                '''

                fsa1 = k2.Fsa.from_str(s1)
                fsa2 = k2.Fsa.from_str(s2)
                fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
                path = k2.random_paths(fsa_vec,
                                       use_double_scores=True,
                                       num_paths=1)
                # fsa 0
                #  iter 0, p is 0.5, select arc 1
                #  iter 1, p is 0, select arc 2
                #  iter 2, p is 0, select arc 4
                #  path 0 is [1 2 4]
                # fsa 1
                #  iter 0, p is 0.5, select arc 2
                #  iter 1, p is 0, select arc 4
                #  iter 2, p is 0, select arc 7
                #  iter 3, p is 0, select arc 9
                #  path 1 is [2, 4, 7, 9] + 6 = [8, 10, 13, 15]
                self.assertEqual(str(path),
                                 '[ [ [ 1 2 4 ] ] [ [ 8 10 13 15 ] ] ]')

                path = k2.random_paths(fsa_vec,
                                       use_double_scores=True,
                                       num_paths=2)
                # fsa 1
                #  path 0:
                #    iter 0, p is 0.25, select arc 1
                #    iter 1, p is 0, select arc 4
                #    iter 2, p is 0, select arc 7
                #    iter 3, p is 0, select arc 9
                #    path 0 is [1, 4, 7, 9] + 6 = [7, 10, 13, 15]
                #  path 1:
                #    iter 0, p is 0.75, select arc 3
                #    iter 1, p is 0, select arc 4
                #    iter 2, p is 0, select arc 7
                #    iter 3, p is 0, select arc 9
                #    path 1 is [3, 4, 7, 9] + 6 = [9, 10, 13, 15]
                self.assertEqual(
                    str(path),
                    '[ [ [ 0 3 4 ] [ 1 3 4 ] ] [ [ 7 10 13 15 ] [ 9 10 13 15 ] ] ]'  # noqa
                )
                path = k2.random_paths(fsa_vec,
                                       use_double_scores=True,
                                       num_paths=4)
                # fsa 1
                #  path 0
                #    iter 0, p is 0.125, select arc 0
                #    iter 1, p is 0.125/0.25=0.5, select arc 5
                #    iter 2, p is (0.5 - 0.3333)/(0.6667-0.3333)=0.4995,
                #            select arc 7
                #            (note p is 0.5 in theory, but there are round-off
                #             errors)
                #    iter 3, p is 0.99911, select arc 9
                #    path 0 is [0, 5, 7, 9] + 6 = [6, 11, 13, 15]
                #  path 1
                #    iter 0, p is 0.375, select arc 1
                #    iter 1, p is (0.375 - 0.25)/0.25 = 0.5, select arc 5
                #    iter 2, p is (0.5 - 0.3333)/(0.6667-0.3333)=0.4995,
                #            select arc 7
                #            (note p is 0.5 in theory, but there are round-off
                #             errors)
                #    iter 3, p is 0.99911, select arc 9
                #    path 1 is [1, 5, 7, 9] + 6 = [7, 11, 13, 15]
                #  path 2
                #    iter 0, p is 0.625, select arc 2
                #    iter 1, p is (0.625 - 0.5)/0.25 = 0.5, select arc 5
                #    iter 2, p is (0.5 - 0.3333)/(0.6667-0.3333)=0.4995,
                #            select arc 7
                #            (note p is 0.5 in theory, but there are round-off
                #             errors)
                #    iter 3, p is 0.99911, select arc 9
                #    path 2 is [2, 5, 7, 9] + 6 = [8, 11, 13, 15]
                #  path 3
                #    iter 0, p is 0.875, select arc 3
                #    iter 1, p is (0.875 - 0.75)/0.25 = 0.5, select arc 5
                #    iter 2, p is (0.5 - 0.3333)/(0.6667-0.3333)=0.4995,
                #            select arc 7
                #            (note p is 0.5 in theory, but there are round-off
                #             errors)
                #    iter 3, p is 0.99911, select arc 9
                #    path 3 is [3, 5, 7, 9] + 6 = [9, 11, 13, 15]
                self.assertEqual(
                    str(path),
                    '[ [ [ 0 2 5 ] [ 0 3 5 ] [ 1 2 5 ] [ 1 3 5 ] ] [ [ 6 11 13 15 ] [ 7 11 13 15 ] [ 8 11 13 15 ] [ 9 11 13 15 ] ] ]'  # noqa
                )


if __name__ == '__main__':
    unittest.main()
