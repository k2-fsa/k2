#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R closure_test_py

import unittest

import k2
import torch


class TestClosure(unittest.TestCase):

    def test_simple_fsa(self):
        s = '''
            0 1 1 0.1
            1 2 2 0.2
            2 3 -1 0.3
            3
        '''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa.aux_labels = torch.tensor([10, 20, -1],
                                          dtype=torch.int32).to(device)
            ans = k2.closure(fsa)

            assert torch.allclose(
                ans.arcs.values()[:, :3],
                torch.tensor([
                    [0, 1, 1],
                    [0, 3, -1],  # this arc is added by closure
                    [1, 2, 2],
                    [2, 0, 0],  # this arc's dest state and label are changed
                ]).to(torch.int32).to(device))
            scale = torch.arange(ans.arcs.values().shape[0]).to(device)
            (ans.scores * scale).sum().backward()
            assert torch.allclose(fsa.grad,
                                  torch.tensor([0., 2., 3.]).to(fsa.grad))

            assert torch.allclose(
                ans.scores,
                torch.tensor([0.1, 0.0, 0.2, 0.3]).to(device))

            assert torch.allclose(
                ans.aux_labels,
                torch.tensor([10, -1, 20, 0]).to(ans.aux_labels))

    def test_complex_fsa(self):
        s = '''
            0 1 1 0.0
            0 2 2 0.1
            0 3 -1 0.2
            1 0 0 0.3
            1 2 2 0.4
            1 3 -1 0.5
            2 0 0 0.6
            2 1 1 0.7
            2 2 2 0.8
            2 3 -1 0.9
            3
        '''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa.aux_labels = torch.tensor([0, 1, -1, 2, 3, -1, 4, 5, 6, -1],
                                          dtype=torch.int32).to(device)
            ans = k2.closure(fsa)
            assert torch.allclose(
                ans.arcs.values()[:, :3],
                torch.tensor([
                    [0, 1, 1],
                    [0, 2, 2],
                    [0, 0, 0],
                    [0, 3, -1],
                    [1, 0, 0],
                    [1, 2, 2],
                    [1, 0, 0],
                    [2, 0, 0],
                    [2, 1, 1],
                    [2, 2, 2],
                    [2, 0, 0],
                ]).to(torch.int32).to(device))
            assert torch.allclose(
                ans.scores,
                torch.tensor(
                    [0., 0.1, 0.2, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9]).to(device))
            scale = torch.arange(ans.arcs.values().shape[0]).to(device)
            (ans.scores * scale).sum().backward()
            assert torch.allclose(
                fsa.grad,
                torch.tensor([0., 1., 2., 4, 5, 6, 7, 8, 9, 10.]).to(fsa.grad))

            assert torch.allclose(
                ans.aux_labels,
                torch.tensor([0, 1, 0, -1, 2, 3, 0, 4, 5, 6,
                              0]).to(ans.aux_labels))


if __name__ == '__main__':
    unittest.main()
