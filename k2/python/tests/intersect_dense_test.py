#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corp.       (authors: Daniel Povey, Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R intersect_dense_test_py

import unittest

import k2
import torch


class TestIntersectDense(unittest.TestCase):

    def test_simple(self):
        s = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        fsa_vec = k2.create_fsa_vec([fsa])
        log_prob = torch.tensor([[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06]]],
                                dtype=torch.float32,
                                requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 2]], dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense(fsa_vec,
                                     dense_fsa_vec,
                                     output_beam=100000)
        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_double_scores=False)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        expected_scores_out_fsa = torch.tensor([1.2, 2.06, 3.0])
        expected_grad_fsa = torch.tensor([1.0, 0.0, 1.0, 1.0])
        expected_grad_log_prob = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0,
                                               1.0]).reshape_as(log_prob)
        assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa, fsa.scores.grad)
        assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_dense(self):
        s = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        fsa_vec = k2.create_fsa_vec([fsa, fsa])
        log_prob = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
             [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 3], [1, 0, 2]],
                                            dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense(fsa_vec,
                                     dense_fsa_vec,
                                     output_beam=100000)
        assert out_fsa.shape == (2, None, None), 'There should be two FSAs!'

        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_double_scores=False)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        #  expected_scores_out_fsa = torch.tensor(
        #      [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0])
        expected_grad_fsa = torch.tensor([2.0, 1.0, 2.0, 2.0])
        #  expected_grad_log_prob = torch.tensor([
        #      0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
        #      0.0, 0.0, 0.0, 1.0
        #  ]).reshape_as(log_prob)

        # TODO(dan):: fix this..
        # assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa, fsa.scores.grad)
        # assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_fsas(self):
        s1 = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        s2 = '''
            0 1 1 1.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)

        fsa1.requires_grad_(True)
        fsa2.requires_grad_(True)

        fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

        log_prob = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
             [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True)

        supervision_segments = torch.tensor([[0, 0, 3], [1, 0, 2]],
                                            dtype=torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
        out_fsa = k2.intersect_dense(fsa_vec,
                                     dense_fsa_vec,
                                     output_beam=100000)
        assert out_fsa.shape == (2, None, None), 'There should be two FSAs!'

        scores = k2.get_tot_scores(out_fsa,
                                   log_semiring=False,
                                   use_double_scores=False)
        scores.sum().backward()

        # `expected` results are computed using gtn.
        # See https://bit.ly/3oYObeb
        #  expected_scores_out_fsa = torch.tensor(
        #      [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0])

        expected_grad_fsa1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        expected_grad_fsa2 = torch.tensor([1.0, 1.0, 1.0])
        print("fsa2 is ", fsa2.__str__())
        # TODO(dan):: fix this..
        #  expected_grad_log_prob = torch.tensor([
        #      0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
        #      0.0, 0.0, 0.0, 1.0
        #  ]).reshape_as(log_prob)

        # assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
        assert torch.allclose(expected_grad_fsa1, fsa1.scores.grad)
        assert torch.allclose(expected_grad_fsa2, fsa2.scores.grad)
        # assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_fsas_long(self):
        # as test_two_fsas, but generate long DenseFsaVec for easier profiling.
        s1 = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        s2 = '''
            0 1 1 1.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))
        for device in devices:
            fsa1 = k2.Fsa.from_str(s1)
            fsa2 = k2.Fsa.from_str(s2)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
            log_prob = torch.rand((2, 100, 3),
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)

            supervision_segments = torch.tensor([[0, 0, 95], [1, 20, 50]],
                                                dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            fsa_vec = fsa_vec.to(device)
            out_fsa = k2.intersect_dense(fsa_vec,
                                         dense_fsa_vec,
                                         output_beam=100000)
            assert out_fsa.shape == (2, None,
                                     None), 'There should be two FSAs!'

            scores = k2.get_tot_scores(out_fsa,
                                       log_semiring=False,
                                       use_double_scores=False)
            scores.sum().backward()


if __name__ == '__main__':
    unittest.main()
