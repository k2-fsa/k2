#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R intersect_dense_pruned_test_py

import unittest

import k2
import torch


class TestIntersectDensePruned(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_simple(self):
        s = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            fsa_vec = k2.create_fsa_vec([fsa])
            log_prob = torch.tensor([[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06]]],
                                    dtype=torch.float32,
                                    device=device,
                                    requires_grad=True)

            supervision_segments = torch.tensor([[0, 0, 2]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                                dense_fsa_vec,
                                                search_beam=100000,
                                                output_beam=100000,
                                                min_active_states=0,
                                                max_active_states=10000,
                                                seqframe_idx_name='seqframe',
                                                frame_idx_name='frame')
            assert torch.all(
                torch.eq(out_fsa.seqframe,
                         torch.tensor([0, 1, 2], device=device)))

            assert torch.all(
                torch.eq(out_fsa.frame, torch.tensor([0, 1, 2],
                                                     device=device)))

            scores = out_fsa.get_tot_scores(log_semiring=False,
                                            use_double_scores=False)
            scores.sum().backward()

            # `expected` results are computed using gtn.
            # See https://bit.ly/3oYObeb
            expected_scores_out_fsa = torch.tensor([1.2, 2.06, 3.0],
                                                   device=device)
            expected_grad_fsa = torch.tensor([1.0, 0.0, 1.0, 1.0],
                                             device=device)
            expected_grad_log_prob = torch.tensor(
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                device=device).reshape_as(log_prob)

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
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.requires_grad_(True)
            fsa_vec = k2.create_fsa_vec([fsa])
            log_prob = torch.tensor(
                [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
                 [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
                device=device,
                requires_grad=True)

            supervision_segments = torch.tensor([[0, 0, 2], [1, 0, 3]],
                                                dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                                dense_fsa_vec,
                                                search_beam=100000,
                                                output_beam=100000,
                                                min_active_states=0,
                                                max_active_states=10000,
                                                seqframe_idx_name='seqframe',
                                                frame_idx_name='frame')
            assert torch.all(
                torch.eq(out_fsa.seqframe,
                         torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)))

            assert torch.all(
                torch.eq(out_fsa.frame,
                         torch.tensor([0, 1, 2, 0, 1, 2, 3], device=device)))

            assert out_fsa.shape == (2, None,
                                     None), 'There should be two FSAs!'

            scores = out_fsa.get_tot_scores(log_semiring=False,
                                            use_double_scores=False)
            scores.sum().backward()

            # `expected` results are computed using gtn.
            # See https://bit.ly/3oYObeb
            expected_scores_out_fsa = torch.tensor(
                [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0], device=device)
            expected_grad_fsa = torch.tensor([2.0, 1.0, 2.0, 2.0],
                                             device=device)
            expected_grad_log_prob = torch.tensor([
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0
            ]).reshape_as(log_prob).to(device)

            assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
            assert torch.allclose(expected_grad_fsa, fsa.scores.grad)
            assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_fsas(self):
        s1 = '''
            0 1 1 1.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''

        s2 = '''
            0 1 1 1.0
            1 1 1 50.0
            1 2 2 2.0
            2 3 -1 3.0
            3
        '''
        for device in self.devices:
            fsa1 = k2.Fsa.from_str(s1).to(device)
            fsa2 = k2.Fsa.from_str(s2).to(device)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

            log_prob = torch.tensor(
                [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06], [0.0, 0.0, 0.0]],
                 [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
                device=device,
                requires_grad=True)

            supervision_segments = torch.tensor([[0, 0, 2], [1, 0, 3]],
                                                dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                                dense_fsa_vec,
                                                search_beam=100000,
                                                output_beam=100000,
                                                min_active_states=0,
                                                max_active_states=10000,
                                                seqframe_idx_name='seqframe',
                                                frame_idx_name='frame')
            assert torch.all(
                torch.eq(out_fsa.seqframe,
                         torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)))

            assert torch.all(
                torch.eq(out_fsa.frame,
                         torch.tensor([0, 1, 2, 0, 1, 2, 3], device=device)))

            assert out_fsa.shape == (2, None,
                                     None), 'There should be two FSAs!'

            scores = out_fsa.get_tot_scores(log_semiring=False,
                                            use_double_scores=False)
            scores.sum().backward()

            # `expected` results are computed using gtn.
            # See https://bit.ly/3oYObeb
            expected_scores_out_fsa = torch.tensor(
                [1.2, 2.06, 3.0, 1.2, 50.5, 2.0, 3.0], device=device)

            expected_grad_fsa1 = torch.tensor([1.0, 1.0, 1.0], device=device)
            expected_grad_fsa2 = torch.tensor([1.0, 1.0, 1.0, 1.0],
                                              device=device)
            expected_grad_log_prob = torch.tensor([
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0
            ]).reshape_as(log_prob).to(device)

            assert torch.allclose(out_fsa.scores, expected_scores_out_fsa)
            assert torch.allclose(expected_grad_fsa1, fsa1.scores.grad)
            assert torch.allclose(expected_grad_fsa2, fsa2.scores.grad)
            assert torch.allclose(expected_grad_log_prob, log_prob.grad)

    def test_two_fsas_long_pruned(self):
        # as test_two_fsas_long in intersect_dense_test.py,
        # but with pruned intersection
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
        for device in self.devices:
            fsa1 = k2.Fsa.from_str(s1)
            fsa2 = k2.Fsa.from_str(s2)

            fsa1.requires_grad_(True)
            fsa2.requires_grad_(True)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])
            log_prob = torch.rand((2, 100, 3),
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)

            supervision_segments = torch.tensor([[0, 1, 95], [1, 20, 50]],
                                                dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            fsa_vec = fsa_vec.to(device)
            out_fsa = k2.intersect_dense_pruned(fsa_vec,
                                                dense_fsa_vec,
                                                search_beam=100,
                                                output_beam=100,
                                                min_active_states=1,
                                                max_active_states=10,
                                                seqframe_idx_name='seqframe',
                                                frame_idx_name='frame')

            expected_seqframe = torch.arange(96).to(torch.int32).to(device)
            assert torch.allclose(out_fsa.seqframe, expected_seqframe)

            # the second output FSA is empty since there is no self-loop in fsa2
            assert torch.allclose(out_fsa.frame, expected_seqframe)

            assert out_fsa.shape == (2, None,
                                     None), 'There should be two FSAs!'

            scores = out_fsa.get_tot_scores(log_semiring=False,
                                            use_double_scores=False)
            scores.sum().backward()


if __name__ == '__main__':
    unittest.main()
