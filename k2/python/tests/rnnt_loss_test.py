#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation   (authors: Daniel Povey,
#                                                     Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R rnnt_loss_test_py

import unittest

import k2
import random
import torch


class TestRnntLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_rnnt_logprobs_basic(self):
        B = 1
        S = 3
        T = 4
        # C = 3
        for device in self.devices:
            # lm: [B][S+1][C]
            lm = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                              dtype=torch.float,
                              device=device)
            # am: [B][T][C]
            am = torch.tensor([[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                              dtype=torch.float,
                              device=device)
            termination_symbol = 2
            symbols = torch.tensor([[0, 1, 0]],
                                   dtype=torch.long,
                                   device=device)

            px, py = k2.get_rnnt_logprobs(lm, am, symbols, termination_symbol)
            assert px.shape == (B, S, T + 1)
            assert py.shape == (B, S + 1, T)
            assert symbols.shape == (B, S)
            m = k2.mutual_information_recursion(px, py)

            if device == torch.device("cpu"):
                expected = m
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            probs = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(probs, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.0,
                                 boundary=None)
            assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            probs = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(probs, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.0,
                                 boundary=None)
            assert torch.allclose(m, expected.to(device))

    def test_rnnt_logprobs_random(self):
        B = 5
        S = 20
        T = 300
        C = 100
        am_ = torch.randn((B, T, C), dtype=torch.float64)
        lm_ = torch.randn((B, S + 1, C), dtype=torch.float64)
        symbols_ = torch.randint(0, C, (B, S))
        termination_symbol = C - 1

        for device in self.devices:
            # lm: [B][S+1][C]
            lm = lm_.to(device)
            # am: [B][T][C]
            am = am_.to(device)
            symbols = symbols_.to(device)

            px, py = k2.get_rnnt_logprobs(lm, am, symbols, termination_symbol)
            assert px.shape == (B, S, T + 1)
            assert py.shape == (B, S + 1, T)
            assert symbols.shape == (B, S)
            m = k2.mutual_information_recursion(px, py)

            if device == torch.device("cpu"):
                expected = m
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            probs = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(probs, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.0,
                                 boundary=None)
            assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            probs = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(probs, symbols, termination_symbol, None)
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.0,
                                 boundary=None)
            assert torch.allclose(m, expected.to(device))

    def test_rnnt_logprobs_aux(self):
        B = 1
        S = 3
        T = 4
        # C = 3
        for device in self.devices:
            # lm: [B][S+1][C]
            lm = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                              dtype=torch.float,
                              device=device)
            # am: [B][T][C]
            am = torch.tensor([[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                              dtype=torch.float,
                              device=device)

            termination_symbol = 2
            symbols = torch.tensor([[0, 1, 0]],
                                   dtype=torch.long,
                                   device=device)

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.333,
                                 boundary=None)

            if device == torch.device("cpu"):
                expected = m
            assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_aux(lm,
                                 am,
                                 symbols,
                                 termination_symbol,
                                 lm_only_scale=0.0,
                                 am_only_scale=0.333,
                                 boundary=None)
            assert torch.allclose(m, expected.to(device))

    def test_rnnt_logprobs_pruned(self):
        B = 4
        T = 300
        S = 50
        C = 10
        am_ = torch.randn((B, T, C), dtype=torch.float64)
        lm_ = torch.randn((B, S + 1, C), dtype=torch.float64)
        symbols_ = torch.randint(0, C, (B, S))
        terminal_symbol = C - 1

        for device in self.devices:
            # normal rnnt
            am = am_.to(device)
            lm = lm_.to(device)
            symbols = symbols_.to(device)
            t_am = am.unsqueeze(2).float()
            t_lm = lm.unsqueeze(1).float()
            t_prob = t_am + t_lm
            # nonlinear transform
            t_prob = torch.sigmoid(t_prob)
            k2_loss = k2.rnnt_loss(t_prob, symbols, terminal_symbol, None)

            print("unpruned rnnt loss: ", k2_loss)

            # pruning
            k2_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
                lm, am, symbols, terminal_symbol, None, True)

            for r in range(2, 50, 5):
                boundary = torch.zeros((B, 4),
                                       dtype=torch.int64,
                                       device=device)
                boundary[:, 2] = S
                boundary[:, 3] = T
                ranges = k2.get_rnnt_prune_ranges(px_grad, py_grad, boundary, r)
                # (B, T, r, C)
                am_p, lm_p = k2.do_rnnt_pruning(am, lm, ranges)

                t_prob_p = am_p + lm_p

                # nonlinear transform
                t_prob_p = torch.sigmoid(t_prob_p)
                boundary = torch.zeros((B, 4),
                                       dtype=torch.int64,
                                       device=device)
                boundary[:, 2] = ranges[:, -1, -1]
                boundary[:, 3] = T

                pruning_loss = k2.rnnt_loss_pruned(t_prob_p, symbols, ranges,
                                                   terminal_symbol, boundary)
                print(f"pruning loss with range {r} : ", pruning_loss)


if __name__ == "__main__":
    unittest.main()
