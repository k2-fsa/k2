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


def generate_mask(S: int, ranges: torch.Tensor) -> torch.Tensor:
    """
    Generate a boolean tensor of shape (B, T, S), where the elements with
    indexes in tensor ranges are False, others are True.

    Example:
    >>> ranges = torch.tensor(
            [[[0, 1, 2],
              [1, 2, 3],
              [1, 2, 3],
              [2, 3, 4],
              [2, 3, 4],
              [2, 3, 4]]]
        )
    >>> print (ranges)
    tensor([[[0, 1, 2],
         [1, 2, 3],
         [1, 2, 3],
         [2, 3, 4],
         [2, 3, 4],
         [2, 3, 4]]])
    >>> mask = generate_mask(5, ranges)
    >>> print (mask)
    tensor([[[False, False, False,  True,  True],
         [ True, False, False, False,  True],
         [ True, False, False, False,  True],
         [ True,  True, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True, False, False, False]]])

    Args:
      S:
        The expected size of third dimension of returned tensor.
      ranges:
        A index tensor with shape (B, T, s_range), all its elements must
        satisfy `0 <= ranges[:] < S`.
    """
    assert torch.all(ranges < S)
    B, T, s_range = ranges.shape
    mask = torch.cat(
        [
            torch.zeros((B, T, s_range), device=ranges.device),
            torch.ones((B, T, S - s_range), device=ranges.device),
        ],
        dim=2,
    )
    index = (
        torch.arange(S, device=ranges.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - ranges[:, :, 0].reshape(B, T, 1)) % S
    mask = torch.gather(mask, 2, index).bool()
    return mask


class TestRnntLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))
        try:
            import torchaudio
            import torchaudio.functional

            if hasattr(torchaudio.functional, "rnnt_loss"):
                cls.has_torch_rnnt_loss = True
            else:
                cls.has_torch_rnnt_loss = False
                print(
                    f"Current torchaudio version: {torchaudio.__version__}\n"
                    "Skipping the tests of comparing rnnt loss with torch "
                    "one, to enable these tests please install a "
                    "version >= 0.10.0"
                )
        except ImportError as e:
            cls.has_torch_rnnt_loss = False
            print(
                f"Import torchaudio error, error message: {e}\n"
                "Skipping the tests of comparing rnnt loss with torch "
                "one, to enable these tests, please install torchaudio "
                "with version >= 0.10.0"
            )

    def test_rnnt_loss_basic(self):
        B = 1
        S = 3
        T = 4
        # C = 3
        for device in self.devices:
            # lm: [B][S+1][C]
            lm = torch.tensor(
                [[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                dtype=torch.float,
                device=device,
            )
            # am: [B][T][C]
            am = torch.tensor(
                [[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                dtype=torch.float,
                device=device,
            )
            termination_symbol = 2
            symbols = torch.tensor([[0, 1, 0]], dtype=torch.long, device=device)

            px, py = k2.get_rnnt_logprobs(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
            )
            assert px.shape == (B, S, T + 1)
            assert py.shape == (B, S + 1, T)
            assert symbols.shape == (B, S)
            m = k2.mutual_information_recursion(px=px, py=py, boundary=None)

            if device == torch.device("cpu"):
                expected = -m
            assert torch.allclose(-m, expected.to(device))

            # test rnnt_loss_simple
            m = k2.rnnt_loss_simple(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            # test rnnt_loss_smoothed
            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            logits = am.unsqueeze(2) + lm.unsqueeze(1)

            # test rnnt_loss
            m = k2.rnnt_loss(
                logits=logits,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            # compare with torchaudio rnnt_loss
            if self.has_torch_rnnt_loss:
                import torchaudio.functional

                m = torchaudio.functional.rnnt_loss(
                    logits=logits,
                    targets=symbols.int(),
                    logit_lengths=torch.tensor(
                        [T] * B, dtype=torch.int32, device=device
                    ),
                    target_lengths=torch.tensor(
                        [S] * B, dtype=torch.int32, device=device
                    ),
                    blank=termination_symbol,
                    reduction="none",
                )
                assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_simple(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            logits = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(
                logits=logits,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

    def test_rnnt_loss_random(self):
        B = 5
        S = 20
        T = 300
        C = 100
        frames = torch.randint(S, T, (B,))
        seq_length = torch.randint(3, S - 1, (B,))
        T = torch.max(frames)
        S = torch.max(seq_length)

        am_ = torch.randn((B, T, C), dtype=torch.float32)
        lm_ = torch.randn((B, S + 1, C), dtype=torch.float32)
        symbols_ = torch.randint(0, C - 1, (B, S))
        termination_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_length
        boundary_[:, 3] = frames

        for rnnt_type in ["regular", "modified", "constrained"]:
            for device in self.devices:
                # lm: [B][S+1][C]
                lm = lm_.to(device)
                # am: [B][T][C]
                am = am_.to(device)
                symbols = symbols_.to(device)
                boundary = boundary_.to(device)

                px, py = k2.get_rnnt_logprobs(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert (
                    px.shape == (B, S, T) if rnnt_type != "regular" else (B, S, T + 1)
                )
                assert py.shape == (B, S + 1, T)
                assert symbols.shape == (B, S)
                m = k2.mutual_information_recursion(px=px, py=py, boundary=boundary)

                if device == torch.device("cpu"):
                    expected = -torch.mean(m)
                assert torch.allclose(-torch.mean(m), expected.to(device))

                m = k2.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

                m = k2.rnnt_loss_smoothed(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    lm_only_scale=0.0,
                    am_only_scale=0.0,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                m = k2.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

                # compare with torchaudio rnnt_loss
                if self.has_torch_rnnt_loss and rnnt_type == "regular":
                    import torchaudio.functional

                    m = torchaudio.functional.rnnt_loss(
                        logits=logits,
                        targets=symbols.int(),
                        logit_lengths=boundary[:, 3].int(),
                        target_lengths=boundary[:, 2].int(),
                        blank=termination_symbol,
                    )
                    assert torch.allclose(m, expected.to(device))

                # should be invariant to adding a constant for any frame.
                lm += torch.randn(B, S + 1, 1, device=device)
                am += torch.randn(B, T, 1, device=device)

                m = k2.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                m = k2.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

                m = k2.rnnt_loss_smoothed(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    lm_only_scale=0.0,
                    am_only_scale=0.0,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                )
                assert torch.allclose(m, expected.to(device))

    def test_rnnt_loss_gradient(self):
        if self.has_torch_rnnt_loss:
            import torchaudio.functional

            B = 5
            S = 20
            T = 300
            C = 100
            frames = torch.randint(S, T, (B,))
            seq_length = torch.randint(3, S - 1, (B,))
            T = torch.max(frames)
            S = torch.max(seq_length)

            am_ = torch.randn((B, T, C), dtype=torch.float32)
            lm_ = torch.randn((B, S + 1, C), dtype=torch.float32)
            symbols_ = torch.randint(0, C - 1, (B, S))
            termination_symbol = C - 1

            boundary_ = torch.zeros((B, 4), dtype=torch.int64)
            boundary_[:, 2] = seq_length
            boundary_[:, 3] = frames

            for device in self.devices:

                # lm: [B][S+1][C]
                lm = lm_.to(device)
                # am: [B][T][C]
                am = am_.to(device)
                symbols = symbols_.to(device)
                boundary = boundary_.to(device)

                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits.requires_grad_()
                k2_loss = k2.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=termination_symbol,
                    boundary=boundary,
                )
                k2_grad = torch.autograd.grad(k2_loss, logits)
                k2_grad = k2_grad[0]

                logits2 = logits.detach().clone().float()
                logits2.requires_grad_()
                torch_loss = torchaudio.functional.rnnt_loss(
                    logits=logits2,
                    targets=symbols.int(),
                    logit_lengths=boundary[:, 3].int(),
                    target_lengths=boundary[:, 2].int(),
                    blank=termination_symbol,
                )
                torch_grad = torch.autograd.grad(torch_loss, logits2)
                torch_grad = torch_grad[0]

                assert torch.allclose(k2_loss, torch_loss, atol=1e-2, rtol=1e-2)

                assert torch.allclose(k2_grad, torch_grad, atol=1e-2, rtol=1e-2)

    def test_rnnt_loss_smoothed(self):
        B = 1
        S = 3
        T = 4
        # C = 3
        for device in self.devices:
            # lm: [B][S+1][C]
            lm = torch.tensor(
                [[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                dtype=torch.float,
                device=device,
            )
            # am: [B][T][C]
            am = torch.tensor(
                [[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                dtype=torch.float,
                device=device,
            )

            termination_symbol = 2
            symbols = torch.tensor([[0, 1, 0]], dtype=torch.long, device=device)

            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.333,
                boundary=None,
            )

            if device == torch.device("cpu"):
                expected = m
            assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.333,
                boundary=None,
            )
            assert torch.allclose(m, expected.to(device))

    def test_rnnt_loss_pruned(self):
        B = 4
        T = 300
        S = 50
        C = 10

        frames = torch.randint(S, T, (B,))
        seq_length = torch.randint(3, S - 1, (B,))
        T = torch.max(frames)
        S = torch.max(seq_length)

        am_ = torch.randn((B, T, C), dtype=torch.float64)
        lm_ = torch.randn((B, S + 1, C), dtype=torch.float64)
        symbols_ = torch.randint(0, C - 1, (B, S))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_length
        boundary_[:, 3] = frames

        for rnnt_type in ["regular", "modified", "constrained"]:
            for device in self.devices:
                # normal rnnt
                am = am_.to(device)
                lm = lm_.to(device)
                symbols = symbols_.to(device)
                boundary = boundary_.to(device)

                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits = logits.float()

                # nonlinear transform
                logits = torch.sigmoid(logits)
                k2_loss = k2.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    reduction="none",
                )

                print(f"Unpruned rnnt loss with {rnnt_type} rnnt : {k2_loss}")

                # pruning
                k2_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                )

                for r in range(2, 50, 5):
                    ranges = k2.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )
                    # (B, T, r, C)
                    pruned_am, pruned_lm = k2.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm
                    # nonlinear transform
                    logits = torch.sigmoid(logits)

                    pruned_loss = k2.rnnt_loss_pruned(
                        logits=logits,
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="none",
                    )
                    print(f"Pruned loss with range {r} : {pruned_loss}")

    # Test the sequences that only have small number of symbols,
    # at this circumstance, the s_range would be greater than S, which will
    # raise errors (like, nan or inf loss) in our previous versions.
    def test_rnnt_loss_pruned_small_symbols_number(self):
        B = 2
        T = 20
        S = 3
        C = 10

        frames = torch.randint(S + 1, T, (B,))
        seq_lengths = torch.randint(1, S, (B,))
        T = torch.max(frames)
        S = torch.max(seq_lengths)

        am_ = torch.randn((B, T, C), dtype=torch.float64)
        lm_ = torch.randn((B, S + 1, C), dtype=torch.float64)
        symbols_ = torch.randint(0, C, (B, S))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_lengths
        boundary_[:, 3] = frames

        print(f"B = {B}, T = {T}, S = {S}, C = {C}")

        for rnnt_type in ["regular", "modified", "constrained"]:
            for device in self.devices:
                # normal rnnt
                am = am_.to(device)
                lm = lm_.to(device)
                symbols = symbols_.to(device)
                boundary = boundary_.to(device)

                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits = logits.float()

                # nonlinear transform
                logits = torch.sigmoid(logits)

                k2_loss = k2.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    reduction="none",
                )

                print(f"Unpruned rnnt loss with {rnnt_type} rnnt : {k2_loss}")

                # pruning
                k2_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                )

                S0 = 2
                if rnnt_type != "regular":
                    S0 = 1

                for r in range(S0, S + 2):
                    ranges = k2.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )
                    # (B, T, r, C)
                    pruned_am, pruned_lm = k2.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = torch.sigmoid(logits)

                    pruned_loss = k2.rnnt_loss_pruned(
                        logits=logits,
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="none",
                    )
                    print(f"Pruned loss with range {r} : {pruned_loss}")

    # Test more exact pruning bounds.
    # In our previous versions we use a less exact method to generate
    # pruning bounds which is different from the method we
    # publish in our paper(https://arxiv.org/pdf/2206.13236.pdf).
    # This tests the difference between these two methods.
    # We won't do any assertion in this test, just printing out the losses,
    # because we can not 100% sure that the new method is better than the old
    # one all the time, both of them are local optimal bounds.
    def test_prune_ranges(self):
        B = 5
        T = 200
        S = 100
        C = 50

        frames = torch.randint(S + 1, T, (B,))
        seq_lengths = torch.randint(1, S, (B,))
        T = torch.max(frames)
        S = torch.max(seq_lengths)

        am_ = torch.rand((B, T, C), dtype=torch.float64)
        lm_ = torch.rand((B, S + 1, C), dtype=torch.float64)
        symbols_ = torch.randint(0, C, (B, S))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_lengths
        boundary_[:, 3] = frames

        for device in self.devices:
            am = am_.to(device)
            lm = lm_.to(device)
            symbols = symbols_.to(device)
            boundary = boundary_.to(device)

            k2_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=terminal_symbol,
                boundary=boundary,
                return_grad=True,
                reduction="none",
            )

            for r in range(2, 20, 5):
                new_ranges = k2.get_rnnt_prune_ranges(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=r,
                )
                am_pruned, lm_pruned = k2.do_rnnt_pruning(
                    am=am,
                    lm=lm,
                    ranges=new_ranges,
                )

                logits = am_pruned + lm_pruned

                loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=symbols,
                    ranges=new_ranges,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    reduction="none",
                )

                print(f"Pruned with new ranges {r} : {loss}")

                old_ranges = k2.get_rnnt_prune_ranges_deprecated(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=r,
                )

                am_pruned, lm_pruned = k2.do_rnnt_pruning(
                    am=am,
                    lm=lm,
                    ranges=old_ranges,
                )
                logits = am_pruned + lm_pruned

                loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=symbols,
                    ranges=old_ranges,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    reduction="none",
                )

                print(f"Pruned with old ranges {r} : {loss}")

    # Check that training with an empty reference does not cause a crash.
    def test_rnnt_loss_empty_reference(self):
        B = 1
        S = 0
        T = 4
        # C = 3
        for device in self.devices:
            # lm: [B][S+1][C]
            lm = torch.tensor(
                [[[0, 0, 1]]],
                dtype=torch.float,
                device=device,
            )
            # am: [B][T][C]
            am = torch.tensor(
                [[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                dtype=torch.float,
                device=device,
            )
            termination_symbol = 2
            symbols = torch.tensor([[]], dtype=torch.long, device=device)

            px, py = k2.get_rnnt_logprobs(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
            )
            assert px.shape == (B, S, T + 1)
            assert py.shape == (B, S + 1, T)
            assert symbols.shape == (B, S)
            m = k2.mutual_information_recursion(px=px, py=py, boundary=None)

            if device == torch.device("cpu"):
                expected = -m
            assert torch.allclose(-m, expected.to(device))

            # test rnnt_loss_simple
            m = k2.rnnt_loss_simple(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            # test rnnt_loss_smoothed
            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            logits = am.unsqueeze(2) + lm.unsqueeze(1)

            # test rnnt_loss
            m = k2.rnnt_loss(
                logits=logits,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            # compare with torchaudio rnnt_loss
            if self.has_torch_rnnt_loss:
                import torchaudio.functional

                m = torchaudio.functional.rnnt_loss(
                    logits=logits,
                    targets=symbols.int(),
                    logit_lengths=torch.tensor(
                        [T] * B, dtype=torch.int32, device=device
                    ),
                    target_lengths=torch.tensor(
                        [S] * B, dtype=torch.int32, device=device
                    ),
                    blank=termination_symbol,
                    reduction="none",
                )
                assert torch.allclose(m, expected.to(device))

            # should be invariant to adding a constant for any frame.
            lm += torch.randn(B, S + 1, 1, device=device)
            am += torch.randn(B, T, 1, device=device)

            m = k2.rnnt_loss_simple(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            m = k2.rnnt_loss_smoothed(
                lm=lm,
                am=am,
                symbols=symbols,
                termination_symbol=termination_symbol,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))

            logits = am.unsqueeze(2) + lm.unsqueeze(1)
            m = k2.rnnt_loss(
                logits=logits,
                symbols=symbols,
                termination_symbol=termination_symbol,
                boundary=None,
                reduction="none",
            )
            assert torch.allclose(m, expected.to(device))


if __name__ == "__main__":
    unittest.main()
