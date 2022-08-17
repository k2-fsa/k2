#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corp.        (authors: Wei Kang)
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
#  ctest --verbose -R generate_denominator_lattice_test_py

import unittest

import k2
import torch

from torch.distributions.categorical import Categorical
from typing import Tuple


def _roll_by_shifts(
    src: torch.Tensor, shifts: torch.LongTensor
) -> torch.Tensor:
    """Roll tensor with different shifts for each row.

    Note:
      We assume the src is a 3 dimensions tensor and roll the last dimension.

    Example:

      >>> src = torch.arange(15).reshape((1,3,5))
      >>> src
      tensor([[[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]]])
      >>> shift = torch.tensor([[1, 2, 3]])
      >>> shift
      tensor([[1, 2, 3]])
      >>> _roll_by_shifts(src, shift)
      tensor([[[ 4,  0,  1,  2,  3],
               [ 8,  9,  5,  6,  7],
               [12, 13, 14, 10, 11]]])
    """
    assert src.dim() == 3
    (B, T, S) = src.shape
    assert shifts.shape == (B, T)

    index = (
        torch.arange(S, device=src.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - shifts.reshape(B, T, 1)) % S
    return torch.gather(src, 2, index)


def simulate_importance_sampling(
    boundary: torch.Tensor,
    vocab_size: int,
    path_length: int,
    num_paths: int,
    context_size: int = 2,
    blank_id: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      boundary:
        It is a tensor with shape (B,), containing the number of frames for
        each sequence.
      vocab_size:
        Vocabulary size.
      path_length:
        How many symbols we will sample for each path.
      num_paths:
        How many paths we will sample for each sequence.
      context_size:
        The number of left symbols.
      blank_id:
        Stands for null context.

    Returns:
      Three tensors will be returned.
      - sampled_paths:
        A tensor of shape (batch_size, num_paths, path_length), containing the
        sampled symbol ids.
      - sampling_probs:
        A tensor of shape (batch_size, num_paths, path_length), containing the
        sampling probabilities of the sampled symbols.
      - left_symbols:
        A tensor of shape (batch_size, num_paths, path_length, context_size),
        containing the left symbols of the sampled symbols.
      - frame_ids:
        A tensor of shape (batch_size, num_paths, path_length), containing the
        frame ids at which we sampled the symbols.
    """
    # we sample paths from frame 0
    batch_size = boundary.numel()

    t_index = torch.zeros(
        (batch_size, num_paths), dtype=torch.int64, device=device
    )

    t_index_max = boundary.view(batch_size, 1).expand(batch_size, num_paths)

    left_symbols = torch.tensor(
        [blank_id], dtype=torch.int64, device=device
    ).expand(batch_size, num_paths, context_size)

    sampled_paths_list = []
    sampling_probs_list = []
    frame_ids_list = []
    left_symbols_list = []

    for i in range(path_length):
        probs = torch.randn(batch_size, num_paths, vocab_size)
        probs = torch.softmax(probs, -1)
        # sampler: https://pytorch.org/docs/stable/distributions.html#categorical
        sampler = Categorical(probs=probs)

        # sample one symbol for each path
        # index : (batch_size, num_paths)
        index = sampler.sample()
        sampled_paths_list.append(index)

        # gather sampling probabilities for corresponding indexs
        # sampling_prob : (batch_size, num_paths, 1)
        sampling_probs = torch.gather(probs, dim=2, index=index.unsqueeze(2))
        sampling_probs_list.append(sampling_probs.squeeze(2))

        frame_ids_list.append(t_index)

        left_symbols_list.append(left_symbols)

        # update (t, s) for each path
        # index == 0 means the sampled symbol is blank
        t_mask = index == 0
        # t_index = torch.where(t_mask, t_index + 1, t_index)
        t_index = t_index + 1

        final_mask = t_index >= t_index_max
        reach_final = torch.any(final_mask)
        if reach_final:
            new_t_index = torch.randint(0, torch.min(t_index_max) - 1, (1,)).item()
            t_index.masked_fill_(final_mask, new_t_index)

        current_symbols = torch.cat([left_symbols, index.unsqueeze(2)], dim=2)
        left_symbols = _roll_by_shifts(current_symbols, t_mask.to(torch.int64))
        left_symbols = left_symbols[:, :, 1:]
        if reach_final:
            left_symbols.masked_fill_(final_mask.unsqueeze(2), blank_id)

    # sampled_paths : (batch_size, num_paths, path_lengths)
    sampled_paths = torch.stack(sampled_paths_list, dim=2).int()
    # sampling_probs : (batch_size, num_paths, path_lengths)
    sampling_probs = torch.stack(sampling_probs_list, dim=2)
    # frame_ids : (batch_size , num_paths, path_lengths)
    frame_ids = torch.stack(frame_ids_list, dim=2).int()
    # left_symbols : (batch_size, num_paths, path_lengths, context_size)
    left_symbols = torch.stack(left_symbols_list, dim=2).int()
    return sampled_paths, frame_ids, sampling_probs, left_symbols


class TestConnect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test(self):
        context_size = 2
        batch_size, num_paths, path_length, vocab_size = 2, 3, 10, 10
        boundary_ = torch.tensor([6, 9], dtype=torch.int32)
        (
            sampled_paths_,
            frame_ids_,
            sampling_probs_,
            left_symbols_,
        ) = simulate_importance_sampling(
            boundary=boundary_,
            vocab_size=vocab_size,
            num_paths=num_paths,
            path_length=path_length,
            context_size=context_size,
        )
        path_scores_ = torch.randn(
            (batch_size, num_paths, path_length), dtype=torch.float
        )
        for device in self.devices:
            boundary = boundary_.to(device)
            sampled_paths = sampled_paths_.to(device)
            sampling_probs = sampling_probs_.to(device)
            frame_ids = frame_ids_.to(device)
            left_symbols = left_symbols_.to(device)
            path_scores = path_scores_.detach().clone().to(device)
            path_scores.requires_grad_(True)
            fsa = k2.generate_denominator_lattice(
                sampled_paths=sampled_paths,
                frame_ids=frame_ids,
                left_symbols=left_symbols,
                sampling_probs=sampling_probs,
                boundary=boundary,
                path_scores=path_scores,
                vocab_size=vocab_size,
                context_size=context_size,
            )
            print(fsa)
            fsa = k2.connect(k2.top_sort(fsa))
            scores = torch.sum(
                fsa.get_tot_scores(log_semiring=True, use_double_scores=False)
            )
            scores.backward()
            print(path_scores.grad)


if __name__ == "__main__":
    unittest.main()
