# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)

import _k2
import numpy as np
import torch


def dense_fsa(log_probs: torch.Tensor,
              input_lengths: torch.Tensor) -> _k2.DenseFsaVec:
    '''Construct a DenseFsa from neural net log-softmax outputs.

    Args:
      log_probs:
        A 3-D tensor of dtype ``torch.float32`` with shape ``(N, T, C)``, where
        ``N`` is the batch size, ``T`` the maximum input length, and ``C`` the
        number of output classes.
      input_lengths:
        A 1-D tensor of dtype ``torch.int32`` with size ``(N)``, where ``N`` is
        the batch size. It represents the length of the input (must each NOT
        be greater than ``T``).
    Returns:
      An instance of ``_k2.DenseFsaVec``. You can treat it as an opaque object
      in Python.
    '''
    assert log_probs.ndim == 3
    assert log_probs.dtype == torch.float32
    assert input_lengths.ndim == 1
    assert input_lengths.dtype == torch.int32

    N = log_probs.size(0)
    assert input_lengths.size(0) == N

    num_rows = input_lengths.sum().item() + N
    # TODO(fangjun): remove the requirement for contiguous.
    # dense_fsa is supposed to be constructed inside a loss function,
    # so we do not set the `requires_grad` attribute for scores
    scores = torch.empty((num_rows, log_probs.size(-1) + 1),
                         device=log_probs.device,
                         dtype=log_probs.dtype).contiguous()
    scores[:, 0] = np.NINF  # negative infinity
    cur = 0
    for i in range(N):
        seq_len = input_lengths[i]
        end_row = cur + seq_len
        scores[end_row, 1:] = np.NINF
        scores[end_row, 0] = 0
        scores[cur:end_row, 1:] = log_probs[i, :seq_len, :]
        cur = end_row + 1
    sizes = torch.empty(N + 1,
                        device=input_lengths.device,
                        dtype=input_lengths.dtype)
    sizes[:-1] = input_lengths + 1
    return _k2.DenseFsaVec(scores, sizes)
