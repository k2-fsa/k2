# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors

import torch
import _k2
import numpy as np

from .fsa import Fsa


class DenseFsaVec(object):

    def __init__(self, log_probs: torch.Tensor,
                 supervision_segments: torch.Tensor) -> None:
        '''Construct a DenseFsaVec from neural net log-softmax outputs.

        Args:
          log_probs:
            A 3-D tensor of dtype ``torch.float32`` with shape ``(N, T, C)``,
            where ``N`` is the number of sequences, ``T`` the maximum input
            length, and ``C`` the number of output classes.
          supervision_segments:
            A 2-D **CPU** tensor of dtype ``torch.int32`` with 3 columns.
            Each row contains information for a supervision segment. Column 0
            is the ``sequence_index`` indicating which sequence this segment
            comes from; column 1 specifies the ``start_frame`` of this segment
            within the sequence; column 2 contains the ``duration`` of this
            segment.

            Note:
              - ``0 < start_frame + duration <= T``
              - ``0 <= start_frame < T``
              - ``duration > 0``
        '''
        assert log_probs.ndim == 3
        assert log_probs.dtype == torch.float32
        assert supervision_segments.ndim == 2
        assert supervision_segments.dtype == torch.int32
        assert supervision_segments.device.type == 'cpu'

        N, T, C = log_probs.shape

        # Also, if a particular FSA has T frames of neural net output,
        # we actually have T+1 potential indexes, 0 through T, so there is
        # space for the terminating final-symbol on frame T.  (On the last
        # frame, the final symbol has logprob=0, the others have logprob=-inf).
        placeholder = torch.tensor([0])  # this extra row is for the last frame
        indexes = []
        last_frame_indexes = []
        cur = 0
        for segment in supervision_segments:
            segment_index, start_frame, duration = segment.tolist()
            assert 0 <= segment_index < N
            assert 0 <= start_frame < T
            assert duration > 0
            assert start_frame + duration <= T
            offset = segment_index * T
            indexes.append(
                torch.arange(start_frame, start_frame + duration) + offset)
            indexes.append(placeholder)
            cur += duration
            last_frame_indexes.append(cur)
            cur += 1  # increment for the extra row

        device = log_probs.device
        indexes = torch.cat(indexes).to(device)

        scores = torch.empty(cur, C + 1, dtype=log_probs.dtype, device=device)
        scores[:, 1:] = log_probs.reshape(-1, C).index_select(0, indexes)

        # `scores` contains -infinity in certain locations: in scores[j,0] where
        # j is not the last row-index for a given FSA-index, and scores[j,k]
        # where j is the last row-index for a given FSA-index and k>0.
        # The remaining locations contain the neural net output, except
        # scores[j,0] where j is the last row-index for a given FSA-index;
        # this contains zero.
        scores[:, 0] = np.NINF
        scores[last_frame_indexes] = torch.tensor([0] + [np.NINF] * C,
                                                  device=device)

        row_splits = torch.zeros(supervision_segments.size(0) + 1,
                                 device='cpu',
                                 dtype=torch.int32)
        row_splits[1:] = torch.tensor(last_frame_indexes) + 1
        row_splits = row_splits.to(device)
        self.dense_fsa_vec = _k2.DenseFsaVec(scores, row_splits)
        self.scores = scores  # for back propagation

    def dim0(self) -> int:
        '''Return number of supervision segments'''
        return self.dense_fsa_vec.dim0()

    def __str__(self) -> str:
        return self.dense_fsa_vec.to_str()


def convert_dense_to_fsa_vec(dense_fsa_vec: DenseFsaVec) -> Fsa:
    '''Convert a DenseFsaVec to an FsaVec.

    Caution:
      Intended for use in testing/debug mode only. This operation is NOT
      differentiable.

    Args:
      dense_fsa_vec:
        DenseFsaVec to convert.

    Returns:
      The converted FsaVec .
    '''
    ragged_arc = _k2.convert_dense_to_fsa_vec(dense_fsa_vec.dense_fsa_vec)
    return Fsa(ragged_arc)
