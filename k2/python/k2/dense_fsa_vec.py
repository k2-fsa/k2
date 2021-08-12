# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

from typing import Union

import torch
import _k2

from .fsa import Fsa


class DenseFsaVec(object):

    # Note: you can access members self.scores and self.dense_fsa_vec.
    # self.scores is a torch.Tensor containing the scores; it will
    # contain rows of the `log_probs` arg given to __init__ interspersed
    # with rows representing final-arcs.  The structure is something like:
    #
    #  [ [ -inf x x x x x x  ]
    #    [ -inf x x x x x x  ]
    #    [ -inf x x x x x x  ]
    #    [  0 -inf -inf -inf.. ]
    #    [ -inf x x x x x x  ]
    #     ...
    #  ]
    # where the x's come from the `log_probs` arg, and the 0's and
    # -inf's are added by this class (those special rows with no x's
    # correspond to the final-arcs in the FSAs; the 0 corresponds to
    # symbol -1.)

    def __init__(self,
                 log_probs: torch.Tensor,
                 supervision_segments: torch.Tensor,
                 allow_truncate: int = 0) -> None:
        '''Construct a DenseFsaVec from neural net log-softmax outputs.

        Args:
          log_probs:
            A 3-D tensor of dtype `torch.float32` with shape `(N, T, C)`,
            where `N` is the number of sequences, `T` the maximum input
            length, and `C` the number of output classes.
          supervision_segments:
            A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
            Each row contains information for a supervision segment. Column 0
            is the `sequence_index` indicating which sequence this segment
            comes from; column 1 specifies the `start_frame` of this segment
            within the sequence; column 2 contains the `duration` of this
            segment.

            Note:
              - `0 < start_frame + duration <= T + allow_truncate`
              - `0 <= start_frame < T`
              - `duration > 0`

            Caution:
              If the resulting dense fsa vec is used as an input to
              `k2.intersect_dense`, then the last column, i.e., the duration
              column, has to be sorted in **decreasing** order.
              That is, the first supervision_segment (the first row) has the
              largest duration.
              Otherwise, you don't need to sort the last column.

              `k2.intersect_dense` is often used in the training stage, so
              you should usually sort dense fsa vecs by its duration
              in training. `k2.intersect_dense_pruned` is usually used in the
              decoding stage, so you don't need to sort dense fsa vecs in
              decoding.
          allow_truncate:
            If not zero, it truncates at most this number of frames from
            duration in case start_frame + duration > T.
        '''
        assert log_probs.ndim == 3
        assert log_probs.dtype == torch.float32
        assert supervision_segments.ndim == 2
        assert supervision_segments.dtype == torch.int32
        assert supervision_segments.device.type == 'cpu'
        assert allow_truncate >= 0

        N, T, C = log_probs.shape

        # Also, if a particular FSA has T frames of neural net output,
        # we actually have T+1 potential indexes, 0 through T, so there is
        # space for the terminating final-symbol on frame T.  (On the last
        # frame, the final symbol has logprob=0, the others have logprob=-inf).
        placeholder = torch.tensor([0])  # this extra row is for the last frame
        indexes = []
        last_frame_indexes = []
        cur = 0
        for segment in supervision_segments.tolist():
            segment_index, start_frame, duration = segment
            assert 0 <= segment_index < N
            assert 0 <= start_frame < T
            assert duration > 0
            assert start_frame + duration <= T + allow_truncate
            offset = segment_index * T
            end_frame = min(start_frame + duration, T)  # exclusive

            # update duration if it's too large
            duration = end_frame - start_frame

            indexes.append(torch.arange(start_frame, end_frame) + offset)
            indexes.append(placeholder)
            cur += duration  # NOTE: the duration may be updated above
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
        scores[:, 0] = float('-inf')
        scores[last_frame_indexes] = torch.tensor([0] + [float('-inf')] * C,
                                                  device=device)

        row_splits = torch.zeros(supervision_segments.size(0) + 1,
                                 device='cpu',
                                 dtype=torch.int32)
        row_splits[1:] = torch.tensor(last_frame_indexes) + 1

        # minus one to exclude the fake row [0, -inf, -inf, ...]
        self._duration = row_splits[1:] - row_splits[:-1] - 1

        row_splits = row_splits.to(device)
        self.dense_fsa_vec = _k2.DenseFsaVec(scores, row_splits)
        self.scores = scores  # for back propagation

    @property
    def duration(self) -> torch.Tensor:
        '''Return the duration (on CPU) of each seq.
        '''
        if not hasattr(self, '_duration'):
            row_splits = self.dense_fsa_vec.shape().row_splits(1)
            # minus one to exclude the fake row [0, -inf, -inf, ...]
            duration = row_splits[1:] - row_splits[:-1] - 1
            self._duration = duration.cpu()
        return self._duration

    @classmethod
    def _from_dense_fsa_vec(cls, dense_fsa_vec: _k2.DenseFsaVec,
                            scores: torch.Tensor) -> 'DenseFsaVec':
        '''Construct a DenseFsaVec from `_k2.DenseFsaVec` and `scores`.

        Note:
          It is intended for internal use. Users will normally not use it.

        Args:
          dense_fsa_vec: An instance of `_k2.DenseFsaVec`.
          scores: The `scores` of `_k2.DenseFsaVec` for back propagation.

        Return:
          An instance of DenseFsaVec.
        '''
        ans = cls.__new__(cls)
        super(DenseFsaVec, ans).__init__()
        ans.dense_fsa_vec = dense_fsa_vec
        ans.scores = scores
        return ans

    def dim0(self) -> int:
        '''Return number of supervision segments.'''
        return self.dense_fsa_vec.dim0()

    def __str__(self) -> str:
        return self.dense_fsa_vec.to_str()

    @property
    def device(self) -> torch.device:
        return self.scores.device

    def is_cpu(self) -> bool:
        '''Return true if this DenseFsaVec is on CPU.

        Returns:
          True if the DenseFsaVec is on CPU; False otherwise.
        '''
        return self.device.type == 'cpu'

    def is_cuda(self) -> bool:
        '''Return true if this DenseFsaVec is on GPU.

        Returns:
          True if the DenseFsaVec is on GPU; False otherwise.
        '''
        return self.device.type == 'cuda'

    def to(self, device: Union[torch.device, str]) -> 'DenseFsaVec':
        '''Move the DenseFsaVec onto a given device.

        Args:
          device:
            An instance of `torch.device` or a string that can be used to
            construct a `torch.device`, e.g., 'cpu', 'cuda:0'.
            It supports only cpu and cuda devices.

        Returns:
          Returns a new DenseFsaVec which is this object copied to the given
          device (or this object itself, if the device was the same).
        '''
        if isinstance(device, str):
            device = torch.device(device)

        assert device.type in ('cpu', 'cuda')
        if device == self.scores.device:
            return self

        scores = self.scores.to(device)
        dense_fsa_vec = self.dense_fsa_vec.to(device)
        return DenseFsaVec._from_dense_fsa_vec(dense_fsa_vec, scores)


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
