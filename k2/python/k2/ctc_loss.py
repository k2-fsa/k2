# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

try:
    from typing import Literal  # for Python >= 3.8
except ImportError:
    from typing_extensions import Literal  # for python < 3.8

from typing import Optional

import torch
import torch.nn as nn

from .autograd import intersect_dense
from .dense_fsa_vec import DenseFsaVec
from .fsa import Fsa


class CtcLoss(nn.Module):
    '''Ctc Loss computation in k2. It produces the same output as `torch.CtcLoss`
    if given the same input.

    One difference between `k2.CtcLoss` and `torch.CtcLoss` is that k2 accepts
    a general FSA while PyTorch requires a linear FSA (represented as a list).
    That means, `k2.CtcLoss` supports words with multiple pronunciations.

    See `k2/python/tests/ctc_loss_test.py` for usage.

    We assume that the blank label is always 0. The arguments `reduction` and
    `target_lengths` have the same meaning as `torch.CtcLoss`.
    '''

    def __init__(self):
        super().__init__()

    def forward(self,
                decoding_graph: Fsa,
                dense_fsa_vec: DenseFsaVec,
                output_beam: float = 10,
                reduction: Literal['none', 'mean', 'sum'] = 'mean',
                use_double_scores: bool = True,
                target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Compute the CTC loss given a decoding graph and a dense fsa vector.

        Args:
          decoding_graph:
            An FsaVec. It can be the composition result of a CTC topology
            and a transcript.
          dense_fsa_vec:
            It represents the neural network output. Refer to the help
            information in :class:`k2.DenseFsaVec`.
          output_beam:
             Beam to prune output, similar to lattice-beam in Kaldi.  Relative
             to best path of output.
          reduction:
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the output losses
            will be **divided** by the target lengths and then the **mean** over
            the batch is taken. 'sum': sum the output losses over batches.
          use_double_scores:
            True to use double precision floating point in computing
            the total scores. False to use single precision.
          target_lengths:
            Used only when `reduction` is `mean`. It is a 1-D tensor of batch
            size representing lengths of the targets, e.g., number of phones or
            number of word pieces in a sentence.
        Returns:
          If `reduction` is `none`, return a 1-D tensor with size equal to batch
          size. If `reduction` is `mean` or `sum`, return a scalar.
        '''
        assert reduction in ('none', 'mean', 'sum')
        lattice = intersect_dense(decoding_graph, dense_fsa_vec, output_beam)

        tot_scores = lattice.get_tot_scores(
            log_semiring=True, use_double_scores=use_double_scores)
        loss = -1 * tot_scores
        loss = loss.to(torch.float32)

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            assert reduction == 'mean'
            loss /= target_lengths
            return loss.mean()


def ctc_loss(decoding_graph: Fsa,
             dense_fsa_vec: DenseFsaVec,
             output_beam: float = 10,
             reduction: Literal['none', 'mean', 'sum'] = 'mean',
             use_double_scores: bool = True,
             target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''Compute the CTC loss given a decoding graph and a dense fsa vector.

    Args:
      decoding_graph:
        An FsaVec. It can be the composition result of a ctc topology
        and a transcript.
      dense_fsa_vec:
        It represents the neural network output. Refer to the help information
        in :class:`k2.DenseFsaVec`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      reduction:
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied, 'mean': the output losses will be
        divided by the target lengths and then the mean over the batch is taken.
        'sum': sum the output losses over batches.
      use_double_scores:
        True to use double precision floating point in computing
        the total scores. False to use single precision.
      target_lengths:
        Used only when `reduction` is `mean`. It is a 1-D tensor of batch
        size representing lengths of the targets, e.g., number of phones or
        number of word pieces in a sentence.
    Returns:
      If `reduction` is `none`, return a 1-D tensor with size equal to batch
      size. If `reduction` is `mean` or `sum`, return a scalar.
    '''
    return CtcLoss()(decoding_graph, dense_fsa_vec, output_beam, reduction,
                     use_double_scores, target_lengths)
