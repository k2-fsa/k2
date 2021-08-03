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

    See `k2/python/tests/ctc_loss_test.py <https://github.com/k2-fsa/k2/blob/master/k2/python/tests/ctc_loss_test.py>`_
    for usage.

    We assume that the blank label is always 0. The arguments `reduction` and
    `target_lengths` have the same meaning as their counterparts in
    `torch.CtcLoss`.
    '''

    def __init__(self,
                 output_beam: float,
                 reduction: Literal['none', 'mean', 'sum'] = 'sum',
                 use_double_scores: bool = True):
        '''
        Args:
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
        '''
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.output_beam = output_beam
        self.reduction = reduction
        self.use_double_scores = use_double_scores

    def forward(self,
                decoding_graph: Fsa,
                dense_fsa_vec: DenseFsaVec,
                target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Compute the CTC loss given a decoding graph and a dense fsa vector.

        Args:
          decoding_graph:
            An FsaVec. It can be the composition result of a CTC topology
            and a transcript.
          dense_fsa_vec:
            It represents the neural network output. Refer to the help
            information in :class:`k2.DenseFsaVec`.
          target_lengths:
            Used only when `reduction` is `mean`. It is a 1-D tensor of batch
            size representing lengths of the targets, e.g., number of phones or
            number of word pieces in a sentence.
        Returns:
          If `reduction` is `none`, return a 1-D tensor with size equal to batch
          size. If `reduction` is `mean` or `sum`, return a scalar.
        '''
        lattice = intersect_dense(decoding_graph, dense_fsa_vec,
                                  self.output_beam)

        tot_scores = lattice.get_tot_scores(
            log_semiring=True, use_double_scores=self.use_double_scores)
        loss = -1 * tot_scores
        loss = loss.to(torch.float32)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            assert self.reduction == 'mean'
            loss /= target_lengths
            return loss.mean()


def ctc_loss(decoding_graph: Fsa,
             dense_fsa_vec: DenseFsaVec,
             output_beam: float = 10,
             reduction: Literal['none', 'mean', 'sum'] = 'sum',
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
    m = CtcLoss(output_beam=output_beam,
                reduction=reduction,
                use_double_scores=use_double_scores)

    return m(decoding_graph, dense_fsa_vec, target_lengths)
