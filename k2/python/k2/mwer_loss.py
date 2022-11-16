# Copyright (c)  2022  Xiaomi Corporation (authors: Liyong Guo)

from typing import List, Literal, Union

import torch
import k2


class MWERLoss(torch.nn.Module):
    '''Minimum Word Error Rate Loss compuration in k2.

    See equation 2 of https://arxiv.org/pdf/2106.02302.pdf about its definition.
    '''

    def __init__(self,
                 temperature: float = 1.0,
                 use_double_scores: bool = True,
                 reduction: Literal['none', 'mean', 'sum'] = 'sum',
                 ) -> Union[torch.Tensor, k2.RaggedTensor]:
        '''
        Args:
          temperature:
            For long utterances, the dynamic range of scores will be too large
            and the posteriors will be mostly 0 or 1.
            To prevent this it might be a good idea to have an extra argument
            that functions like a temperature.
            We scale the logprobs by before doing the normalization.
          use_double_scores:
            True to use double precision floating point.
            False to use single precision.
          reduction:
            Specifies the reduction to apply to the output:
            'none' | 'sum' | 'mean'.
            'none': no reduction will be applied.
                    The returned 'loss' is a k2.RaggedTensor, with
                    loss.tot_size(0) == batch_size.
                    loss.tot_size(1) == total_num_paths_of_current_batch
                    If you want the MWER loss for each utterance, just do:
                    `loss_per_utt = loss.sum()`
                    Then loss_per_utt.shape[0] should be batch_size.
                    See more example usages in 'k2/python/tests/mwer_test.py'
            'sum': sum loss of each path over the whole batch together.
            'mean': divide above 'sum' by total num paths over the whole batch.
        '''

        assert reduction in ('none', 'mean', 'sum')
        super().__init__()
        self.temperature = temperature
        self.use_double_scores = use_double_scores
        self.reduction = reduction

    def forward(self,
                lattice: k2.Fsa,
                ref_texts: Union[k2.RaggedTensor, List[List[int]]],
                nbest_scale: float,
                num_paths: int) -> torch.Tensor:
        '''Compute the Minimum Word Error loss given
        a lattice and corresponding ref_texts.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          ref_texts:
            It can be one of the following types:
              - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
              - An instance of :class:`k2.RaggedTensor`.
                Must have `num_axes == 2` and with dtype `torch.int32`.
          nbest_scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
        Returns:
            Minimum Word Error Rate loss.
        '''

        nbest = k2.Nbest.from_lattice(
            lattice=lattice,
            num_paths=num_paths,
            use_double_scores=self.use_double_scores,
            nbest_scale=nbest_scale,
        )
        device = lattice.scores.device
        path_arc_shape = nbest.kept_path.shape.to(device)
        stream_path_shape = nbest.shape.to(device)

        hyps = nbest.build_levenshtein_graphs()
        refs = k2.levenshtein_graph(ref_texts, device=hyps.device)
        levenshtein_alignment = k2.levenshtein_alignment(
            refs=refs,
            hyps=hyps,
            hyp_to_ref_map=nbest.shape.row_ids(1),
            sorted_match_ref=True,
        )
        tot_scores = levenshtein_alignment.get_tot_scores(
            use_double_scores=self.use_double_scores, log_semiring=False
        )
        # Each path has a corresponding wer.
        wers = -tot_scores.to(device)

        # Group each log_prob into [path][arc]
        ragged_nbest_logp = k2.RaggedTensor(path_arc_shape, nbest.fsa.scores)
        # Get the probalitity of each path, in log format,
        # with shape [stream][path].
        path_logp = ragged_nbest_logp.sum() / self.temperature
        ragged_path_prob = k2.RaggedTensor(stream_path_shape, path_logp.exp())

        # Normalize prob of each path.
        den_prob = ragged_path_prob.sum()
        den_logp = torch.index_select(
            den_prob.log(), 0, ragged_path_prob.shape.row_ids(1))
        prob_normalized = (path_logp - den_logp).exp()

        prob_normalized = prob_normalized * wers
        if self.reduction == 'sum':
            loss = prob_normalized.sum()
        elif self.reduction == 'mean':
            loss = prob_normalized.mean()
        else:
            loss = k2.RaggedTensor(stream_path_shape, prob_normalized)
        return loss


def mwer_loss(lattice,
              ref_texts,
              nbest_scale=0.5,
              num_paths=200,
              temperature=1.0,
              use_double_scores=True,
              reduction: Literal['none', 'mean', 'sum'] = 'sum',
              ) -> Union[torch.Tensor, k2.RaggedTensor]:
    '''Compute the Minimum loss given a lattice and corresponding ref_texts.

    Args:
       lattice:
         An FsaVec with axes [utt][state][arc].
       ref_texts:
         It can be one of the following types:
           - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
           - An instance of :class:`k2.RaggedTensor`.
             Must have `num_axes == 2` and with dtype `torch.int32`.
       nbest_scale:
         Scale `lattice.score` before passing it to :func:`k2.random_paths`.
         A smaller value leads to more unique paths at the risk of being not
         to sample the path with the best score.''
       num_paths:
         Number of paths to **sample** from the lattice
         using :func:`k2.random_paths`.
       temperature:
         For long utterances, the dynamic range of scores will be too large
         and the posteriors will be mostly 0 or 1.
         To prevent this it might be a good idea to have an extra argument
         that functions like a temperature.
         We scale the logprobs by before doing the normalization.
       use_double_scores:
         True to use double precision floating point.
         False to use single precision.
       reduction:
         Specifies the reduction to apply to the output:
         'none' | 'sum' | 'mean'.
         'none': no reduction will be applied.
                 The returned 'loss' is a k2.RaggedTensor, with
                 loss.tot_size(0) == batch_size.
                 loss.tot_size(1) == total_num_paths_of_current_batch
                 If you want the MWER loss for each utterance, just do:
                 `loss_per_utt = loss.sum()`
                 Then loss_per_utt.shape[0] should be batch_size.
                 See more example usages in 'k2/python/tests/mwer_test.py'
         'sum': sum loss of each path over the whole batch together.
         'mean': divide above 'sum' by total num paths over the whole batch.
    Returns:
       Minimum Word Error Rate loss.
    '''
    assert reduction in ('none', 'mean', 'sum')
    m = MWERLoss(temperature, use_double_scores, reduction)
    return m(lattice, ref_texts, nbest_scale, num_paths)
