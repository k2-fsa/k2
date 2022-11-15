# Copyright (c)  2022  Xiaomi Corporation (authors: Liyong Guo)

from typing import List, Union

import torch
import k2


class MWERLoss(torch.nn.Module):
    '''Minimum Word Error Rate Loss compuration in k2.

    See equation 2 of https://arxiv.org/pdf/2106.02302.pdf about its definition.
    '''

    def __init__(self,
                 temperature: float = 1.0,
                 use_double_scores: bool = True):
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
        '''

        super().__init__()
        self.temperature = temperature
        self.use_double_scores = use_double_scores

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
        wers = -tot_scores

        # Group each log_prob into [path][arc]
        ragged_nbest_logp = k2.RaggedTensor(nbest.kept_path.shape,
                                            nbest.fsa.scores)
        # Get the probalitity of each path, in log format,
        # with shape [stream][path].
        path_logp = ragged_nbest_logp.sum() / self.temperature
        ragged_path_prob = k2.RaggedTensor(nbest.shape, path_logp.exp())

        # Normalize prob of each path.
        den_prob = ragged_path_prob.sum()
        den_logp = torch.index_select(
            den_prob.log(), 0, ragged_path_prob.shape.row_ids(1))
        prob_normalized = (path_logp - den_logp).exp()

        loss = (prob_normalized * wers).sum()
        return loss


def mwer_loss(
        lattice,
        ref_texts,
        nbest_scale=0.5,
        num_paths=200,
        temperature=1.0,
        use_double_scores=True) -> torch.Tensor:
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
    Returns:
       Minimum Word Error Rate loss.
    '''
    m = MWERLoss(temperature, use_double_scores)
    return m(lattice, ref_texts, nbest_scale, num_paths)
