# Copyright      2022  Xiaomi Corp.
"""
This file contains various functions for CTC decoding.
"""

from typing import List

import torch

from .autograd import intersect_dense_pruned
from .dense_fsa_vec import DenseFsaVec
from .fsa import Fsa
from .fsa_algo import shortest_path
from .ragged import RaggedTensor


def get_lattice(
    log_prob: torch.Tensor,
    log_prob_len: torch.Tensor,
    decoding_graph: Fsa,
    search_beam: float = 20,
    output_beam: float = 8,
    min_active_states: int = 30,
    max_active_states: int = 10000,
    subsampling_factor: int = 1,
) -> Fsa:
    """Get the decoding lattice from a decoding graph and  log_softmax output.
    Args:
      log_prob:
        Output from a log_softmax layer of shape ``(N, T, C)``.
      log_prob_len:
        A tensor of shape ``(N,)`` containing number of valid frames from
        ``log_prob`` before padding.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an ``HLG`` or an ``H``.
        You can use :func:`ctc_topo` to build an ``H``.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    assert log_prob.ndim == 3, log_prob.shape
    assert log_prob_len.ndim == 1, log_prob_len.shape
    assert log_prob.size(0) == log_prob_len.size(0), (
        log_prob.shape,
        log_prob_len.shape,
    )

    batch_size = log_prob.size(0)

    supervision_segment = (
        torch.stack(
            [
                torch.arange(batch_size),
                torch.zeros(batch_size),
                log_prob_len.cpu(),
            ],
        )
        .t()
        .to(torch.int32)
    )

    dense_fsa_vec = DenseFsaVec(
        log_prob,
        supervision_segment,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


def one_best_decoding(
    lattice: Fsa,
    use_double_scores: bool = True,
) -> Fsa:
    """Get the best path from a lattice.

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Return:
      An FsaVec containing linear paths.
    """
    best_path = shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path


def get_aux_labels(best_paths: Fsa) -> List[List[int]]:
    """Extract aux_labels from the best-path FSAs and remove 0s and -1s.
    Args:
      best_paths:
        An Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of `shortest_path` (otherwise the returned values won't
        be meaningful).

    TODO:
      Also return timestamps of each label.

    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    return aux_labels.tolist()
