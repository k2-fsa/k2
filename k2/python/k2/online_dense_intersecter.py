# Copyright      2022  Xiaomi Corp.       (author: Wei Kang)
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

from typing import List
from typing import Tuple

import k2
import torch
import _k2

from k2 import Fsa
from _k2 import DecodeStateInfo
from .dense_fsa_vec import DenseFsaVec


class OnlineDenseIntersecter(object):
    def __init__(
        self,
        decoding_graph: Fsa,
        num_streams: int,
        search_beam: float,
        output_beam: float,
        min_active_states: int,
        max_active_states: int,
    ) -> None:
        """Create a new online intersecter object.
        Args:
          decoding_graph:
            The decoding graph used in this intersecter.
          num_streams:
            How many streams can this intersecter handle parallelly.
          search_beam:
            Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
            (less pruning). This is the default value; it may be modified by
            ``min_active_states`` and ``max_active_states``.
          output_beam:
            Pruning beam for the output of intersection (vs. best path);
            equivalent to kaldi's lattice-beam.  E.g. 8.
          min_active_states:
            Minimum number of FSA states that are allowed to be active on any
            given frame for any given intersection/composition task. This is
            advisory, in that it will try not to have fewer than this number
            active. Set it to zero if there is no constraint.
          max_active_states:
            Maximum number of FSA states that are allowed to be active on any
            given frame for any given intersection/composition task. This is
            advisory, in that it will try not to exceed that but may not always
            succeed. You can use a very large number if no constraint is needed.
        Examples:
          .. code-block:: python
            # create a ``OnlineDenseIntersecter`` which can handle 2 streams.
            intersecter = k2.OnlineDenseIntersecter(...,num_streams=2,...)
            # decode_states stores the info of decoding states for each
            # sequences, suppose there are 3 sequences.
            decode_states = [None] * 3
            # now, we want to decode first chunk of sequence 1 and sequence 2
            # gather their history decoding states from ``decode_states``
            current_decode_states = [decode_states[1], decode_states[2]]
            lattice, new_decode_states = intersecter.decode(
                dense_fsas_12,
                current_decode_states
            )
            # we can get the partial results from ``lattice`` here.
            # update the decoding states of sequence 1 and sequence 2
            decode_states[1] = new_decode_states[0]
            decode_states[2] = new_decode_states[1]
            # then, we want to decode first chunk of sequence 0 and second chunk
            # of sequence 1.
            # gather their history decoding states from ``decode_states``
            current_decode_states = [decode_states[0], decode_states[1]]
            lattice, new_decode_states = intersecter.decode(
                dense_fsas_01,
                current_decode_states
            )
            # we can get the partial results from ``lattice`` here.
            # update the decoding states of sequence 1 and sequence 2
            decode_states[0] = new_decode_states[0]
            decode_states[1] = new_decode_states[1]
            ...
        """
        self.decoding_graph = decoding_graph
        self.device = decoding_graph.device
        self.intersecter = _k2.OnlineDenseIntersecter(
            self.decoding_graph.arcs,
            num_streams,
            search_beam,
            output_beam,
            min_active_states,
            max_active_states,
        )

    def decode(
        self, dense_fsas: DenseFsaVec, decode_states: List[DecodeStateInfo]
    ) -> Tuple[Fsa, List[DecodeStateInfo]]:
        """Does intersection/composition for current chunk of nnet_output(given
        by a DenseFsaVec), sequences in every chunk may come from different
        sources.
        Args:
          dense_fsas:
            The neural-net output, with each frame containing the log-likes of
            each modeling unit.
          decode_states:
            A list of history decoding states for current batch of sequences,
            its length equals to ``dense_fsas.dim0()`` (i.e. batch size).
            Each element in ``decode_states`` belongs to the sequence at the
            corresponding position in current batch.
            For a new sequence(i.e. has no history states), just put ``None``
            at the corresponding position.
        Return:
          Return a tuple containing an Fsa and a List of new decoding states.
          The Fsa which has 3 axes(i.e. (batch, state, arc)) contains the output
          lattices. See the example in the constructor to get more info about
          how to use the list of new decoding states.
        """
        ragged_arc, arc_map, new_decode_states = self.intersecter.decode(
            dense_fsas.dense_fsa_vec, decode_states
        )
        out_fsa = k2.utils.fsa_from_unary_function_tensor(
            self.decoding_graph, ragged_arc, arc_map
        )
        return out_fsa, new_decode_states
