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
from k2 import RaggedShape
from k2 import RaggedTensor
from torch import Tensor
from .ops import index_select
from _k2 import RnntDecodingConfig


class RnntDecodingStream(object):
    """Create a new rnnt decoding stream.

    Every sequence(wave data) need a decoding stream, this function is expected
    to be called when a new sequence comes. We support different decoding graphs
    for different streams.

    Args:
      graph:
        The decoding graph used in this stream.

    Returns:
      A rnnt decoding stream object, which will be combined into
      `RnntDecodingStreams` to do decoding together with other
      sequences in parallel.
    """
    def __init__(self, fsa: Fsa) -> None:
        self.fsa = fsa
        self.stream = _k2.create_rnnt_decoding_stream(fsa.arcs)
        self.device = fsa.device

    """Return a string representation of this object

    For visualization and debug only.
    """
    def __str__(self) -> str:
        return f"{self.stream}, device : {self.device}\n"


class RnntDecodingStreams(object):
    """
    Combines multiple RnntDecodingStream objects to create a RnntDecodingStreams
    object, then all these RnntDecodingStreams can do decoding in parallel.

    Args:
      src_streams:
        A list of RnntDecodingStream object to be combined.
      config:
        A configuration object which contains decoding parameters like
        `vocab-size`, `decoder_history_len`, `beam`, `max_states`,
        `max_contexts` etc.

    Returns:
      Return a RnntDecodingStreams object.
    """
    def __init__(
        self, src_streams: List[RnntDecodingStream], config: RnntDecodingConfig
    ) -> None:
        assert len(src_streams) > 0
        self.num_streams = len(src_streams)
        self.src_streams = src_streams
        self.device = self.src_streams[0].device
        streams = [x.stream for x in self.src_streams]
        self.streams = _k2.RnntDecodingStreams(streams, config)

    '''Return a string representation of this object

    For visualization and debug only.
    '''
    def __str__(self) -> str:
        s = f"num_streams : {self.num_streams}\n"
        for i in range(self.num_streams):
            s += f"stream[{i}] : {self.src_streams[i]}"
        return s

    """
    This function must be called prior to evaluating the joiner network
    for a particular frame.  It tells the calling code which contexts
    it must evaluate the joiner network for.

    Returns:
      Return a two elements tuple containing a RaggedShape and a tensor.
      shape:
        A RaggedShape with 2 axes, representing [stream][context].
      contexts:
        A tensor of shape [tot_contexts][decoder_history_len], where
        tot_contexts == shape->TotSize(1) and decoder_history_len comes from
        the config, it represents the number of symbols in the context of the
        decode network (assumed to be finite). It contains the token ids
        into the vocabulary(i.e. `0 <= value < vocab_size`).
    """
    def get_contexts(self) -> Tuple[RaggedShape, Tensor]:
        return self.streams.get_contexts()

    """
    Advance decoding streams by one frame.

    Args:
      logprobs:
        A tensor of shape [tot_contexts][num_symbols], containing log-probs of
        symbols given the contexts output by `get_contexts()`. Will satisfy
        logprobs.Dim0() == shape.TotSize(1).
    """
    def advance(self, logprobs: Tensor) -> None:
        self.streams.advance(logprobs)

    """
    Detach the RnntDecodingStreams. It will update the decoding states and store
    the decoding results currently got for each of the individual streams.

    Note: We can not decode with this object anymore after calling detach().
    """
    def detach(self) -> None:
        self.streams.detach()

    """
    Generate the lattice Fsa currently got.

    Note: The attributes of the generated lattice is a union of the attributes
          of all the decoding graphs. For example, a streams contains three
          individual stream, each stream has its own decoding graphs, graph[0]
          has attributes attr1, attr2; graph[1] has attributes attr1, attr3;
          graph[2] has attributes attr3, attr4; then the generated lattice has
          attributes attr1, attr2, attr3, attr4.

    Args:
      num_frames:
        A List containing the number of frames we want to gather for each stream
        (note: the frames we have ever received for the corresponding stream).
        It MUST satisfy `len(num_frames) == self.num_streams`.
    Returns:
      Return the lattice Fsa with all the attributes propagated. The returned
      Fsa has 3 axes with `fsa.dim0==self.num_streams`.
    """
    def format_output(self, num_frames: List[int]) -> Fsa:
        assert len(num_frames) == self.num_streams

        ragged_arcs, out_map = self.streams.format_output(num_frames)
        fsa = Fsa(ragged_arcs)

        # propagate attributes
        tensor_attr_info = dict()
        # gather the attributes info of all the decoding graphs,
        for i in range(self.num_streams):
            src = self.src_streams[i].fsa
            for name, value in src.named_tensor_attr(include_scores=False):
                if name not in tensor_attr_info:
                    filler = 0.0
                    if isinstance(value, Tensor):
                        filler = float(src.get_filler(name))
                        dtype = value.dtype
                        tensor_type = "Tensor"
                    else:
                        assert isinstance(value, k2.RaggedTensor)
                        # Only integer types ragged attributes are supported now
                        assert value.dtype == torch.int32
                        assert value.num_axes == 2
                        dtype = torch.int32
                        tensor_type = "RaggedTensor"
                    tensor_attr_info[name] = {
                        "filler": filler,
                        "dtype": dtype,
                        "tensor_type": tensor_type,
                    }
        # combine the attributes propagating from different decoding graphs
        for name, info in tensor_attr_info.items():
            values = list()
            start = 0
            for i in range(self.num_streams):
                src = self.src_streams[i].fsa
                device = self.device
                num_arcs = fsa[i].num_arcs
                arc_map = out_map[start:start + num_arcs]
                start = start + num_arcs
                if hasattr(src, name):
                    value = getattr(src, name)
                    if info["tensor_type"] == "Tensor":
                        assert isinstance(value, Tensor)
                        new_value = index_select(
                            value, arc_map, default_value=filler
                        )
                    else:
                        assert isinstance(value, RaggedTensor)
                        # Only integer types ragged attributes are supported now
                        assert value.num_axes == 2
                        assert value.dtype == torch.int32
                        new_value, _ = value.index(
                            arc_map, axis=0, need_value_indexes=False
                        )
                else:
                    if info["tensor_type"] == "Tensor":
                        # fill with filler value
                        new_value = torch.tensor(
                            [filler] * num_arcs,
                            dtype=info["dtype"],
                            device=device,
                        )
                    else:
                        # fill with empty RaggedTensor
                        new_value = RaggedTensor(
                            torch.empty(
                                (num_arcs, 0),
                                dtype=info["dtype"],
                                device=device,
                            )
                        )
                values.append(new_value)
            if info["tensor_type"] == "Tensor":
                new_value = torch.cat(values)
            else:
                new_value = k2.ragged.cat(values, axis=0)
            setattr(fsa, name, new_value)

        # set non_tensor_attrs
        for i in range(self.num_streams):
            src = self.src_streams[i].fsa
            for name, value in src.named_non_tensor_attr():
                setattr(fsa, name, value)

        return fsa
