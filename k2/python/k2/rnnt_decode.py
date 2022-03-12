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
    def __init__(self, fsa: Fsa) -> None:
        self.fsa = fsa
        self.stream = _k2.create_rnnt_decoding_stream(fsa.arcs)
        self.device = fsa.device


class RnntDecodingStreams(object):
    def __init__(
        self, src_streams: List[RnntDecodingStream], config: RnntDecodingConfig
    ) -> None:
        self.src_streams = src_streams
        streams = [x.stream for x in src_streams]
        self.streams = _k2.RnntDecodingStreams(streams, config)
        self.num_streams = len(src_streams)

    def get_contexts(self) -> Tuple[RaggedShape, Tensor]:
        return self.streams.get_contexts()

    def advance(self, logprobs: Tensor) -> None:
        self.streams.advance(logprobs)

    def detach(self) -> List[RnntDecodingStream]:
        self.streams.detach()

    def format_output(self, num_frames: List[int]) -> Fsa:
        ragged_arcs, out_map = self.streams.format_output(num_frames)
        fsa = Fsa(ragged_arcs)

        # propagate attributes
        tensor_attr_info = dict()
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

            # set non_tensor_attrs
            for name, value in src.named_non_tensor_attr():
                setattr(fsa, name, value)

        for name, info in tensor_attr_info.items():
            values = list()
            start = 0
            for i in range(self.num_streams):
                src = self.src_streams[i].fsa
                device = self.src_streams[i].device
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
        return fsa
