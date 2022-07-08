#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corporation      (authors: Wei Kang)
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

# To run this single test, use
#
#  ctest --verbose -R online_dense_intersecter_test_py

import unittest

import k2
import torch


class TestOnlineDenseIntersecter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test(self):
        for device in self.devices:
            vocab_size = 10
            num_streams = 2
            search_beam = 10
            output_beam = 2
            min_active_states = 1
            max_active_states = 10
            decoding_graph = k2.ctc_topo(vocab_size - 1, device=device)
            decoding_graph = k2.Fsa.from_fsas([decoding_graph])

            intersector = k2.OnlineDenseIntersecter(
                decoding_graph=decoding_graph,
                num_streams=num_streams,
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
            )

            num_chunks = 3
            chunk_size = 5

            decode_states = [None] * num_streams

            for i in range(num_chunks):
                logits = torch.randn(
                    (num_streams, chunk_size, vocab_size), device=device
                )
                supervision_segments = torch.tensor(
                    # seq_index, start_time, duration
                    [[i, 0, chunk_size] for i in range(num_streams)],
                    dtype=torch.int32,
                )
                dense_fsa_vec = k2.DenseFsaVec(logits, supervision_segments)
                ofsa, decode_states = intersector.decode(
                    dense_fsa_vec, decode_states
                )
                print(ofsa)


if __name__ == "__main__":
    unittest.main()
