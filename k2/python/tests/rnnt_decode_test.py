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
#  ctest --verbose -R rnnt_decode_test_py

import unittest

import k2
import torch


class TestRnntDecode(unittest.TestCase):
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
            fsa1 = k2.ctc_topo(5, device=device)
            fsa1.attr1 = torch.tensor([1] * fsa1.num_arcs, device=device)

            stream1 = k2.RnntDecodingStream(fsa1)

            fsa2 = k2.trivial_graph(3, device=device)
            fsa2.attr1 = torch.tensor([2] * fsa2.num_arcs, device=device)
            fsa2.attr2 = torch.tensor([22] * fsa2.num_arcs, device=device)

            stream2 = k2.RnntDecodingStream(fsa2)

            fsa3 = k2.ctc_topo(3, modified=True, device=device)
            fsa3.attr3 = k2.RaggedTensor(
                torch.ones((fsa3.num_arcs, 2), dtype=torch.int32, device=device)
                * 3
            )

            stream3 = k2.RnntDecodingStream(fsa3)

            config = k2.RnntDecodingConfig(10, 2, 3.0, 3, 3)
            streams = k2.RnntDecodingStreams(
                [stream1, stream2, stream3], config
            )

            for i in range(5):
                shape, context = streams.get_contexts()
                logprobs = torch.randn(
                    (context.shape[0], 10), dtype=torch.float32, device=device
                )
                streams.advance(logprobs)

            streams.terminate_and_flush_to_streams()
            ofsa = streams.format_output([3, 4, 5])
            print(ofsa)


if __name__ == "__main__":
    unittest.main()
