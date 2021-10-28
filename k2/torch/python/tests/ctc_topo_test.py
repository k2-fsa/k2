#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation      (authors: Wei Kang)
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
#  ctest --verbose -R ctc_topo_test_py

import unittest

import k2
import torch


class TestCtcTopo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        for device in self.devices:
            topo = k2.ctc_topo(3, False, device)
            expected_arcs = torch.tensor(
                [[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, -1],
                 [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, -1],
                 [2, 0, 0], [2, 1, 1], [2, 2, 2], [2, 3, 3], [2, 4, -1],
                 [3, 0, 0], [3, 1, 1], [3, 2, 2], [3, 3, 3], [3, 4, -1]],
                dtype=torch.int32,
                device=device)
            aux_label_ref = torch.tensor([
                0, 1, 2, 3, -1, 0, 0, 2, 3, -1, 0, 1, 0, 3, -1, 0, 1, 2, 0, -1
            ],
                                         dtype=torch.int32,
                                         device=device)

            assert torch.allclose(topo.scores, torch.zeros_like(topo.scores))
            assert torch.all(torch.eq(topo.arcs[:, :-1], expected_arcs))
            assert torch.all(torch.eq(topo.aux_labels, aux_label_ref))

    def test_simplified(self):
        for device in self.devices:
            topo = k2.ctc_topo(3, True, device)
            expected_arcs = torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 1],
                 [0, 2, 2], [0, 3, 3], [0, 4, -1], [1, 1, 1], [1, 0, 1],
                 [2, 2, 2], [2, 0, 2], [3, 3, 3], [3, 0, 3]],
                dtype=torch.int32,
                device=device)
            aux_label_ref = torch.tensor(
                [0, 1, 2, 3, 1, 2, 3, -1, 0, 0, 0, 0, 0, 0],
                dtype=torch.int32,
                device=device)

            assert torch.allclose(topo.scores, torch.zeros_like(topo.scores))
            assert torch.all(torch.eq(topo.arcs[:, :-1], expected_arcs))
            assert torch.all(torch.eq(topo.aux_labels, aux_label_ref))


if __name__ == '__main__':
    unittest.main()
