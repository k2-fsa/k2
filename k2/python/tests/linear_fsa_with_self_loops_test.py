#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corporation      (authors: Fangjun Kuang)
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
#  ctest --verbose -R linear_fsa_self_loops_test_py

import torch
import k2
import unittest


class TestLinearFsa(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_single_fsa(self):
        for device in self.devices:
            labels = [2, 0, 0, 0, 5, 8]
            src = k2.linear_fsa(labels, device)
            dst = k2.linear_fsa_with_self_loops(src)
            assert src.device == dst.device
            expected_labels = [0, 2, 0, 5, 0, 8, 0, -1]
            assert dst.labels.tolist() == expected_labels

    def test_multiple_fsa(self):
        for device in self.devices:
            labels = [[2, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0], [1, 2],
                      [0, 0, 0, 3, 0, 2]]
            src = k2.linear_fsa(labels, device)
            dst = k2.linear_fsa_with_self_loops(src)
            assert src.device == dst.device
            expected_labels0 = [0, 2, 0, 5, 0, 8, 0, -1]
            expected_labels1 = [0, 1, 0, 2, 0, -1]
            expected_labels2 = [0, 3, 0, 2, 0, -1]
            expected_labels = expected_labels0 + expected_labels1 + expected_labels2  # noqa
            assert dst.labels.tolist() == expected_labels


if __name__ == '__main__':
    unittest.main()
