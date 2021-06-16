#!/usr/bin/env python3
#
# Copyright      2021  Mobvoi Inc.        (authors: Yaguang Hu)
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
#  ctest --verbose -R linear_fst_test_py

import unittest

import k2
import torch


class TestLinearFst(unittest.TestCase):

    def test_single_fst(self):
        labels = [2, 5, 8]
        aux_labels = [3, 6, 9]
        fst = k2.linear_fst(labels, aux_labels)
        assert len(fst.shape) == 2
        assert fst.shape[0] == len(labels) + 2, 'There should be 5 states'
        assert fst.aux_labels.shape[0] == len(aux_labels) + 1
        assert torch.all(
            torch.eq(fst.scores,
                     torch.zeros(len(labels) + 1, dtype=torch.float32)))
        assert torch.all(
            torch.eq(
                fst.arcs.values()[:, :-1],  # skip the last field `scores`
                torch.tensor([[0, 1, 2], [1, 2, 5], [2, 3, 8], [3, 4, -1]],
                             dtype=torch.int32)))
        assert torch.all(
            torch.eq(fst.aux_labels,
                     torch.tensor(aux_labels + [-1], dtype=torch.int32)))

    def test_fst_vec(self):
        labels = [
            [1, 3, 5],
            [2, 6],
            [8, 7, 9],
        ]
        aux_labels = [
            [2, 4, 6],
            [6, 2],
            [8, 7, 9],
        ]
        num_labels = sum([len(s) for s in labels])
        fst = k2.linear_fst(labels, aux_labels)
        assert len(fst.shape) == 3
        assert fst.shape[0] == 3, 'There should be 3 FSTs'
        assert torch.all(
            torch.eq(
                fst.scores,
                torch.zeros(num_labels + len(labels), dtype=torch.float32)))
        expected_aux_labels = [2, 4, 6, -1, 6, 2, -1, 8, 7, 9, -1]
        assert torch.all(
            torch.eq(fst.aux_labels,
                     torch.tensor(expected_aux_labels, dtype=torch.int32)))


if __name__ == '__main__':
    unittest.main()
