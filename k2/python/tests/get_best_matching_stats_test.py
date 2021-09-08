#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation    (author: Wei Kang)
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
#  ctest --verbose -R get_best_matching_stats_test_py

import unittest

import k2
import torch


class TestGetBestMatchingStats(unittest.TestCase):

    def test(self):
        s = '[ [ [ 5 1 4 6 ] [ 5 1 2 6 ] [ 5 3 4 6 ] ] ]'
        tokens = k2.RaggedTensor(s)
        scores = torch.tensor([1, 2, 3, 4, 5, 7, 8, 6, 0, 0, 0, 0],
                              dtype=torch.float32)
        counts = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                              dtype=torch.int32)
        eos = 6
        min_token = 1
        max_token = 6
        max_order = 2
        mean, var, counts_out, ngram_order = k2.get_best_matching_stats(
            tokens, scores, counts, eos, min_token, max_token, max_order)

        mean_ref = torch.tensor([3, 4.5, 3, 4, 3, 4.5, 4.5, 5, 3, 4.5, 3, 4],
                                dtype=torch.float32)
        var_ref = torch.tensor([4, 6.25, 0, 0, 4, 6.25, 5.25, 1, 4, 5.25, 0, 0],
                               dtype=torch.float32)
        counts_out_ref = torch.tensor([2, 2, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1],
                                      dtype=torch.int32)
        ngram_order_ref = torch.tensor([2, 2, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2],
                                       dtype=torch.int32)
        assert torch.allclose(mean, mean_ref)
        assert torch.allclose(var, var_ref)
        assert torch.all(torch.eq(counts_out, counts_out_ref))
        assert torch.all(torch.eq(ngram_order, ngram_order_ref))


if __name__ == '__main__':
    unittest.main()
