#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R compose_test_py

import unittest

import torch
import k2


class TestCompose(unittest.TestCase):

    def test_compose(self):
        s = '''
            0 1 11 1 1.0
            0 2 12 2 2.5
            1 3 -1 -1 0
            2 3 -1 -1 2.5
            3
        '''
        a_fsa = k2.Fsa.from_str(s, num_aux_labels=1).requires_grad_(True)

        s = '''
            0 1 1 1 1.0
            0 2 2 3 3.0
            1 2 3 2 2.5
            2 3 -1 -1 2.0
            3
        '''
        b_fsa = k2.Fsa.from_str(s, num_aux_labels=1).requires_grad_(True)

        ans = k2.compose(a_fsa, b_fsa, inner_labels='inner')
        ans = k2.connect(ans)

        ans = k2.create_fsa_vec([ans])

        scores = ans.get_tot_scores(log_semiring=True, use_double_scores=False)
        # The reference values for `scores`, `a_fsa.grad` and `b_fsa.grad`
        # are computed using GTN.
        # See https://bit.ly/3heLAJq
        assert scores.item() == 10
        scores.backward()
        assert torch.allclose(a_fsa.grad, torch.tensor([0., 1., 0., 1.]))
        assert torch.allclose(b_fsa.grad, torch.tensor([0., 1., 0., 1.]))

    def test_compose_inner_labels(self):
        s1 = '''
            0 1 1 2 0.1
            0 2 0 2 0.2
            1 3 3 5 0.3
            2 3 5 4 0.4
            3 4 3 3 0.5
            3 5 2 2 0.6
            4 6 -1 -1 0.7
            5 6 -1 -1 0.8
            6
        '''

        s2 = '''
            0 0 2 1 1
            0 1 4 3 2
            0 1 6 2 2
            0 2 -1 -1 0
            1 1 2 5 3
            1 2 -1 -1 4
            2
        '''

        # https://git.io/JqN2j
        fsa1 = k2.Fsa.from_str(s1, num_aux_labels=1)

        # https://git.io/JqNaJ
        fsa2 = k2.Fsa.from_str(s2, num_aux_labels=1)

        # https://git.io/JqNaT
        ans = k2.connect(k2.compose(fsa1, fsa2, inner_labels='phones'))

        assert torch.all(torch.eq(ans.labels, torch.tensor([0, 5, 2, -1])))
        assert torch.all(torch.eq(ans.phones, torch.tensor([2, 4, 2, -1])))
        assert torch.all(torch.eq(ans.aux_labels, torch.tensor([1, 3, 5, -1])))


# TODO(fangjun): add more tests for ragged attributes and test for CUDA.

if __name__ == '__main__':
    unittest.main()
