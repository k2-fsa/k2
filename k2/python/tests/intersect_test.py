#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
#  ctest --verbose -R intersect_test_py

import unittest

import k2
import torch


class TestIntersect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_treat_epsilon_specially_false(self):
        for device in self.devices:
            # a_fsa recognizes `(0|1)2*`
            s1 = '''
                0 1 0 0.1
                0 1 1 0.2
                1 1 2 0.3
                1 2 -1 0.4
                2
            '''
            a_fsa = k2.Fsa.from_str(s1).to(device)
            a_fsa.requires_grad_(True)

            # b_fsa recognizes `1|2`
            s2 = '''
                0 1 1 1
                0 1 2 2
                1 2 -1 3
                2
            '''
            b_fsa = k2.Fsa.from_str(s2).to(device)
            b_fsa.requires_grad_(True)

            # fsa recognizes `1`
            fsa = k2.intersect(a_fsa, b_fsa, treat_epsilons_specially=False)
            assert len(fsa.shape) == 2
            actual_str = k2.to_str_simple(fsa)
            expected_str = '\n'.join(['0 1 1 1.2', '1 2 -1 3.4', '2'])
            assert actual_str.strip() == expected_str

            loss = fsa.scores.sum()
            (-loss).backward()
            # arc 1 and 3 of a_fsa are kept in the final intersected FSA
            assert torch.allclose(a_fsa.grad,
                                  torch.tensor([0, -1, 0, -1]).to(a_fsa.grad))

            # arc 0 and 2 of b_fsa are kept in the final intersected FSA
            assert torch.allclose(b_fsa.grad,
                                  torch.tensor([-1, 0, -1]).to(b_fsa.grad))

            # if any of the input FSA is an FsaVec,
            # the output FSA is also an FsaVec.
            a_fsa.scores.grad = None
            b_fsa.scores.grad = None
            a_fsa = k2.create_fsa_vec([a_fsa])
            fsa = k2.intersect(a_fsa, b_fsa, treat_epsilons_specially=False)
            assert len(fsa.shape) == 3

    def test_treat_epsilon_specially_true(self):
        # this version works only on CPU and requires
        # arc-sorted inputs
        # a_fsa recognizes `(1|3)?2*`
        s1 = '''
            0 1 3 0.0
            0 1 1 0.2
            0 1 0 0.1
            1 1 2 0.3
            1 2 -1 0.4
            2
        '''
        a_fsa = k2.Fsa.from_str(s1)
        a_fsa.requires_grad_(True)

        # b_fsa recognizes `1|2|5`
        s2 = '''
            0 1 5 0
            0 1 1 1
            0 1 2 2
            1 2 -1 3
            2
        '''
        b_fsa = k2.Fsa.from_str(s2)
        b_fsa.requires_grad_(True)

        # fsa recognizes 1|2
        fsa = k2.intersect(k2.arc_sort(a_fsa), k2.arc_sort(b_fsa))
        assert len(fsa.shape) == 2
        actual_str = k2.to_str_simple(fsa)
        expected_str = '\n'.join(
            ['0 1 0 0.1', '0 2 1 1.2', '1 2 2 2.3', '2 3 -1 3.4', '3'])
        assert actual_str.strip() == expected_str

        loss = fsa.scores.sum()
        (-loss).backward()
        # arc 1, 2, 3, and 4 of a_fsa are kept in the final intersected FSA
        assert torch.allclose(a_fsa.grad,
                              torch.tensor([0, -1, -1, -1, -1]).to(a_fsa.grad))

        # arc 1, 2, and 3 of b_fsa are kept in the final intersected FSA
        assert torch.allclose(b_fsa.grad,
                              torch.tensor([0, -1, -1, -1]).to(b_fsa.grad))

        # if any of the input FSA is an FsaVec,
        # the output FSA is also an FsaVec.
        a_fsa.scores.grad = None
        b_fsa.scores.grad = None
        a_fsa = k2.create_fsa_vec([a_fsa])
        fsa = k2.intersect(k2.arc_sort(a_fsa), k2.arc_sort(b_fsa))
        assert len(fsa.shape) == 3


# TODO(fangjun): add more tests for ragged attributes

if __name__ == '__main__':
    unittest.main()
