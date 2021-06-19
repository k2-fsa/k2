#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
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
#  ctest --verbose -R  multi_gpu_test_py

import unittest

import k2
import torch


class TestMultiGPU(unittest.TestCase):

    def _test_ragged(self):
        if torch.cuda.is_available() is False:
            print('skip it since CUDA is not available')
            return
        if torch.cuda.device_count() < 2:
            print('skip it since number of GPUs is 1')
            return

        if not k2.with_cuda:
            return

        device0 = torch.device('cuda', 0)
        device1 = torch.device('cuda', 1)

        torch.cuda.set_device(device1)

        r0 = k2.RaggedInt('[ [[0] [1]] ]').to(device0)
        r1 = k2.RaggedInt('[ [[0] [1]] ]').to(device1)

        assert torch.cuda.current_device() == 1

        r0 = k2.ragged.remove_axis(r0, 0)
        r1 = k2.ragged.remove_axis(r1, 0)

        expected_r0 = k2.RaggedInt('[[0] [1]]').to(device0)
        expected_r1 = k2.RaggedInt('[[0] [1]]').to(device1)

        assert torch.all(torch.eq(r0.row_splits(1), expected_r0.row_splits(1)))
        assert torch.all(torch.eq(r1.row_splits(1), expected_r1.row_splits(1)))

        assert torch.all(torch.eq(r0.row_ids(1), expected_r0.row_ids(1)))
        assert torch.all(torch.eq(r1.row_ids(1), expected_r1.row_ids(1)))

        assert r0.num_elements() == expected_r0.num_elements()
        assert r1.num_elements() == expected_r1.num_elements()

        try:
            # will throw an exception because they two are not on
            # the same device
            assert torch.all(
                torch.eq(r0.row_splits(1), expected_r1.row_splits(1)))
        except RuntimeError as e:
            print(e)

        assert torch.cuda.current_device() == 1

    def test_fsa(self):
        if torch.cuda.is_available() is False:
            print('skip it since CUDA is not available')
            return
        if torch.cuda.device_count() < 2:
            print('skip it since number of GPUs is 1')
            return

        if not k2.with_cuda:
            return

        device0 = torch.device('cuda', 0)
        device1 = torch.device('cuda', 1)

        torch.cuda.set_device(device1)

        s = '''
            0 1 1 0.1
            1 2 -1 0.2
            2
        '''
        fsa0 = k2.Fsa.from_str(s).to(device0).requires_grad_(True)
        fsa1 = k2.Fsa.from_str(s).to(device1).requires_grad_(True)

        fsa0 = k2.create_fsa_vec([fsa0, fsa0])
        fsa1 = k2.create_fsa_vec([fsa1, fsa1])

        tot_scores0 = fsa0.get_forward_scores(True, True)
        (tot_scores0[0] * 2 + tot_scores0[1]).backward()

        tot_scores1 = fsa1.get_forward_scores(True, True)
        (tot_scores1[0] * 2 + tot_scores1[1]).backward()


if __name__ == '__main__':
    unittest.main()
