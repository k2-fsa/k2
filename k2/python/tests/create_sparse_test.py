#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corp.       (authors: Fangjun Kuang)
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
#  ctest --verbose -R create_sparse_test_py

import unittest

import k2
import torch


class TestCreaseSparse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_create_sparse(self):
        s = '''
            0 1 10 0.1
            0 1 11 0.2
            1 2 20 0.3
            2 3 21 0.4
            2 3 24 0.5
            3 4 -1 0.6
            4
        '''

        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.phones = torch.tensor([10, 11, 20, 21, 24, -1],
                                      dtype=torch.int32,
                                      device=device)
            fsa.seqframes = torch.tensor([0, 0, 1, 2, 2, 3],
                                         dtype=torch.int32,
                                         device=device)
            fsa.requires_grad_(True)

            tensor = k2.create_sparse(rows=fsa.seqframes,
                                      cols=fsa.phones,
                                      values=fsa.scores,
                                      size=(6, 25),
                                      min_col_index=0)
            assert tensor.device == device
            assert tensor.is_sparse
            assert torch.allclose(tensor._indices()[0],
                                  fsa.seqframes[:-1].to(torch.int64))
            assert torch.allclose(tensor._indices()[1],
                                  fsa.phones[:-1].to(torch.int64))
            assert torch.allclose(tensor._values(), fsa.scores[:-1])
            assert tensor.requires_grad == fsa.requires_grad
            assert tensor.dtype == fsa.scores.dtype


if __name__ == '__main__':
    unittest.main()
