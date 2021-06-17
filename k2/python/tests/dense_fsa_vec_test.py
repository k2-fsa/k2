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
#  ctest --verbose -R dense_fsa_vec_test_py

import unittest

import k2
import torch


class TestDenseFsaVec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_dense_fsa_vec(self):
        for device in self.devices:
            log_prob = torch.arange(20, dtype=torch.float32,
                                    device=device).reshape(
                                        2, 5, 2).log_softmax(dim=-1)
            supervision_segments = torch.tensor(
                [
                    # seq_index, start_time, duration
                    [0, 0, 3],
                    [0, 1, 4],
                    [1, 0, 2],
                    [0, 2, 3],
                    [1, 3, 2],
                ],
                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_prob, supervision_segments)
            assert dense_fsa_vec.dim0() == 5, 'It should contain 5 segments'
            assert dense_fsa_vec.device == device
            assert dense_fsa_vec.duration.device == torch.device('cpu')
            assert torch.all(
                torch.eq(dense_fsa_vec.duration, supervision_segments[:, 2]))

            del dense_fsa_vec._duration
            assert torch.all(
                torch.eq(dense_fsa_vec.duration, supervision_segments[:, 2]))

            assert torch.allclose(dense_fsa_vec.scores[:3, 1:],
                                  log_prob[0][0:3])

            offset = 3 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 4, 1:],
                                  log_prob[0][1:5])

            offset += 4 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 2, 1:],
                                  log_prob[1][0:2])

            offset += 2 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 3, 1:],
                                  log_prob[0][2:5])

            offset += 3 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 2, 1:],
                                  log_prob[1][3:5])

            dense_fsa_vec.to('cpu')

    def test_duration(self):
        for device in self.devices:
            log_prob = torch.arange(20, dtype=torch.float32,
                                    device=device).reshape(
                                        2, 5, 2).log_softmax(dim=-1)

            supervision_segments = torch.tensor(
                [
                    # seq_index, start_time, duration
                    [0, 0, 3],
                    [0, 4, 2],  # exceed 1
                    [0, 3, 4],  # exceed 2
                    [1, 1, 7],  # exceed 3
                ],
                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_prob,
                                           supervision_segments,
                                           allow_truncate=3)
            assert torch.all(
                torch.eq(dense_fsa_vec.duration, torch.tensor([3, 1, 2, 4])))

            assert torch.allclose(dense_fsa_vec.scores[:3, 1:],
                                  log_prob[0][0:3])

            offset = 3 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 1, 1:],
                                  log_prob[0][4:5])

            offset += 1 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 2, 1:],
                                  log_prob[0][3:5])

            offset += 2 + 1
            assert torch.allclose(dense_fsa_vec.scores[offset:offset + 4, 1:],
                                  log_prob[1][1:5])

            dense_fsa_vec.to('cpu')


if __name__ == '__main__':
    unittest.main()
