#!/usr/bin/env python3
#
# Copyright      2020-2021  Xiaomi Corporation (authors: Haowen Qiu
#                                                        Fangjun Kuang)
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
#  ctest --verbose -R remove_epsilon_test_py

import unittest

import torch
import k2


class TestRemoveEpsilonHost(unittest.TestCase):

    def test1(self):
        s = '''
            0 4 1 1
            0 1 1 1
            1 2 0 2
            1 3 0 3
            1 4 0 2
            2 7 0 4
            3 7 0 5
            4 6 1 2
            4 6 0 3
            4 8 1 3
            4 9 -1 2
            5 9 -1 4
            6 9 -1 3
            7 9 -1 5
            8 9 -1 6
            9
        '''
        fsa = k2.Fsa.from_str(s)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))

    def test_autograd(self):
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        src = k2.Fsa.from_str(s).requires_grad_(True)
        scores_copy = src.scores.detach().clone().requires_grad_(True)

        src.attr1 = "hello"
        src.attr2 = "k2"
        float_attr = torch.tensor([0.1, 0.2, 0.3],
                                  dtype=torch.float32,
                                  requires_grad=True)

        src.float_attr = float_attr.detach().clone().requires_grad_(True)
        src.int_attr = torch.tensor([1, 2, 3], dtype=torch.int32)
        src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50], [60, 70]])

        dest = k2.remove_epsilon(src)
        # arc map is [[1] [0 2] [2]]

        assert dest.attr1 == src.attr1
        assert dest.attr2 == src.attr2

        expected_int_attr = k2.RaggedTensor([[2], [1, 3], [3]])
        assert dest.int_attr == expected_int_attr

        expected_ragged_attr = k2.RaggedTensor([[30, 40, 50], [10, 20, 60, 70],
                                                [60, 70]])
        assert dest.ragged_attr == expected_ragged_attr

        expected_float_attr = torch.empty_like(dest.float_attr)
        expected_float_attr[0] = float_attr[1]
        expected_float_attr[1] = float_attr[0] + float_attr[2]
        expected_float_attr[2] = float_attr[2]

        assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

        expected_scores = torch.empty_like(dest.scores)
        expected_scores[0] = scores_copy[1]
        expected_scores[1] = scores_copy[0] + scores_copy[2]
        expected_scores[2] = scores_copy[2]

        assert torch.all(torch.eq(dest.scores, expected_scores))

        scale = torch.tensor([10, 20, 30]).to(float_attr)

        (dest.float_attr * scale).sum().backward()
        (expected_float_attr * scale).sum().backward()
        assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

        (dest.scores * scale).sum().backward()
        (expected_scores * scale).sum().backward()
        assert torch.all(torch.eq(src.scores.grad, scores_copy.grad))


class TestRemoveEpsilonDevice(unittest.TestCase):

    def test1(self):
        if not torch.cuda.is_available():
            return

        if not k2.with_cuda:
            return

        device = torch.device('cuda', 0)
        s = '''
            0 1 0 1 1
            1 2 0 2 1
            2 3 0 3 1
            3 4 4 4 1
            3 5 -1 5 1
            4 5 -1 6 1
            5
        '''
        fsa = k2.Fsa.from_str(s, num_aux_labels=1).to(device)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))

        # just make sure that it runs.
        dest2 = k2.remove_epsilon_and_add_self_loops(fsa)
        dest3 = k2.remove_epsilon(dest2)

        self.assertTrue(
            k2.is_rand_equivalent(dest,
                                  dest3,
                                  log_semiring,
                                  treat_epsilons_specially=False))
        self.assertFalse(
            k2.is_rand_equivalent(dest,
                                  dest2,
                                  log_semiring,
                                  treat_epsilons_specially=False,
                                  npath=10000))
        self.assertTrue(
            k2.is_rand_equivalent(dest,
                                  dest2,
                                  log_semiring,
                                  treat_epsilons_specially=True))

    def test_autograd(self):
        if not torch.cuda.is_available():
            return

        if not k2.with_cuda:
            return

        devices = [torch.device('cuda', 0)]
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)
            devices.append(torch.device('cuda', 1))

        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        for device in devices:
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            scores_copy = src.scores.detach().clone().requires_grad_(True)

            src.attr1 = "hello"
            src.attr2 = "k2"
            float_attr = torch.tensor([0.1, 0.2, 0.3],
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)

            src.float_attr = float_attr.detach().clone().requires_grad_(True)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50],
                                               [60, 70]]).to(device)

            dest = k2.remove_epsilon(src)
            # arc map is [[1] [0 2] [2]]

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            expected_int_attr = k2.RaggedTensor([[2], [1, 3], [3]]).to(device)
            assert dest.int_attr == expected_int_attr

            expected_ragged_attr = k2.RaggedTensor([[30, 40, 50],
                                                    [10, 20, 60, 70],
                                                    [60, 70]]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = float_attr[1]
            expected_float_attr[1] = float_attr[0] + float_attr[2]
            expected_float_attr[2] = float_attr[2]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = scores_copy[1]
            expected_scores[1] = scores_copy[0] + scores_copy[2]
            expected_scores[2] = scores_copy[2]

            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30]).to(float_attr)

            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()
            assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()
            assert torch.all(torch.eq(src.scores.grad, scores_copy.grad))

    def test_autograd_remove_epsilon_and_add_self_loops(self):
        if not torch.cuda.is_available():
            return

        if not k2.with_cuda:
            return

        devices = [torch.device('cuda', 0)]
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)
            devices.append(torch.device('cuda', 1))

        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        for device in devices:
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            scores_copy = src.scores.detach().clone().requires_grad_(True)

            src.attr1 = "hello"
            src.attr2 = "k2"
            float_attr = torch.tensor([0.1, 0.2, 0.3],
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)

            src.float_attr = float_attr.detach().clone().requires_grad_(True)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedTensor([[10, 20], [30, 40, 50],
                                               [60, 70]]).to(device)

            dest = k2.remove_epsilon_and_add_self_loops(src)
            # without add_self_loops, the arc map is [[1] [0 2] [2]]
            # with add_self_loops, the arc map is [[] [1] [0 2] [] [2]]

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            expected_int_attr = k2.RaggedTensor([[], [2], [1, 3], [],
                                                 [3]]).to(device)
            assert dest.int_attr == expected_int_attr

            expected_ragged_attr = k2.RaggedTensor([[], [30, 40, 50],
                                                    [10, 20, 60, 70], [],
                                                    [60, 70]]).to(device)
            assert dest.ragged_attr == expected_ragged_attr

            expected_float_attr = torch.empty_like(dest.float_attr)
            expected_float_attr[0] = 0
            expected_float_attr[1] = float_attr[1]
            expected_float_attr[2] = float_attr[0] + float_attr[2]
            expected_float_attr[3] = 0
            expected_float_attr[4] = float_attr[2]

            assert torch.all(torch.eq(dest.float_attr, expected_float_attr))

            expected_scores = torch.empty_like(dest.scores)
            expected_scores[0] = 0
            expected_scores[1] = scores_copy[1]
            expected_scores[2] = scores_copy[0] + scores_copy[2]
            expected_scores[3] = 0
            expected_scores[4] = scores_copy[2]

            assert torch.all(torch.eq(dest.scores, expected_scores))

            scale = torch.tensor([10, 20, 30, 40, 50]).to(float_attr)

            (dest.float_attr * scale).sum().backward()
            (expected_float_attr * scale).sum().backward()
            assert torch.all(torch.eq(src.float_attr.grad, float_attr.grad))

            (dest.scores * scale).sum().backward()
            (expected_scores * scale).sum().backward()
            assert torch.all(torch.eq(src.scores.grad, scores_copy.grad))


class TestRemoveEpsilonDeviceFillers(unittest.TestCase):
    ''' aim to test code relating to _filler attributes. '''

    def test1(self):
        if not torch.cuda.is_available():
            return

        if not k2.with_cuda:
            return

        device = torch.device('cuda', 0)
        s = '''
            0 1 0 1 1
            1 2 0 2 1
            2 3 0 3 1
            3 4 4 4 1
            3 5 -1 5 1
            4 5 -1 6 1
            5
        '''
        fsa = k2.Fsa.from_str(s, aux_label_names=['foo']).to(device)
        filler = 2
        fsa.foo_filler = filler
        print("Before removing epsilons: ", fsa)
        prop = fsa.properties
        self.assertFalse(prop & k2.fsa_properties.EPSILON_FREE)
        dest = k2.remove_epsilon(fsa)
        prop = dest.properties
        self.assertTrue(prop & k2.fsa_properties.EPSILON_FREE)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))

        print("After removing epsilons: ", dest)
        assert torch.where(dest.foo.values == filler)[0].numel() == 0


if __name__ == '__main__':
    unittest.main()
