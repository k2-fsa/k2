#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang
#                                                   Wei kang)
#                2021  Mobvoi Inc. (authors: Yaguang Hu)
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
#  ctest --verbose -R ragged_ops_test_py

import unittest

import random
import torch
import _k2
import k2


class TestRaggedOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

        cls.dtypes = [torch.int32, torch.float32, torch.float64]

    def test_remove_axis_ragged_array(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [3 0 ] [ 2 ] ] ]
        '''
        for device in self.devices:
            src = k2.RaggedTensor(s, device=device)

            ans = src.remove_axis(0)
            assert ans == k2.RaggedTensor('[ [ 1 2 ] [ 0 ] [ 3 0 ] [ 2 ] ]',
                                          device=device)

            ans = src.remove_axis(1)
            assert ans == k2.RaggedTensor('[ [ 1 2 0 ] [ 3 0 2 ] ]',
                                          device=device)

    def test_remove_axis_ragged_shape(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [[x x] [] [x]] [[] [x x] [x x x] [x]] ]')
            shape = shape.to(device)

            ans = shape.remove_axis(0)
            expected = k2.RaggedShape(
                '[[x x] [] [x] [] [x x] [x x x] [x]]').to(device)
            assert ans == expected

            ans = shape.remove_axis(1)
            expected = k2.RaggedShape('[[x x x] [x x x x x x]]').to(device)
            assert ans == expected

            ans = shape.remove_axis(2)
            expected = k2.RaggedShape('[[x x x] [x x x x]]').to(device)
            assert ans == expected

    def test_tolist(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [ 3 0 ] [ 2 ] ] ]
        '''
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(s, dtype).to(device)

                ans = src.remove_axis(0)
                assert ans.tolist() == [[1, 2], [0], [3, 0], [2]]

    def test_pad(self):
        s = '''
            [ [ 1 2 ] [ 3 ] [ ] [ 4 5 6 ] [ 7 8 9 10 ] ]
        '''
        for device in self.devices:
            src = k2.RaggedTensor(s).to(device)
            value = random.randint(0, 1000)
            ans = src.pad('constant', value)
            expected = torch.tensor(
                [[1, 2, value, value],
                 [3, value, value, value],
                 [value, value, value, value],
                 [4, 5, 6, value],
                 [7, 8, 9, 10]], dtype=torch.int32, device=device)

            assert torch.all(torch.eq(ans, expected))

    def test_pad_empty(self):
        s = '''
            [ [ ] [ ] [ ] ]
        '''
        for device in self.devices:
            src = k2.RaggedTensor(s).to(device)
            value = random.randint(0, 1000)
            ans = src.pad('constant', value)
            assert ans.numel() == 0
            assert ans.size() == (3, 0)

    def test_pad_float(self):
        s = '''
            [ [ 1.0 2.0 ] [ 3.0 ] [ ] [ 4.0 5.0 6.0 ] [ 7.0 8.0 9.0 10.0 ] ]
        '''
        for device in self.devices:
            src = k2.RaggedTensor(s).to(device)
            value = random.random() * 10
            ans = src.pad('constant', value)
            expected = torch.tensor(
                [[1.0, 2.0, value, value],
                 [3.0, value, value, value],
                 [value, value, value, value],
                 [4.0, 5.0, 6.0, value],
                 [7.0, 8.0, 9.0, 10.0]], dtype=torch.float32, device=device)
            assert torch.allclose(ans, expected)

    def test_pad_replicate(self):
        s = '''
            [ [1 2] [10] [] [3 5 8 9] ]
        '''
        for device in self.devices:
            for dtype in self.dtypes:
                value = random.randint(0, 1000)
                src = k2.RaggedTensor(s, dtype).to(device)
                padded = src.pad('replicate', value)
                expected = torch.tensor(
                    [[1, 2, 2, 2], [10, 10, 10, 10],
                     [value, value, value, value], [3, 5, 8, 9]],).to(padded)
                assert torch.allclose(padded, expected)

    def test_remove_values_leq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(s, dtype=dtype, device=device)

                ans = src.remove_values_leq(0)
                assert ans == k2.RaggedTensor(
                    '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]',
                    device=device,
                    dtype=dtype)

                ans = src.remove_values_leq(1)
                assert ans == k2.RaggedTensor('[ [ 2 ] [ 3 2 ] [ 8 6 ] [ ] ]',
                                              dtype=dtype,
                                              device=device)

                ans = src.remove_values_leq(6)
                assert ans == k2.RaggedTensor('[ [ ] [ ] [ 8 ] [ ] ]',
                                              device=device,
                                              dtype=dtype)

                ans = src.remove_values_leq(8)
                assert ans == k2.RaggedTensor('[ [ ] [ ] [ ] [ ] ]',
                                              dtype=dtype,
                                              device=device)

    def test_remove_values_eq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(s, device=device, dtype=dtype)

                ans = src.remove_values_eq(0)
                assert ans == k2.RaggedTensor(
                    '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]',
                    device=device,
                    dtype=dtype)

                ans = src.remove_values_eq(1)
                assert ans == k2.RaggedTensor(
                    '[ [ 2 0 ] [ 3 0 2 ] [ 0 8 0 6 0 ] [ 0 ] ]',
                    device=device,
                    dtype=dtype)

                ans = src.remove_values_eq(6)
                assert ans == k2.RaggedTensor(
                    '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 8 0 0 ] [ 0 ] ]',
                    device=device,
                    dtype=dtype)

                ans = src.remove_values_eq(8)
                assert ans == k2.RaggedTensor(
                    '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 0 6 0 ] [ 0 ] ]',
                    device=device,
                    dtype=dtype)

    def test_normalize_scores_use_log_non_zero_stride(self):
        s = '''
            [ [1 -1 0] [2 10] [] [3] [5 8] ]
        '''
        for device in self.devices:
            for dtype in [torch.float32, torch.float64]:
                src = k2.RaggedTensor(s, dtype).to(device)
                saved = src.values.clone().detach()
                saved.requires_grad_(True)
                src.requires_grad_(True)

                ans = src.normalize(use_log=True)

                scale = torch.arange(ans.numel(), device=device)

                # the stride of grad is not 0
                (ans.values * scale).sum().backward()

                expected = saved.new_zeros(*ans.values.shape)

                normalizer = saved[:3].exp().sum().log()
                expected[:3] = saved[:3] - normalizer

                normalizer = saved[3:5].exp().sum().log()
                expected[3:5] = saved[3:5] - normalizer

                expected[5] = 0  # it has only one entry

                normalizer = saved[6:8].exp().sum().log()
                expected[6:8] = saved[6:8] - normalizer

                self.assertTrue(torch.allclose(expected, ans.values))
                (expected * scale).sum().backward()

                self.assertTrue(torch.allclose(saved.grad, src.grad))

    def test_normalize_scores_use_log_zero_stride(self):
        s = '''
            [ [1 3 5] [2 -1] [] [3] [5 2] ]
        '''
        for device in self.devices:
            for dtype in [torch.float32, torch.float64]:
                src = k2.RaggedTensor(s, dtype).to(device)
                saved = src.values.clone().detach()
                saved.requires_grad_(True)
                src.requires_grad_(True)

                ans = src.normalize(use_log=True)

                # the stride of grad is 0
                ans.values.sum().backward()

                expected = saved.new_zeros(*ans.values.shape)

                normalizer = saved[:3].exp().sum().log()
                expected[:3] = saved[:3] - normalizer

                normalizer = saved[3:5].exp().sum().log()
                expected[3:5] = saved[3:5] - normalizer

                expected[5] = 0  # it has only one entry

                normalizer = saved[6:8].exp().sum().log()
                expected[6:8] = saved[6:8] - normalizer

                self.assertTrue(torch.allclose(expected, ans.values))
                expected.sum().backward()

                self.assertTrue(torch.allclose(saved.grad, src.grad))

    def test_normalize_scores_use_log_from_shape(self):
        s = '''
            0 1 1 0.
            0 1 2 0.
            0 1 3 0.
            1 2 4 0.
            1 2 5 0.
            2 3 -1 0.
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            scores = torch.arange(fsa.scores.numel(),
                                  dtype=torch.float32,
                                  device=device)
            scores.requires_grad_(True)

            ragged_scores = k2.RaggedTensor(fsa.arcs.shape(), scores)
            assert ragged_scores.requires_grad is True

            normalized_scores = ragged_scores.normalize(use_log=True)
            assert normalized_scores.requires_grad is True

            fsa.scores = normalized_scores.values
            assert fsa.scores.requires_grad is True

            # arcs leaving state 0
            self.assertAlmostEqual(fsa.scores[:3].exp().sum().item(),
                                   1.0,
                                   places=6)

            # arcs leaving state 1
            self.assertAlmostEqual(fsa.scores[3:5].exp().sum().item(),
                                   1.0,
                                   places=6)

            # arcs leaving state 2
            self.assertAlmostEqual(fsa.scores[5].exp().sum().item(),
                                   1.0,
                                   places=6)

    def test_normalize_scores(self):
        for device in self.devices:
            for dtype in [torch.float32, torch.float64]:
                s = '''
                    [ [1 3 5] [2 -1] [] [3] [5 2] ]
                '''
                src = k2.RaggedTensor(s, dtype=dtype).to(device)
                saved = src.values

                ans = src.normalize(use_log=False)
                expected = saved.new_zeros(*ans.values.shape)
                expected[:3] = saved[:3] / saved[:3].sum()
                expected[3:5] = saved[3:5] / saved[3:5].sum()
                expected[5] = 1
                expected[6:8] = saved[6:8] / saved[6:8].sum()

                assert torch.allclose(ans.values, expected)

    def test_sum_per_sublist(self):
        s = '''
            0 1 1 0.
            0 1 2 0.
            0 1 3 0.
            1 2 4 0.
            1 2 5 0.
            2 3 -1 0.
            3
        '''
        for device in self.devices:
            fsa = k2.Fsa.from_str(s).to(device)
            scores = torch.randn_like(fsa.scores)
            fsa.set_scores_stochastic_(scores)
            ragged = k2.RaggedTensor(fsa.arcs.shape(), fsa.scores.exp())
            normalized_scores = ragged.sum()
            assert normalized_scores.numel() == fsa.arcs.dim0()

            assert torch.allclose(
                normalized_scores[:-1],
                torch.ones(normalized_scores.numel() - 1, device=device))

            # the final state has no leaving arcs
            assert normalized_scores[-1].item() == 0

    def test_cat(self):
        for device in self.devices:
            for dtype in self.dtypes:
                ragged1 = k2.RaggedTensor('[ [1 2 3] [] [4 5] ]',
                                          dtype).to(device)
                ragged2 = k2.RaggedTensor('[ [] [10 20] [30] [40 50] ]',
                                          dtype).to(device)
                ragged = k2.ragged.cat([ragged1, ragged2], axis=0)
                assert ragged == k2.RaggedTensor(
                    '[ [ 1 2 3 ] [ ] [ 4 5 ] [ ] [ 10 20 ] [ 30 ] [ 40 50 ] ]',
                    dtype=dtype,
                    device=device)

    def test_cat_axis1(self):
        for device in self.devices:
            for dtype in self.dtypes:
                ragged1 = k2.RaggedTensor('[ [1 2 3] [] [4 5] ]',
                                          dtype).to(device)
                ragged2 = k2.RaggedTensor('[ [10 20] [8] [9 10] ]',
                                          dtype).to(device)
                ragged = k2.ragged.cat([ragged1, ragged2], axis=1)
                assert ragged == k2.RaggedTensor(
                    '[ [ 1 2 3 10 20 ] [ 8 ] [ 4 5 9 10 ] ]',
                    device=device,
                    dtype=dtype)

    def test_get_layer_two_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [x x x] [x] [] [x x] ]').to(device)
            subshape = shape.get_layer(0)
            # subshape should contain the same information as shape
            assert subshape.num_axes == 2
            assert subshape == shape

    def test_get_layer_three_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape(
                '[ [[x x] [] [x] [x x x]] [[] [] [x x] [x] [x x]] ]')
            shape = shape.to(device)
            shape0 = shape.get_layer(0)
            expected_shape0 = k2.RaggedShape('[ [x x x x] [x x x x x] ]').to(
                device)
            assert shape0 == expected_shape0

            shape1 = shape.get_layer(1)
            expected_shape1 = k2.RaggedShape(
                '[ [x x] [] [x] [x x x] [] [] [x x] [x] [x x] ]').to(device)
            assert shape1 == expected_shape1

    def test_create_ragged2(self):
        lst = [[7, 9], [12, 13], []]
        ragged = k2.RaggedTensor(lst)
        expected = k2.RaggedTensor('[[7 9] [12 13] []]')
        assert ragged == expected

        float_lst = [[1.2], [], [3.4, 5.6, 7.8]]
        ragged = k2.RaggedTensor(float_lst)
        expected = k2.RaggedTensor('[[1.2] [] [3.4 5.6 7.8]]')
        assert ragged == expected

    def test_unique_sequences_two_axes(self):
        for device in self.devices:
            ragged = k2.RaggedTensor(
                '[[1 3] [1 2] [1 2] [1 4] [1 3] [1 2] [1]]').to(device)
            unique, num_repeats, new2old = ragged.unique(
                need_num_repeats=True, need_new2old_indexes=True)
            # [1, 3] has a larger hash value than [1, 2]; after sorting,
            # [1, 3] is placed after [1, 2]
            expected = k2.RaggedTensor('[[1] [1 2] [1 3] [1 4]]').to(device)
            assert unique == expected

            expected_num_repeats = k2.RaggedTensor('[[1 3 2 1]]').to(device)
            assert num_repeats == expected_num_repeats
            expected_new2old = torch.tensor([6, 1, 0, 3]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

        for device in self.devices:
            ragged = k2.RaggedTensor('[ [1 3] [1 2] [1] [1 4]]').to(device)
            unique, num_repeats, new2old = ragged.unique(
                need_num_repeats=True, need_new2old_indexes=True)

            expected = k2.RaggedTensor('[[1] [1 2] [1 3] [1 4]]').to(device)
            assert unique == expected

            expected_num_repeats = k2.RaggedTensor('[[1 1 1 1]]').to(device)
            assert num_repeats == expected_num_repeats

            # CAUTION: The output sublists are ordered by their hash value!
            expected_new2old = torch.tensor([2, 1, 0, 3]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

    def test_unique_sequences_three_axes(self):
        for device in self.devices:
            ragged = k2.RaggedTensor(
                '[ [[1] [1 2] [1 3] [1] [1 3]] [[1 4] [1 2] [1 3] [1 3] [1 2] [1]] ]'  # noqa
            ).to(device)
            unique, num_repeats, new2old = ragged.unique(
                need_num_repeats=True, need_new2old_indexes=True)
            expected = k2.RaggedTensor(
                '[ [[1] [1 2] [1 3]] [[1] [1 2] [1 3] [1 4]] ]').to(device)
            assert unique == expected

            expected_num_repeats = k2.RaggedTensor('[ [2 1 2] [1 2 2 1] ]').to(
                device)
            assert num_repeats == expected_num_repeats
            expected_new2old = torch.tensor([0, 1, 2, 10, 6, 7, 5]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

        for device in self.devices:
            ragged = k2.RaggedTensor(
                '[ [[1 3] [1] [1 2]] [[1 2] [1 3] [1 4 5] [1]] ]').to(device)
            unique, num_repeats, new2old = ragged.unique(
                need_num_repeats=True, need_new2old_indexes=True)

            expected = k2.RaggedTensor(
                '[ [[1] [1 2] [1 3]] [[1] [1 2] [1 3] [1 4 5]] ]').to(device)
            assert unique == expected

            expected_num_repeats = k2.RaggedTensor('[[1 1 1 ] [1 1 1 1]]').to(
                device)
            assert num_repeats == expected_num_repeats

            # CAUTION: The output sublists are ordered by their hash value!
            expected_new2old = torch.tensor([1, 2, 0, 6, 3, 4, 5]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

    def test_index_ragged_shape_two_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [x x] [] [x x x] ]').to(device)
            indexes = torch.tensor([-1, 0, -1, 0, 1, 2, 0, 2, 1, -1],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = shape.index(axis=0,
                                             indexes=indexes,
                                             need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [] [x x] [] [x x] [] [x x x] [x x] [x x x] [] [] ]').to(
                    device)
            assert ans == expected_ans

            expected_value_indexes = torch.tensor(
                [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 1
            indexes = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = shape.index(axis=1,
                                             indexes=indexes,
                                             need_value_indexes=True)
            expected_ans = k2.RaggedShape('[ [x x x x] [] [x x x x x] ]').to(
                device)
            assert ans == expected_ans
            assert torch.all(torch.eq(indexes, value_indexes))

    def test_index_ragged_shape_three_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [[x x x] [x x] []] [[x] [x x x]] ]').to(
                device)
            indexes = torch.tensor([-1, 0, 1, 1, -1, 0],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = shape.index(axis=0,
                                             indexes=indexes,
                                             need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [] [[x x x] [x x] []] [[x] [x x x]] [[x] [x x x]] [] [[x x x] [x x] []] ]'  # noqa
            ).to(device)
            assert ans == expected_ans
            expected_value_indexes = torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 0, 1, 2, 3, 4],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 1
            indexes = torch.tensor([0, 0, 2, 3, 3, 4],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = shape.index(axis=1,
                                             indexes=indexes,
                                             need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [[x x x] [x x x] []] [[x] [x] [x x x]] ]').to(device)
            assert ans == expected_ans
            expected_value_indexes = torch.tensor(
                [0, 1, 2, 0, 1, 2, 5, 5, 6, 7, 8],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 2
            indexes = torch.tensor([0, 2, 2, 3, 4, 4, 6, 6, 7],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = shape.index(axis=2,
                                             indexes=indexes,
                                             need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [[x x x] [x x x] [] ] [[] [x x x]] ]').to(device)
            assert ans == expected_ans
            assert torch.all(torch.eq(indexes, value_indexes))

    def test_regular_ragged_shape(self):
        shape = k2.ragged.regular_ragged_shape(1, 2)
        expected = k2.RaggedShape('[[x x]]')
        assert shape == expected

        shape = k2.ragged.regular_ragged_shape(2, 3)
        expected = k2.RaggedShape('[[x x x] [x x x]]')
        assert shape == expected

        assert shape.row_splits(1).device.type == 'cpu'

        if torch.cuda.is_available() and k2.with_cuda:
            device = torch.device('cuda', 0)
            shape = shape.to(device)
            assert shape.row_splits(1).is_cuda

    def test_argmax_per_sublist_two_axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(
                    [[1, 3, -1, -2], [1, 0, -1], [3, 2, 1], [], [1], [2, 3]],
                    dtype=dtype).to(device)
                indexes = src.argmax()

                # -1 for an empty sublist
                expected = torch.tensor([1, 4, 7, -1, 10, 12], device=device)
                assert torch.all(torch.eq(indexes, expected))

    def test_argmax_per_sublist_three_axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(
                    [[[3, 2, 1], [0, -1], []], [[2, 5, 3], [1, 10, 9, 8]]],
                    dtype=dtype).to(device)
                indexes = src.argmax()
                # -1 for an empty sublist
                expected = torch.tensor([0, 3, -1, 6, 9], device=device)
                assert torch.all(torch.eq(indexes, expected))

    def test_argmax_per_sublist_two_axes_random(self):
        res = []
        # sublists with single element
        for i in range(10000):
            res.append([random.random() * -100])
        # sublist with huge elements
        res.append([random.random() * -100 for x in range(5000)])
        ragged_cpu = k2.RaggedTensor(res)
        indexes_cpu = ragged_cpu.argmax()
        for device in self.devices:
            ragged = ragged_cpu.to(device)
            indexes = ragged.argmax().to("cpu")
            assert torch.all(torch.eq(indexes, indexes_cpu))

    def test_max_per_sublist_two_axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(
                    [[1, 3, -1, -2], [1, 0, -1], [3, 2, 1], [], [1], [2, 3]],
                    dtype=dtype).to(device)
                indexes = src.max(initial_value=0)
                # 0 for an empty sublist
                expected = torch.tensor([3, 1, 3, 0, 1, 3], device=device)
                assert torch.all(torch.eq(indexes, expected))

    def test_max_per_sublist_three_axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor(
                    [[[3, 2, 1], [0, -1], []], [[2, 5, 3], [1, 10, 9, 8]]],
                    dtype).to(device)
                indexes = src.max(initial_value=0)
                # 0 for an empty sublist
                expected = torch.tensor([3, 0, 0, 5, 10], device=device)
                assert torch.all(torch.eq(indexes, expected))

    def test_sort_sublist_ascending(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor([[3, 2], [], [1, 5, 2]],
                                      dtype=dtype).to(device)
                src_clone = src.clone()
                new2old = src.sort_(descending=False,
                                    need_new2old_indexes=True)
                expected_src = k2.RaggedTensor([[2, 3], [], [1, 2, 5]],
                                               dtype=dtype).to(device)
                expected_new2old = torch.tensor([1, 0, 2, 4, 3],
                                                device=device,
                                                dtype=torch.int32)
                assert src == expected_src
                assert torch.all(torch.eq(new2old, expected_new2old))

                expected_sorted = k2.index_select(src_clone.values, new2old)
                sorted = src.values
                assert torch.all(torch.eq(expected_sorted, sorted))

    def test_sort_sublist_descending(self):
        for device in self.devices:
            for dtype in self.dtypes:
                src = k2.RaggedTensor([[3, 2], [], [1, 5, 2]],
                                      dtype).to(device)
                src_clone = src.clone()
                new2old = src.sort_(descending=True, need_new2old_indexes=True)
                sorted_src = k2.RaggedTensor([[3, 2], [], [5, 2, 1]],
                                             dtype=dtype).to(device)
                expected_new2old = torch.tensor([0, 1, 3, 4, 2],
                                                device=device,
                                                dtype=torch.int32)
                assert src == sorted_src
                assert torch.all(torch.eq(new2old, expected_new2old))

                expected_sorted = k2.index_select(src_clone.values, new2old)
                sorted = src.values
                assert torch.all(torch.eq(expected_sorted, sorted))


if __name__ == '__main__':
    unittest.main()
