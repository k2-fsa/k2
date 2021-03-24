#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#                2021  Mobvoi Inc. (authors: Yaguang Hu)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R ragged_ops_test_py

import unittest

import _k2
import k2
import numpy as np
import torch


class TestRaggedOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))

    def test_remove_axis_ragged_array(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 0 ] [ 3 0 ] [ 2 ] ]')

        ans = k2.ragged.remove_axis(src, 1)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] ]')

    def test_remove_axis_ragged_shape(self):
        shape = k2.RaggedShape('[ [[x x] [] [x]] [[] [x x] [x x x] [x]] ]')

        ans = k2.ragged.remove_axis(shape, 0)
        expected = k2.RaggedShape('[[x x] [] [x] [] [x x] [x x x] [x]]')
        self.assertEqual(str(ans), str(expected))

        ans = k2.ragged.remove_axis(shape, 1)
        expected = k2.RaggedShape('[[x x x] [x x x x x x]]')
        self.assertEqual(str(ans), str(expected))

        ans = k2.ragged.remove_axis(shape, 2)
        expected = k2.RaggedShape('[[x x x] [x x x x]]')
        self.assertEqual(str(ans), str(expected))

    def test_to_list(self):
        s = '''
            [ [ [ 1 2 ] [ 0 ] ] [ [ 3 0 ] [ 2 ] ] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_axis(src, 0)
        self.assertEqual(k2.ragged.to_list(ans), [[1, 2], [0], [3, 0], [2]])

    def test_remove_values_leq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_leq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 6)
        self.assertEqual(str(ans), '[ [ ] [ ] [ 8 ] [ ] ]')

        ans = k2.ragged.remove_values_leq(src, 8)
        self.assertEqual(str(ans), '[ [ ] [ ] [ ] [ ] ]')

    def test_remove_values_eq(self):
        s = '''
            [ [1 2 0] [3 0 2] [0 8 0 6 0] [0] ]
        '''
        src = k2.RaggedInt(s)

        ans = k2.ragged.remove_values_eq(src, 0)
        self.assertEqual(str(ans), '[ [ 1 2 ] [ 3 2 ] [ 8 6 ] [ ] ]')

        ans = k2.ragged.remove_values_eq(src, 1)
        self.assertEqual(str(ans), '[ [ 2 0 ] [ 3 0 2 ] [ 0 8 0 6 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 6)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 8 0 0 ] [ 0 ] ]')

        ans = k2.ragged.remove_values_eq(src, 8)
        self.assertEqual(str(ans), '[ [ 1 2 0 ] [ 3 0 2 ] [ 0 0 6 0 ] [ 0 ] ]')

    def test_normalize_scores_use_log_non_zero_stride(self):
        s = '''
            [ [1 -1 0] [2 10] [] [3] [5 8] ]
        '''
        src = k2.ragged.RaggedFloat(s)
        saved = src.values.clone().detach()
        saved.requires_grad_(True)
        src.requires_grad_(True)

        ans = k2.ragged.normalize_scores(src, use_log=True)

        scale = torch.arange(ans.values.numel())

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
        src = k2.ragged.RaggedFloat(s)
        saved = src.values.clone().detach()
        saved.requires_grad_(True)
        src.requires_grad_(True)

        ans = k2.ragged.normalize_scores(src, use_log=True)

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
        fsa = k2.Fsa.from_str(s)
        scores = torch.arange(fsa.scores.numel(), dtype=torch.float32)
        scores.requires_grad_(True)

        ragged_scores = k2.ragged.RaggedFloat(fsa.arcs.shape(), scores)
        assert ragged_scores.requires_grad is True

        normalized_scores = k2.ragged.normalize_scores(ragged_scores,
                                                       use_log=True)
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
        self.assertAlmostEqual(fsa.scores[5].exp().sum().item(), 1.0, places=6)

    def test_normalize_scores(self):
        for device in self.devices:
            s = '''
                [ [1 3 5] [2 -1] [] [3] [5 2] ]
            '''
            src = k2.ragged.RaggedFloat(s).to(device)
            saved = src.values

            ans = k2.ragged.normalize_scores(src, use_log=False)
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
        fsa = k2.Fsa.from_str(s)
        scores = torch.randn_like(fsa.scores)
        fsa.set_scores_stochastic_(scores)
        normalized_scores = k2.ragged.sum_per_sublist(
            _k2.RaggedFloat(fsa.arcs.shape(), fsa.scores.exp()))
        assert normalized_scores.numel() == fsa.arcs.dim0()

        assert torch.allclose(normalized_scores[:-1],
                              torch.ones(normalized_scores.numel() - 1))

        # the final state has no leaving arcs
        assert normalized_scores[-1].item() == 0

    def test_append(self):
        ragged1 = k2.RaggedInt('[ [1 2 3] [] [4 5] ]')
        ragged2 = k2.RaggedInt('[ [] [10 20] [30] [40 50] ]')
        ragged = k2.ragged.append([ragged1, ragged2])
        self.assertEqual(
            str(ragged),
            '[ [ 1 2 3 ] [ ] [ 4 5 ] [ ] [ 10 20 ] [ 30 ] [ 40 50 ] ]')

    def test_append_axis1(self):
        ragged1 = k2.RaggedInt('[ [1 2 3] [] [4 5] ]')
        ragged2 = k2.RaggedInt('[ [10 20] [8] [9 10] ]')
        ragged = k2.ragged.append([ragged1, ragged2], axis=1)
        self.assertEqual(str(ragged), '[ [ 1 2 3 10 20 ] [ 8 ] [ 4 5 9 10 ] ]')

    def test_get_layer_two_axes(self):
        shape = k2.RaggedShape('[ [x x x] [x] [] [x x] ]')
        subshape = k2.ragged.get_layer(shape, 0)
        # subshape should contain the same information as shape
        self.assertEqual(subshape.num_axes(), 2)
        self.assertEqual(str(subshape), str(shape))

    def test_get_layer_three_axes(self):
        shape = k2.RaggedShape(
            '[ [[x x] [] [x] [x x x]] [[] [] [x x] [x] [x x]] ]')
        shape0 = k2.ragged.get_layer(shape, 0)
        expected_shape0 = k2.RaggedShape('[ [x x x x] [x x x x x] ]')
        self.assertEqual(str(shape0), str(expected_shape0))

        shape1 = k2.ragged.get_layer(shape, 1)
        expected_shape1 = k2.RaggedShape(
            '[ [x x] [] [x] [x x x] [] [] [x x] [x] [x x] ]')
        self.assertEqual(str(shape1), str(expected_shape1))

    def test_create_ragged2(self):
        lst = [[7, 9], [12, 13], []]
        ragged = k2.create_ragged2(lst)
        expected = k2.RaggedInt('[[7 9] [12 13] []]')
        self.assertEqual(str(ragged), str(expected))

        float_lst = [[1.2], [], [3.4, 5.6, 7.8]]
        ragged = k2.create_ragged2(float_lst)
        expected = _k2.RaggedFloat('[[1.2] [] [3.4 5.6 7.8]]')
        self.assertEqual(str(ragged), str(expected))

    def test_unique_sequences_two_axes(self):
        for device in self.devices:
            ragged = k2.RaggedInt(
                '[[1 3] [1 2] [1 2] [1 4] [1 3] [1 2] [1]]').to(device)
            unique, num_repeats, new2old = k2.ragged.unique_sequences(
                ragged, need_num_repeats=True, need_new2old_indexes=True)
            # [1, 3] has a larger hash value than [1, 2]; after sorting,
            # [1, 3] is placed after [1, 2]
            expected = k2.RaggedInt('[[1] [1 2] [1 3] [1 4]]')
            self.assertEqual(str(unique), str(expected))

            expected_num_repeats = k2.RaggedInt('[[1 3 2 1]]')
            self.assertEqual(str(num_repeats), str(expected_num_repeats))
            expected_new2old = torch.tensor([6, 1, 0, 3]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

        for device in self.devices:
            ragged = k2.RaggedInt('[ [1 3] [1 2] [1] [1 4]]').to(device)
            unique, num_repeats, new2old = k2.ragged.unique_sequences(
                ragged, need_num_repeats=True, need_new2old_indexes=True)

            expected = k2.RaggedInt('[[1] [1 2] [1 3] [1 4]]')
            self.assertEqual(str(unique), str(expected))

            expected_num_repeats = k2.RaggedInt('[[1 1 1 1]]')
            self.assertEqual(str(num_repeats), str(expected_num_repeats))

            # CAUTION: The output sublists are ordered by their hash value!
            expected_new2old = torch.tensor([2, 1, 0, 3]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

    def test_unique_sequences_three_axes(self):
        for device in self.devices:
            ragged = k2.RaggedInt(
                '[ [[1] [1 2] [1 3] [1] [1 3]] [[1 4] [1 2] [1 3] [1 3] [1 2] [1]] ]'  # noqa
            ).to(device)
            unique, num_repeats, new2old = k2.ragged.unique_sequences(
                ragged, need_num_repeats=True, need_new2old_indexes=True)
            expected = k2.RaggedInt(
                '[ [[1] [1 2] [1 3]] [[1] [1 2] [1 3] [1 4]] ]')
            self.assertEqual(str(unique), str(expected))

            expected_num_repeats = k2.RaggedInt('[ [2 1 2] [1 2 2 1] ]')
            self.assertEqual(str(num_repeats), str(expected_num_repeats))
            expected_new2old = torch.tensor([0, 1, 2, 10, 6, 7, 5]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

        for device in self.devices:
            ragged = k2.RaggedInt(
                '[ [[1 3] [1] [1 2]] [[1 2] [1 3] [1 4 5] [1]] ]').to(device)
            unique, num_repeats, new2old = k2.ragged.unique_sequences(
                ragged, need_num_repeats=True, need_new2old_indexes=True)

            expected = k2.RaggedInt(
                '[ [[1] [1 2] [1 3]] [[1] [1 2] [1 3] [1 4 5]] ]')
            self.assertEqual(str(unique), str(expected))

            expected_num_repeats = k2.RaggedInt('[[1 1 1 ] [1 1 1 1]]')
            self.assertEqual(str(num_repeats), str(expected_num_repeats))

            # CAUTION: The output sublists are ordered by their hash value!
            expected_new2old = torch.tensor([1, 2, 0, 6, 3, 4, 5]).to(device)
            assert torch.all(torch.eq(new2old, expected_new2old))

    def test_index_ragged_shape_two_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [x x] [] [x x x] ]').to(device)
            indexes = torch.tensor([-1, 0, -1, 0, 1, 2, 0, 2, 1, -1],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = k2.ragged.index(shape,
                                                 indexes,
                                                 axis=0,
                                                 need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [] [x x] [] [x x] [] [x x x] [x x] [x x x] [] [] ]')
            self.assertEqual(str(ans), str(expected_ans))

            expected_value_indexes = torch.tensor(
                [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 1
            indexes = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = k2.ragged.index(shape,
                                                 indexes,
                                                 axis=1,
                                                 need_value_indexes=True)
            expected_ans = k2.RaggedShape('[ [x x x x] [] [x x x x x] ]')
            self.assertEqual(str(ans), str(expected_ans))
            assert torch.all(torch.eq(indexes, value_indexes))

    def test_index_ragged_shape_three_axes(self):
        for device in self.devices:
            shape = k2.RaggedShape('[ [[x x x] [x x] []] [[x] [x x x]] ]').to(
                device)
            indexes = torch.tensor([-1, 0, 1, 1, -1, 0],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = k2.ragged.index(shape,
                                                 indexes,
                                                 axis=0,
                                                 need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [] [[x x x] [x x] []] [[x] [x x x]] [[x] [x x x]] [] [[x x x] [x x] []] ]'  # noqa
            )
            self.assertEqual(str(ans), str(expected_ans))
            expected_value_indexes = torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 0, 1, 2, 3, 4],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 1
            indexes = torch.tensor([0, 0, 2, 3, 3, 4],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = k2.ragged.index(shape,
                                                 indexes,
                                                 axis=1,
                                                 need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [[x x x] [x x x] []] [[x] [x] [x x x]] ]')
            self.assertEqual(str(ans), str(expected_ans))
            expected_value_indexes = torch.tensor(
                [0, 1, 2, 0, 1, 2, 5, 5, 6, 7, 8],
                dtype=torch.int32,
                device=device)
            assert torch.all(torch.eq(value_indexes, expected_value_indexes))

            # for axis == 2
            indexes = torch.tensor([0, 2, 2, 3, 4, 4, 6, 6, 7],
                                   dtype=torch.int32,
                                   device=device)
            ans, value_indexes = k2.ragged.index(shape,
                                                 indexes,
                                                 axis=2,
                                                 need_value_indexes=True)
            expected_ans = k2.RaggedShape(
                '[ [[x x x] [x x x] [] ] [[] [x x x]] ]')
            self.assertEqual(str(ans), str(expected_ans))
            assert torch.all(torch.eq(indexes, value_indexes))

    def test_regular_ragged_shape(self):
        shape = k2.ragged.regular_ragged_shape(1, 2)
        expected = k2.RaggedShape('[[x x]]')
        self.assertEqual(str(shape), str(expected))

        shape = k2.ragged.regular_ragged_shape(2, 3)
        expected = k2.RaggedShape('[[x x x] [x x x]]')
        self.assertEqual(str(shape), str(expected))

        assert shape.row_splits(1).device.type == 'cpu'

        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
            shape = shape.to(device)
            assert shape.row_splits(1).is_cuda

    def test_argmax_per_sublist_two_axes(self):
        for device in self.devices:
            src = k2.RaggedFloat(
                '[[1 3 -1 -2] [1 0 -1] [3 2 1] [] [1] [2 3]]').to(device)
            indexes = k2.ragged.argmax_per_sublist(src)

            # -1 for an empty sublist
            expected = torch.tensor([1, 4, 7, -1, 10, 12], device=device)
            assert torch.all(torch.eq(indexes, expected))

    def test_argmax_per_sublist_three_axes(self):
        for device in self.devices:
            src = k2.RaggedFloat(
                '[ [[3 2 1] [0 -1] []] [[2 5 3] [1 10 9 8]] ]').to(device)
            indexes = k2.ragged.argmax_per_sublist(src)
            # -1 for an empty sublist
            expected = torch.tensor([0, 3, -1, 6, 9], device=device)
            assert torch.all(torch.eq(indexes, expected))

    def test_max_per_sublist_two_axes(self):
        for device in self.devices:
            src = k2.RaggedFloat(
                '[[1 3 -1 -2] [1 0 -1] [3 2 1] [] [1] [2 3]]').to(device)
            indexes = k2.ragged.max_per_sublist(src, initial_value=0)
            # 0 for an empty sublist
            expected = torch.tensor([3, 1, 3, 0, 1, 3], device=device)
            assert torch.all(torch.eq(indexes, expected))

    def test_max_per_sublist_three_axes(self):
        for device in self.devices:
            src = k2.RaggedFloat(
                '[ [[3 2 1] [0 -1] []] [[2 5 3] [1 10 9 8]] ]').to(device)
            indexes = k2.ragged.max_per_sublist(src, initial_value=0)
            # 0 for an empty sublist
            expected = torch.tensor([3, 0, 0, 5, 10], device=device)
            assert torch.all(torch.eq(indexes, expected))


if __name__ == '__main__':
    unittest.main()
