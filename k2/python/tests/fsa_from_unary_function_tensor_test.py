#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  fsa_from_unary_function_tensor_test_py

import unittest

import k2
import torch
import _k2


class TestFsaFromUnaryFunctionTensor(unittest.TestCase):

    def test_without_negative_1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            s = '''
                0 1 2 10
                0 1 1 20
                1 2 -1 30
                2
            '''
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            src.float_attr = torch.tensor([0.1, 0.2, 0.3],
                                          dtype=torch.float32,
                                          requires_grad=True,
                                          device=device)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedInt('[[1 2 3] [5 6] []]').to(device)
            src.attr1 = 'src'
            src.attr2 = 'fsa'

            ragged_arc, arc_map = _k2.arc_sort(src.arcs, need_arc_map=True)

            dest = k2.utils.fsa_from_unary_function_tensor(
                src, ragged_arc, arc_map)

            assert torch.allclose(
                dest.float_attr,
                torch.tensor([0.2, 0.1, 0.3],
                             dtype=torch.float32,
                             device=device))

            assert torch.all(
                torch.eq(
                    dest.scores,
                    torch.tensor([20, 10, 30],
                                 dtype=torch.float32,
                                 device=device)))

            assert torch.all(
                torch.eq(
                    dest.int_attr,
                    torch.tensor([2, 1, 3], dtype=torch.int32, device=device)))

            expected_ragged_attr = k2.RaggedInt('[ [5 6] [1 2 3] []]')
            self.assertEqual(str(dest.ragged_attr), str(expected_ragged_attr))

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            # now for autograd
            scale = torch.tensor([10, 20, 30], device=device)
            (dest.float_attr * scale).sum().backward()
            (dest.scores * scale).sum().backward()

            expected_grad = torch.tensor([20, 10, 30],
                                         dtype=torch.float32,
                                         device=device)

            assert torch.all(torch.eq(src.float_attr.grad, expected_grad))

            assert torch.all(torch.eq(src.scores.grad, expected_grad))

    def test_with_negative_1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            s = '''
                0 1 2 10
                0 1 1 20
                1 2 -1 30
                2
            '''
            src = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            src.float_attr = torch.tensor([0.1, 0.2, 0.3],
                                          dtype=torch.float32,
                                          requires_grad=True,
                                          device=device)
            src.int_attr = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device=device)
            src.ragged_attr = k2.RaggedInt('[[1 2 3] [5 6] []]').to(device)
            src.attr1 = 'src'
            src.attr2 = 'fsa'
            ragged_arc, arc_map = _k2.add_epsilon_self_loops(src.arcs,
                                                             need_arc_map=True)
            dest = k2.utils.fsa_from_unary_function_tensor(
                src, ragged_arc, arc_map)
            assert torch.allclose(
                dest.float_attr,
                torch.tensor([0.0, 0.1, 0.2, 0.0, 0.3],
                             dtype=torch.float32,
                             device=device))

            assert torch.all(
                torch.eq(
                    dest.scores,
                    torch.tensor([0, 10, 20, 0, 30],
                                 dtype=torch.float32,
                                 device=device)))

            assert torch.all(
                torch.eq(
                    dest.int_attr,
                    torch.tensor([0, 1, 2, 0, 3],
                                 dtype=torch.int32,
                                 device=device)))

            expected_ragged_attr = k2.RaggedInt('[ [] [1 2 3] [5 6] [] []]')
            self.assertEqual(str(dest.ragged_attr), str(expected_ragged_attr))

            assert dest.attr1 == src.attr1
            assert dest.attr2 == src.attr2

            # now for autograd
            scale = torch.tensor([10, 20, 30, 40, 50], device=device)
            (dest.float_attr * scale).sum().backward()
            (dest.scores * scale).sum().backward()

            expected_grad = torch.tensor([20, 30, 50],
                                         dtype=torch.float32,
                                         device=device)

            assert torch.all(torch.eq(src.float_attr.grad, expected_grad))

            assert torch.all(torch.eq(src.scores.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
