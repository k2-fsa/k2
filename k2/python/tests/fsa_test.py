#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_test_py -E "host|dense"

import unittest

import torch
import _k2  # for test only, users should not import it.
import k2
import os


def _remove_leading_spaces(s: str) -> str:
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    return '\n'.join(lines)


class TestFsa(unittest.TestCase):

    def test_acceptor_from_tensor(self):
        fsa_tensor = torch.tensor([[0, 1, 2, _k2.float_as_int(-1.2)],
                                   [0, 2, 10, _k2.float_as_int(-2.2)],
                                   [1, 6, -1, _k2.float_as_int(-3.2)],
                                   [1, 3, 3, _k2.float_as_int(-4.2)],
                                   [2, 6, -1, _k2.float_as_int(-5.2)],
                                   [2, 4, 2, _k2.float_as_int(-6.2)],
                                   [3, 6, -1, _k2.float_as_int(-7.2)],
                                   [5, 0, 1, _k2.float_as_int(-8.2)]],
                                  dtype=torch.int32)

        fsa = k2.Fsa(fsa_tensor)

        expected_str = '''
            0 1 2 -1.2
            0 2 10 -2.2
            1 6 -1 -3.2
            1 3 3 -4.2
            2 6 -1 -5.2
            2 4 2 -6.2
            3 6 -1 -7.2
            5 0 1 -8.2
            6
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa))

        arcs = fsa.arcs.values()[:, :-1]
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.scores,
            torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        fsa.scores *= -1

        assert torch.allclose(
            fsa.scores,
            torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
                         dtype=torch.float32))

    def test_acceptor_from_str(self):
        s = '''
            0 1 2 -1.2
            0 2  10 -2.2
            1 6 -1  -3.2
            1 3  3  -4.2
            2 6 -1  -5.2
            2 4  2  -6.2
            3 6 -1  -7.2
            5 0  1  -8.2
            6
        '''

        fsa = k2.Fsa.from_str(_remove_leading_spaces(s))

        expected_str = '''
            0 1 2 -1.2
            0 2 10 -2.2
            1 6 -1 -3.2
            1 3 3 -4.2
            2 6 -1 -5.2
            2 4 2 -6.2
            3 6 -1 -7.2
            5 0 1 -8.2
            6
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa))

        arcs = fsa.arcs.values()[:, :-1]
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.scores,
            torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        fsa.scores *= -1

        assert torch.allclose(
            fsa.scores,
            torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
                         dtype=torch.float32))

    def test_acceptor_from_openfst(self):
        s = '''
            0 1 2 -1.2
            0 2  10 -2.2
            1 6  1  -3.2
            1 3  3  -4.2
            2 6  2  -5.2
            2 4  2  -6.2
            3 6  3  -7.2
            5 0  1  -8.2
            7
            6 -9.2
        '''

        fsa = k2.Fsa.from_openfst(_remove_leading_spaces(s), acceptor=True)

        expected_str = '''
            0 1 2 -1.2
            0 2 10 -2.2
            1 6 1 -3.2
            1 3 3 -4.2
            2 6 2 -5.2
            2 4 2 -6.2
            3 6 3 -7.2
            5 0 1 -8.2
            6 8 -1 -9.2
            7 8 -1 -0
            8
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa, openfst=True))

        arcs = fsa.arcs.values()[:, :-1]
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (10, 3), 'there should be 10 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.scores,
            torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 0],
                         dtype=torch.float32))

        fsa.scores *= -1

        assert torch.allclose(
            fsa.scores,
            torch.tensor(
                [-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2, -9.2, 0],
                dtype=torch.float32))

    def test_transducer_from_tensor(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
            fsa_tensor = torch.tensor(
                [[0, 1, 2, _k2.float_as_int(-1.2)],
                 [0, 2, 10, _k2.float_as_int(-2.2)],
                 [1, 6, -1, _k2.float_as_int(-4.2)],
                 [1, 3, 3, _k2.float_as_int(-3.2)],
                 [2, 6, -1, _k2.float_as_int(-5.2)],
                 [2, 4, 2, _k2.float_as_int(-6.2)],
                 [3, 6, -1, _k2.float_as_int(-7.2)],
                 [5, 0, 1, _k2.float_as_int(-8.2)]],
                dtype=torch.int32).to(device)
            aux_labels_tensor = torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                                             dtype=torch.int32).to(device)
            fsa = k2.Fsa(fsa_tensor, aux_labels_tensor)
            assert fsa.aux_labels.dtype == torch.int32
            assert fsa.aux_labels.device.type == device.type
            assert torch.allclose(
                fsa.aux_labels,
                torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                             dtype=torch.int32).to(device))

            expected_str = '''
                0 1 2 22 -1.2
                0 2 10 100 -2.2
                1 6 -1 16 -4.2
                1 3 3 33 -3.2
                2 6 -1 26 -5.2
                2 4 2 22 -6.2
                3 6 -1 36 -7.2
                5 0 1 50 -8.2
                6
            '''
            assert _remove_leading_spaces(
                expected_str) == _remove_leading_spaces(k2.to_str(fsa))

    def test_transducer_from_str(self):
        s = '''
            0 1 2 22  -1.2
            0 2  10 100 -2.2
            1 6 -1  16  -4.2
            1 3  3  33  -3.2
            2 6 -1  26  -5.2
            2 4  2  22  -6.2
            3 6 -1  36  -7.2
            5 0  1  50  -8.2
            6
        '''
        fsa = k2.Fsa.from_str(_remove_leading_spaces(s))
        assert fsa.aux_labels.dtype == torch.int32
        assert fsa.aux_labels.device.type == 'cpu'
        assert torch.allclose(
            fsa.aux_labels,
            torch.tensor([22, 100, 16, 33, 26, 22, 36, 50], dtype=torch.int32))

        expected_str = '''
            0 1 2 22 -1.2
            0 2 10 100 -2.2
            1 6 -1 16 -4.2
            1 3 3 33 -3.2
            2 6 -1 26 -5.2
            2 4 2 22 -6.2
            3 6 -1 36 -7.2
            5 0 1 50 -8.2
            6
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa))

    def test_transducer_from_openfst(self):
        s = '''
            0 1 2 22  -1.2
            0 2  10 100 -2.2
            1 6  1  16  -4.2
            1 3  3  33  -3.2
            2 6  2  26  -5.2
            2 4  2  22  -6.2
            3 6  3  36  -7.2
            5 0  1  50  -8.2
            7 -9.2
            6
        '''
        fsa = k2.Fsa.from_openfst(_remove_leading_spaces(s), acceptor=False)
        assert fsa.aux_labels.dtype == torch.int32
        assert fsa.aux_labels.device.type == 'cpu'
        assert torch.allclose(
            fsa.aux_labels,
            torch.tensor([22, 100, 16, 33, 26, 22, 36, 50, -1, -1],
                         dtype=torch.int32))

        expected_str = '''
            0 1 2 22 -1.2
            0 2 10 100 -2.2
            1 6 1 16 -4.2
            1 3 3 33 -3.2
            2 6 2 26 -5.2
            2 4 2 22 -6.2
            3 6 3 36 -7.2
            5 0 1 50 -8.2
            6 8 -1 -1 -0
            7 8 -1 -1 -9.2
            8
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa, openfst=True))

    def test_fsa_io(self):
        s = '''
            0 1 10 0.1
            0 2 20 0.2
            1 3 -1 0.3
            2 3 -1 0.4
            3
        '''
        fsa = k2.Fsa.from_str(_remove_leading_spaces(s))
        tensor = k2.to_tensor(fsa)
        assert tensor.ndim == 2
        assert tensor.dtype == torch.int32
        del fsa  # tensor is still accessible

        fsa = k2.Fsa(tensor)
        del tensor
        assert torch.allclose(
            fsa.scores, torch.tensor([0.1, 0.2, 0.3, 0.4],
                                     dtype=torch.float32))
        assert torch.allclose(
            fsa.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor([[0, 1, 10], [0, 2, 20], [1, 3, -1], [2, 3, -1]],
                         dtype=torch.int32))

        # now test vector of FSAs

        ragged_arc = _k2.fsa_to_fsa_vec(fsa.arcs)
        del fsa
        fsa_vec = k2.Fsa(ragged_arc)
        del ragged_arc

        assert fsa_vec.shape == (1, None, None)

        assert torch.allclose(
            fsa_vec.scores,
            torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32))
        assert torch.allclose(
            fsa_vec.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor([[0, 1, 10], [0, 2, 20], [1, 3, -1], [2, 3, -1]],
                         dtype=torch.int32))

        tensor = k2.to_tensor(fsa_vec)
        assert tensor.ndim == 1
        assert tensor.dtype == torch.int32
        del fsa_vec
        fsa_vec = k2.Fsa(tensor)
        assert torch.allclose(
            fsa_vec.scores,
            torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32))
        assert torch.allclose(
            fsa_vec.arcs.values()[:, :-1],  # skip the last field `scores`
            torch.tensor([[0, 1, 10], [0, 2, 20], [1, 3, -1], [2, 3, -1]],
                         dtype=torch.int32))

    def test_symbol_table_and_dot(self):
        isym_str = '''
            <eps> 0
            a 1
            b 2
            c 3
        '''

        osym_str = '''
            <eps> 0
            x 1
            y 2
            z 3
        '''
        symbols = k2.SymbolTable.from_str(isym_str)
        aux_symbols = k2.SymbolTable.from_str(osym_str)

        rules = '''
            0 1 1 1 0.5
            0 1 2 2 1.5
            1 2 3 3  2.5
            2 3 -1 0 3.5
            3
        '''
        fsa = k2.Fsa.from_str(_remove_leading_spaces(rules))
        fsa.symbols = symbols
        fsa.aux_symbols = aux_symbols

        fsa.draw(filename='foo.png')
        os.remove('foo.png')

    def test_to(self):
        s = '''
            0 1 -1 1
            1
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.is_cpu()

        if torch.cuda.is_available():
            fsa = fsa.to('cuda:0')
            assert fsa.is_cuda()

        fsa = fsa.to('cpu')
        assert fsa.is_cpu()

    def test_getitem(self):
        s0 = '''
            0 1 1 0.1
            1 2 2 0.2
            2 3 -1 0.3
            3
        '''
        s1 = '''
            0 1 -1 0.4
            1
        '''
        fsa0 = k2.Fsa.from_str(s0).requires_grad_(True)
        fsa1 = k2.Fsa.from_str(s1).requires_grad_(True)

        fsa_vec = k2.create_fsa_vec([fsa0, fsa1])
        assert fsa_vec.shape == (2, None, None)

        new_fsa0 = fsa_vec[0]
        assert new_fsa0.shape == (4, None)  # it has 4 states

        scale = torch.arange(new_fsa0.scores.numel())

        (new_fsa0.scores * scale).sum().backward()
        assert torch.allclose(fsa0.scores.grad, torch.tensor([0., 1., 2.]))

        new_fsa1 = fsa_vec[1]
        assert new_fsa1.shape == (2, None)  # it has 2 states

        (new_fsa1.scores * 5).sum().backward()
        assert torch.allclose(fsa1.scores.grad, torch.tensor([5.]))

    def test_ragged_str(self):
        s = '''
            0 1 1 0.1
            0 2 2 0.2
            1 2 3 0.3
            2 3 -1 0.4
            3
        '''
        fsa = k2.Fsa.from_str(s)
        print(fsa.arcs)
        ''' It prints:
        [ [ 0 1 1 0.1 0 2 2 0.2 ] [ 1 2 3 0.3 ] [ 2 3 -1 0.4 ] [ ] ]
        '''

    def test_invert(self):
        s0 = '''
            0 1 1 4 0.1
            1 2 2 5 0.2
            2 3 -1 -1 0.3
            3
        '''
        s1 = '''
            0 1 4 1 0.1
            1 2 5 2 0.2
            2 3 -1 -1 0.3
            1
        '''
        fsa0 = k2.Fsa.from_str(s0).requires_grad_(True)
        fsa1 = k2.Fsa.from_str(s1).requires_grad_(True)

        fsa0.invert_()
        assert str(fsa0) == str(fsa1)
        fsa0.invert_()
        fsa1.invert_()
        assert str(fsa0) == str(fsa1)
        fsa1.invert_()
        assert str(fsa0) != str(fsa1)

    def test_single_fsa_as_dict(self):
        s = '''
            0 1 1 10 0.1
            1 2 -1 -1 0.2
            2
        '''

        sym_str = '''
            a 1
        '''
        symbol_table = k2.SymbolTable.from_str(sym_str)
        fsa = k2.Fsa.from_str(s)
        fsa.symbols = symbol_table
        del symbol_table

        fsa.tensor_attr1 = torch.tensor([1, 2])
        fsa.tensor_attr2 = torch.tensor([[10, 20], [30, 40.]])
        fsa.non_tensor_attr1 = 'test-fsa'
        fsa.non_tensor_attr2 = 20201208

        fsa_dict = fsa.as_dict()
        del fsa

        fsa = k2.Fsa.from_dict(fsa_dict)
        assert torch.all(torch.eq(fsa.tensor_attr1, torch.tensor([1, 2])))
        assert torch.all(
            torch.eq(fsa.tensor_attr2, torch.tensor([[10, 20], [30, 40]])))
        assert fsa.non_tensor_attr1 == 'test-fsa'
        assert fsa.non_tensor_attr2 == 20201208
        assert fsa.symbols.get('a') == 1
        assert fsa.symbols.get(1) == 'a'

    def test_fsa_vec_as_dict(self):
        s1 = '''
            0 1 1 10 0.1
            1 2 -1 -1 0.2
            2
        '''
        s2 = '''
            0 1 -1 30 0.3
            1
        '''
        fsa1 = k2.Fsa.from_str(s1)
        fsa2 = k2.Fsa.from_str(s2)
        fsa = k2.create_fsa_vec([fsa1, fsa2])
        del fsa1, fsa2

        sym_str = '''
            a 1
        '''
        symbol_table = k2.SymbolTable.from_str(sym_str)
        fsa.symbols = symbol_table
        del symbol_table

        fsa.tensor_attr1 = torch.tensor([1, 2, 3])
        fsa.tensor_attr2 = torch.tensor([[10, 20], [30, 40.], [50, 60]])
        fsa.non_tensor_attr1 = 'test-fsa-vec'
        fsa.non_tensor_attr2 = 20201208

        fsa_dict = fsa.as_dict()
        del fsa

        fsa = k2.Fsa.from_dict(fsa_dict)
        assert fsa.shape == (2, None, None)
        assert torch.all(torch.eq(fsa.tensor_attr1, torch.tensor([1, 2, 3])))
        assert torch.all(
            torch.eq(fsa.tensor_attr2,
                     torch.tensor([[10, 20], [30, 40], [50, 60]])))
        assert fsa.non_tensor_attr1 == 'test-fsa-vec'
        assert fsa.non_tensor_attr2 == 20201208
        assert fsa.symbols.get('a') == 1
        assert fsa.symbols.get(1) == 'a'

    def test_fsa_vec_as_dict_ragged(self):
        r = k2.RaggedInt(k2.RaggedShape('[ [ x x ] [x] [ x x ] [x]]'),
                         torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int32))
        g = k2.Fsa.from_str('0  1  3  0.0\n  1 2 -1 0.0\n  2')
        h = k2.create_fsa_vec([g, g])
        h.aux_labels = r
        assert (h[0].aux_labels.dim0() == h[0].labels.shape[0])

    def test_set_scores_stochastic(self):
        s = '''
            0 1 1 0.
            0 1 2 0.
            1 2 3 0.
            1 2 4 0.
            1 2 5 0.
            2 3 -1 0.
            3
        '''
        fsa = k2.Fsa.from_str(s)
        scores = torch.randn_like(fsa.scores)
        fsa.set_scores_stochastic_(scores)

        # scores of state 0 should be normalized
        assert torch.allclose(fsa.scores[0:2].exp().sum(), torch.Tensor([1]))

        # scores of state 1 should be normalized
        assert torch.allclose(fsa.scores[2:5].exp().sum(), torch.Tensor([1]))

        # scores of state 2 should be normalized
        assert torch.allclose(fsa.scores[5].exp().sum(), torch.Tensor([1]))

    def test_scores_autograd(self):
        s = '''
            0 1 -1 100
            1
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)
        s = 8 * fsa.scores
        s.sum().backward()
        assert fsa.grad == 8

        import torch.optim as optim
        optimizer = optim.SGD([{'params': [fsa.scores]}], lr=0.25)
        optimizer.step()

        assert fsa.scores.item() == 98
        assert _k2.as_float(fsa.arcs.values()[:, -1]) == 98

    def test_scores_autograd_with_assignment(self):
        s = '''
            0 1 -1 10.
            1
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.requires_grad_(False)

        scores = torch.tensor([100.], dtype=torch.float32, requires_grad=True)

        import torch.optim as optim
        optimizer = optim.SGD([{'params': [scores]}], lr=0.25)

        # CAUTION: we use [:] here
        fsa.scores[:] = scores

        assert fsa.requires_grad is True
        (fsa.scores * 8).sum().backward()

        assert scores.grad == 8

        optimizer.step()

        assert fsa.scores.item() == 100, f'fsa.scores is {fsa.scores}'
        assert scores.item() == 98

        # CAUTION: had we used fsa.scores = scores,
        # would we have `fsa.scores != fsa.arcs.values()[:, -1]`.
        # That is, `fsa.scores` shares memory with `scores`,
        # but not with fsa.arcs.values!
        assert _k2.as_float(fsa.arcs.values()[:, -1]).item() == 100

    def test_detach(self):
        s = '''
            0 1 -1 10.0
            1
        '''
        fsa = k2.Fsa.from_str(s)
        fsa.requires_grad_(True)

        detached = fsa.detach()
        assert detached.requires_grad is False
        assert fsa.requires_grad is True

        # the underlying memory is shared!
        assert detached.scores.data_ptr() == fsa.scores.data_ptr()


if __name__ == '__main__':
    unittest.main()
