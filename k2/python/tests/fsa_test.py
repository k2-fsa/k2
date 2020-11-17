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


def _remove_leading_spaces(s: str) -> str:
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    return '\n'.join(lines)


class TestFsa(unittest.TestCase):

    def test_acceptor_from_tensor(self):
        fsa_tensor = torch.tensor(
            [[0, 1, 2, _k2._float_as_int(-1.2)],
             [0, 2, 10, _k2._float_as_int(-2.2)],
             [1, 6, -1, _k2._float_as_int(-3.2)],
             [1, 3, 3, _k2._float_as_int(-4.2)],
             [2, 6, -1, _k2._float_as_int(-5.2)],
             [2, 4, 2, _k2._float_as_int(-6.2)],
             [3, 6, -1, _k2._float_as_int(-7.2)],
             [5, 0, 1, _k2._float_as_int(-8.2)]],
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
        device_id = 0
        device = torch.device('cuda', device_id)
        fsa_tensor = torch.tensor(
            [[0, 1, 2, _k2._float_as_int(-1.2)],
             [0, 2, 10, _k2._float_as_int(-2.2)],
             [1, 6, -1, _k2._float_as_int(-4.2)],
             [1, 3, 3, _k2._float_as_int(-3.2)],
             [2, 6, -1, _k2._float_as_int(-5.2)],
             [2, 4, 2, _k2._float_as_int(-6.2)],
             [3, 6, -1, _k2._float_as_int(-7.2)],
             [5, 0, 1, _k2._float_as_int(-8.2)]],
            dtype=torch.int32).to(device)
        aux_labels_tensor = torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                                         dtype=torch.int32).to(device)
        fsa = k2.Fsa(fsa_tensor, aux_labels_tensor)
        assert fsa.aux_labels.dtype == torch.int32
        assert fsa.aux_labels.device.type == 'cuda'
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
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            k2.to_str(fsa))

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

        ragged_arc = _k2._fsa_to_fsa_vec(fsa.arcs)
        del fsa
        fsa_vec = k2.Fsa.from_ragged_arc(ragged_arc)
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
        dot = k2.to_dot(fsa)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            dot.render(filename='fsa',
                       directory=tmp_dir,
                       format='pdf',
                       cleanup=True)
            # the fsa is saved to tmp_dir/fsa.pdf
            import os
            os.system('ls -l {}/fsa.pdf'.format(tmp_dir))

    def test_to(self):
        s = '''
            0 1 -1 1
            1
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.is_cpu()

        device = torch.device('cuda', 0)
        fsa.to_(device)
        assert fsa.is_cuda()
        assert fsa.device == device

        device = torch.device('cpu')
        fsa.to_(device)
        assert fsa.is_cpu()
        assert fsa.device == device

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


if __name__ == '__main__':
    unittest.main()
