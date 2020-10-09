#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R fsa_test_py -E host

import unittest

import k2
import torch

import _k2  # for test only, users should not import it.


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
             [5, 0, 1, _k2._float_as_int(-8.2)]], dtype=torch.int32)

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
            fsa.to_str())

        arcs = fsa.arcs
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.weights,
            torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        fsa = fsa.to('cuda')
        arcs[0][0] += 10
        assert arcs[0][0] == 10, 'arcs should still be accessible'

        arcs = fsa.arcs
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cuda'
        assert arcs.device.index == 0
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(
            arcs[1],
            torch.tensor([0, 2, 10], dtype=torch.int32, device=arcs.device))

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
            fsa.to_str())

        arcs = fsa.arcs
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.weights,
            torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        fsa = fsa.to('cuda')
        arcs[0][0] += 10
        assert arcs[0][0] == 10, 'arcs should still be accessible'

        arcs = fsa.arcs
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cuda'
        assert arcs.device.index == 0
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.allclose(
            arcs[1],
            torch.tensor([0, 2, 10], dtype=torch.int32, device=arcs.device))

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
            fsa.to_str(openfst=True))

        arcs = fsa.arcs
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (10, 3), 'there should be 10 arcs'
        assert torch.allclose(arcs[0],
                              torch.tensor([0, 1, 2], dtype=torch.int32))

        assert torch.allclose(
            fsa.weights,
            torch.tensor(
                [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 0],
                dtype=torch.float32))

        fsa = fsa.to('cuda')
        arcs[0][0] += 10
        assert arcs[0][0] == 10, 'arcs should still be accessible'

        arcs = fsa.arcs
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cuda'
        assert arcs.device.index == 0
        assert arcs.shape == (10, 3), 'there should be 10 arcs'
        assert torch.allclose(
            arcs[1],
            torch.tensor([0, 2, 10], dtype=torch.int32, device=arcs.device))

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
             [5, 0, 1, _k2._float_as_int(-8.2)]], dtype=torch.int32).to(device)
        aux_labels_tensor = torch.tensor(
            [22, 100, 16, 33, 26, 22, 36, 50], dtype=torch.int32).to(device)
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
            fsa.to('cpu').to_str())    # String IO supported on CPU only

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
            fsa.to_str())

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
            torch.tensor([22, 100, 16, 33, 26, 22, 36, 50, 0, 0],
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
            6 8 -1 0 -0
            7 8 -1 0 -9.2
            8
        '''
        assert _remove_leading_spaces(expected_str) == _remove_leading_spaces(
            fsa.to_str(openfst=True))

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
        isym = k2.SymbolTable.from_str(isym_str)
        osym = k2.SymbolTable.from_str(osym_str)

        rules = '''
            0 1 1 1 0.5
            0 1 2 2 1.5
            1 2 3 3  2.5
            2 3 -1 0 3.5
            3
        '''
        fsa = k2.Fsa.from_str(_remove_leading_spaces(rules))
        fsa.set_isymbol(isym)
        fsa.set_osymbol(osym)
        dot = fsa.to_dot()
        dot.render('/tmp/fsa', format='pdf')
        # the fsa is saved to /tmp/fsa.pdf


if __name__ == '__main__':
    unittest.main()
