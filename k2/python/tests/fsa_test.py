#!/usr/bin/env python3
#
# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                Guoguo Chen
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

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

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
        assert _remove_leading_spaces(expected_str) == \
                _remove_leading_spaces(k2.to_str_simple(fsa))

        arcs = fsa.arcs.values()[:, :-1]
        assert isinstance(arcs, torch.Tensor)
        assert arcs.dtype == torch.int32
        assert arcs.device.type == 'cpu'
        assert arcs.shape == (8, 3), 'there should be 8 arcs'
        assert torch.all(
            torch.eq(arcs[0], torch.tensor([0, 1, 2], dtype=torch.int32)))

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
            0 1  2  -1.2
            0 2  10 -2.2
            1 6 -1  -3.2
            1 3  3  -4.2
            2 6 -1  -5.2
            2 4  2  -6.2
            3 6 -1  -7.2
            5 0  1  -8.2
            6
        '''

        for i in range(4):
            if i == 0:
                fsa = k2.Fsa.from_str(s)
            elif i == 1:
                fsa = k2.Fsa.from_str(s, acceptor=True)
            elif i == 2:
                fsa = k2.Fsa.from_str(s, num_aux_labels=0)
            else:
                fsa = k2.Fsa.from_str(s, aux_label_names=[])

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
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa))

            arcs = fsa.arcs.values()[:, :-1]
            assert isinstance(arcs, torch.Tensor)
            assert arcs.dtype == torch.int32
            assert arcs.device.type == 'cpu'
            assert arcs.shape == (8, 3), 'there should be 8 arcs'
            assert torch.all(
                torch.eq(arcs[0], torch.tensor([0, 1, 2], dtype=torch.int32)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, -6.2, -7.2, -8.2],
                             dtype=torch.float32))

            fsa.scores *= -1

            assert torch.allclose(
                fsa.scores,
                torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
                             dtype=torch.float32))

            # test that assigning to labels calls _k2.fix_final_labels as it
            # should.
            fsa.labels = torch.tensor([-1, 10, 0, 1, -1, 1, 0, 2],
                                      dtype=torch.int32)
            assert torch.all(
                torch.eq(
                    fsa.labels,
                    torch.tensor([0, 10, -1, 1, -1, 1, -1, 2],
                                 dtype=torch.int32)))

    def test_acceptor_wo_arcs_from_str(self):
        s1 = '''
        '''

        s2 = '''
            0
            1
        '''

        s3 = '''
            1
        '''

        for device in self.devices:
            fsa1 = k2.Fsa.from_str(s1)
            self.assertEqual(k2.to_str_simple(fsa1), '')

            with self.assertRaises(ValueError):
                _ = k2.Fsa.from_str(s2)

            fsa3 = k2.Fsa.from_str(s3)
            self.assertEqual(fsa3.arcs.dim0(), 2)

    def test_acceptor_from_openfst(self):
        s = '''
            0 1  2 -1.2
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

        for i in range(5):
            if i == 0:
                fsa = k2.Fsa.from_openfst(s)
            elif i == 1:
                fsa = k2.Fsa.from_openfst(s, acceptor=True)
            elif i == 2:
                fsa = k2.Fsa.from_openfst(s, num_aux_labels=0)
            elif i == 3:
                # Test k2.Fsa.from_str(k2.to_str(openfst=True), openfst=True)
                fsa_tmp = k2.Fsa.from_str(s, acceptor=True, openfst=True)
                fsa_tmp_str = k2.to_str(fsa_tmp, openfst=True)
                fsa = k2.Fsa.from_str(fsa_tmp_str, acceptor=True, openfst=True)
            else:
                fsa = k2.Fsa.from_openfst(s, aux_label_names=[])

            expected_str = '''
            0 1 2 -1.2
            0 2 10 -2.2
            1 6 1 -3.2
            1 3 3 -4.2
            2 6 2 -5.2
            2 4 2 -6.2
            3 6 3 -7.2
            5 0 1 -8.2
            6 -9.2
            7 0
            '''
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa, openfst=True))

            arcs = fsa.arcs.values()[:, :-1]
            assert isinstance(arcs, torch.Tensor)
            assert arcs.dtype == torch.int32
            assert arcs.device.type == 'cpu'
            assert arcs.shape == (10, 3), 'there should be 10 arcs'
            assert torch.all(
                torch.eq(arcs[0], torch.tensor([0, 1, 2], dtype=torch.int32)))

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

    def test_acceptor_from_openfst_ragged1(self):
        s = '''
            0 1  2 [] -1.2
            0 2  10 [10] -2.2
            1 6  1 [] -3.2
            1 3  3 [11 12] -4.2
            2 6  2 [] -5.2
            2 4  2 [] -6.2
            3 6  3 [] -7.2
            5 0  1 [13]  -8.2
            7
            6 -9.2
        '''
        for i in range(2):
            if i == 0:
                fsa = k2.Fsa.from_openfst(s,
                                          num_aux_labels=0,
                                          ragged_label_names=['ragged'])
            else:
                # Test k2.Fsa.from_str(k2.to_str(openfst=True), openfst=True)
                fsa_tmp = k2.Fsa.from_str(s,
                                          num_aux_labels=0,
                                          ragged_label_names=['ragged'],
                                          openfst=True)
                fsa_tmp_str = k2.to_str(fsa_tmp, openfst=True)
                fsa = k2.Fsa.from_str(fsa_tmp_str,
                                      num_aux_labels=0,
                                      ragged_label_names=['ragged'],
                                      openfst=True)

        expected_str = '''
        0 1 2 [ ] -1.2
        0 2 10 [ 10 ] -2.2
        1 6 1 [ ] -3.2
        1 3 3 [ 11 12 ] -4.2
        2 6 2 [ ] -5.2
        2 4 2 [ ] -6.2
        3 6 3 [ ] -7.2
        5 0 1 [ 13 ] -8.2
        6 -9.2
        7 0
        '''
        string = _remove_leading_spaces(k2.to_str(fsa, openfst=True))
        print("fsa=", string)
        assert _remove_leading_spaces(expected_str) == string

    def test_acceptor_wo_arcs_from_openfst(self):
        s1 = '''
        '''

        s2 = '''
            0 Inf
            1 0.1
        '''

        s3 = '''
            0 Inf
            1 0.1
            2 0.2
        '''

        for device in self.devices:
            fsa1 = k2.Fsa.from_openfst(s1)
            print("fsa1 = ", k2.to_str_simple(fsa1))
            self.assertEqual('', k2.to_str_simple(fsa1))

            fsa2 = k2.Fsa.from_openfst(s2)
            self.assertEqual(_remove_leading_spaces(k2.to_str_simple(fsa2)),
                             "1 2 -1 -0.1\n2")
            arcs2 = fsa2.arcs.values()[:, :-1]
            assert torch.all(
                torch.eq(arcs2, torch.tensor([[1, 2, -1]], dtype=torch.int32)))

            fsa3 = k2.Fsa.from_openfst(s3)
            self.assertEqual(fsa3.arcs.dim0(), 4)
            self.assertEqual(_remove_leading_spaces(k2.to_str_simple(fsa3)),
                             "1 3 -1 -0.1\n2 3 -1 -0.2\n3")

    def test_transducer_from_tensor(self):
        for device in self.devices:
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
            assert torch.all(
                torch.eq(
                    fsa.aux_labels,
                    torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                                 dtype=torch.int32).to(device)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([-1.2, -2.2, -4.2, -3.2, -5.2, -6.2, -7.2, -8.2],
                             dtype=torch.float32,
                             device=device))

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
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa))

    def test_transducer_from_str(self):
        s = '''
            0 1  2  22  -1.2
            0 2  10 100 -2.2
            1 6 -1  16  -4.2
            1 3  3  33  -3.2
            2 6 -1  26  -5.2
            2 4  2  22  -6.2
            3 6 -1  36  -7.2
            5 0  1  50  -8.2
            6
        '''
        for i in range(3):
            if i == 0:
                fsa = k2.Fsa.from_str(s, num_aux_labels=1)
            elif i == 1:
                fsa = k2.Fsa.from_str(s, acceptor=False)
            else:
                fsa = k2.Fsa.from_str(s, aux_label_names=['aux_labels'])
            assert fsa.aux_labels.dtype == torch.int32
            assert fsa.aux_labels.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels,
                    torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                                 dtype=torch.int32)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([-1.2, -2.2, -4.2, -3.2, -5.2, -6.2, -7.2, -8.2],
                             dtype=torch.float32))

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
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa))

    def test_transducer2_from_str(self):
        s = '''
            0 1  2  22  101 -1.2
            0 2  10 100 102 -2.2
            1 6 -1  16  103 -4.2
            1 3  3  33  104 -3.2
            2 6 -1  26  105 -5.2
            2 4  2  22  106 -6.2
            3 6 -1  36  107 -7.2
            5 0  1  50  108 -8.2
            6
        '''
        for i in range(2):
            if i == 0:
                fsa = k2.Fsa.from_str(s, num_aux_labels=2)
            else:
                fsa = k2.Fsa.from_str(
                    s, aux_label_names=['aux_labels', 'aux_labels2'])
            assert fsa.aux_labels.dtype == torch.int32
            assert fsa.aux_labels.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels,
                    torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                                 dtype=torch.int32)))
            assert torch.all(
                torch.eq(
                    fsa.aux_labels2,
                    torch.tensor([101, 102, 103, 104, 105, 106, 107, 108],
                                 dtype=torch.int32)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([-1.2, -2.2, -4.2, -3.2, -5.2, -6.2, -7.2, -8.2],
                             dtype=torch.float32))

            # only aux_labels will be printed right now..
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
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa))

    def test_transducer2_ragged2_from_str(self):
        s = '''
            0 1  2  22  101 [] [] -1.2
            0 2  10 100 102 [] [] -2.2
            1 6 -1  16  103 [20 30] [40] -4.2
            1 3  3  33  104 [] [] -3.2
            2 6 -1  26  105 [] [] -5.2
            2 4  2  22  106 [] [] -6.2
            3 6 -1  36  107 [] [] -7.2
            5 0  1  50  108 [] [] -8.2
            6
        '''
        fsa = k2.Fsa.from_str(s,
                              aux_label_names=['aux_labels', 'aux_labels2'],
                              ragged_label_names=['ragged1', 'ragged2'])

        assert fsa.aux_labels.dtype == torch.int32
        assert fsa.aux_labels.device.type == 'cpu'
        assert isinstance(fsa.ragged1, k2.RaggedTensor)
        assert isinstance(fsa.ragged2, k2.RaggedTensor)

        assert torch.all(
            torch.eq(
                fsa.aux_labels,
                torch.tensor([22, 100, 16, 33, 26, 22, 36, 50],
                             dtype=torch.int32)))

        assert torch.all(
            torch.eq(
                fsa.aux_labels2,
                torch.tensor([101, 102, 103, 104, 105, 106, 107, 108],
                             dtype=torch.int32)))

        assert torch.allclose(
            fsa.scores,
            torch.tensor([-1.2, -2.2, -4.2, -3.2, -5.2, -6.2, -7.2, -8.2],
                         dtype=torch.float32))

        print("fsa.ragged1 = ", fsa.ragged1)
        print("fsa.ragged2 = ", fsa.ragged2)
        assert fsa.ragged1 == k2.RaggedTensor(
            '[ [] [] [20 30] [] [] [] [] [] ]')
        assert fsa.ragged2 == k2.RaggedTensor('[ [] [] [40] [] [] [] [] [] ]')

        # only aux_labels will be printed right now..
        expected_str = '''
        0 1 2 22 101 [ ] [ ] -1.2
        0 2 10 100 102 [ ] [ ] -2.2
        1 6 -1 16 103 [ 20 30 ] [ 40 ] -4.2
        1 3 3 33 104 [ ] [ ] -3.2
        2 6 -1 26 105 [ ] [ ] -5.2
        2 4 2 22 106 [ ] [ ] -6.2
        3 6 -1 36 107 [ ] [ ] -7.2
        5 0 1 50 108 [ ] [ ] -8.2
        6
        '''
        print("fsa = ", _remove_leading_spaces(k2.to_str(fsa)))
        assert _remove_leading_spaces(expected_str) == \
              _remove_leading_spaces(k2.to_str(fsa))

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
        for i in range(4):
            if i == 0:
                fsa = k2.Fsa.from_openfst(s, acceptor=False)
            elif i == 1:
                fsa = k2.Fsa.from_openfst(s, num_aux_labels=1)
            elif i == 2:
                # Test k2.Fsa.from_str(k2.to_str(openfst=True), openfst=True)
                fsa_tmp = k2.Fsa.from_str(s, acceptor=False, openfst=True)
                fsa_tmp_str = k2.to_str(fsa_tmp, openfst=True)
                fsa = k2.Fsa.from_str(fsa_tmp_str, acceptor=False, openfst=True)
            else:
                fsa = k2.Fsa.from_openfst(s, aux_label_names=['aux_labels'])

            assert fsa.aux_labels.dtype == torch.int32
            assert fsa.aux_labels.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels,
                    torch.tensor([22, 100, 16, 33, 26, 22, 36, 50, -1, -1],
                                 dtype=torch.int32)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([1.2, 2.2, 4.2, 3.2, 5.2, 6.2, 7.2, 8.2, 0, 9.2],
                             dtype=torch.float32))

            expected_str = '''
                0 1 2 22 -1.2
                0 2 10 100 -2.2
                1 6 1 16 -4.2
                1 3 3 33 -3.2
                2 6 2 26 -5.2
                2 4 2 22 -6.2
                3 6 3 36 -7.2
                5 0 1 50 -8.2
                6 0
                7 -9.2
            '''
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa, openfst=True))

    def test_transducer3_from_openfst(self):
        s = '''
            0 1  2  22  33  44  -1.2
            0 2  10 100 101 102 -2.2
            1 6  1  16  17  18  -4.2
            1 3  3  33  34  35  -3.2
            2 6  2  26  27  28  -5.2
            2 4  2  22  23  24  -6.2
            3 6  3  36  37  38  -7.2
            5 0  1  50  51  52  -8.2
            7 -9.2
            6
        '''
        for i in range(3):
            if i == 0:
                fsa = k2.Fsa.from_openfst(s, num_aux_labels=3)
            elif i == 1:
                # Test k2.Fsa.from_str(k2.to_str(openfst=True), openfst=True)
                fsa_tmp = k2.Fsa.from_str(s, num_aux_labels=3, openfst=True)
                fsa_tmp_str = k2.to_str(fsa_tmp, openfst=True)
                fsa = k2.Fsa.from_str(fsa_tmp_str,
                                      num_aux_labels=3,
                                      openfst=True)
            else:
                fsa = k2.Fsa.from_openfst(s,
                                          aux_label_names=[
                                              'aux_labels', 'aux_labels2',
                                              'aux_labels3'
                                          ])

            assert fsa.aux_labels.dtype == torch.int32
            assert fsa.aux_labels.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels,
                    torch.tensor([22, 100, 16, 33, 26, 22, 36, 50, -1, -1],
                                 dtype=torch.int32)))

            assert fsa.aux_labels2.dtype == torch.int32
            assert fsa.aux_labels2.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels2,
                    torch.tensor([33, 101, 17, 34, 27, 23, 37, 51, -1, -1],
                                 dtype=torch.int32)))

            assert fsa.aux_labels3.dtype == torch.int32
            assert fsa.aux_labels3.device.type == 'cpu'
            assert torch.all(
                torch.eq(
                    fsa.aux_labels3,
                    torch.tensor([44, 102, 18, 35, 28, 24, 38, 52, -1, -1],
                                 dtype=torch.int32)))

            assert torch.allclose(
                fsa.scores,
                torch.tensor([1.2, 2.2, 4.2, 3.2, 5.2, 6.2, 7.2, 8.2, 0, 9.2],
                             dtype=torch.float32))

            expected_str = '''
                0 1 2 22 -1.2
                0 2 10 100 -2.2
                1 6 1 16 -4.2
                1 3 3 33 -3.2
                2 6 2 26 -5.2
                2 4 2 22 -6.2
                3 6 3 36 -7.2
                5 0 1 50 -8.2
                6 0
                7 -9.2
            '''
            assert _remove_leading_spaces(expected_str) == \
                    _remove_leading_spaces(k2.to_str_simple(fsa, openfst=True))

    def test_fsa_io(self):
        s = '''
            0 1 10 0.1
            0 2 20 0.2
            1 3 -1 0.3
            2 3 -1 0.4
            3
        '''
        fsa = k2.Fsa.from_str(s)
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
        fsa = k2.Fsa.from_str(rules, num_aux_labels=1)
        fsa.labels_sym = symbols
        fsa.aux_labels_sym = aux_symbols

        import shutil
        if shutil.which('dot') is not None:
            fsa.draw(filename='foo.png')
            os.remove('foo.png')

    def test_to(self):
        s = '''
            0 1 -1 1
            1
        '''
        fsa = k2.Fsa.from_str(s)
        assert fsa.is_cpu()

        if torch.cuda.is_available() and k2.with_cuda:
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
            3
        '''
        fsa0 = k2.Fsa.from_str(s0, num_aux_labels=1).requires_grad_(True)
        fsa1 = k2.Fsa.from_str(s1, num_aux_labels=1).requires_grad_(True)

        fsa0.invert_()
        print("str(fsa0) == ", str(fsa0))
        print("str(fsa1) == ", str(fsa1))
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
        fsa = k2.Fsa.from_str(s, num_aux_labels=1)
        fsa.labels_sym = symbol_table
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
        assert fsa.labels_sym.get('a') == 1
        assert fsa.labels_sym.get(1) == 'a'

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
        fsa1 = k2.Fsa.from_str(s1, num_aux_labels=1)
        fsa2 = k2.Fsa.from_str(s2, num_aux_labels=1)
        fsa = k2.create_fsa_vec([fsa1, fsa2])
        del fsa1, fsa2

        sym_str = '''
            a 1
        '''
        symbol_table = k2.SymbolTable.from_str(sym_str)
        fsa.labels_sym = symbol_table
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
        assert fsa.labels_sym.get('a') == 1
        assert fsa.labels_sym.get(1) == 'a'

    def test_fsa_vec_as_dict_ragged(self):
        r = k2.RaggedTensor(
            k2.RaggedShape('[ [ x x ] [x] [ x x ] [x]]'),
            torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int32))
        g = k2.Fsa.from_str('0  1  3  0.0\n  1 2 -1 0.0\n  2')
        h = k2.create_fsa_vec([g, g])
        h.aux_labels = r
        assert (h[0].aux_labels.dim0 == h[0].labels.shape[0])

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

    def test_create_fsa_vec(self):
        s1 = '''
            0 1 1 0.1
            1 2 -1 0.2
            2
        '''

        s2 = '''
            0 1 -1 10
            1
        '''
        fsa1 = k2.Fsa.from_str(s1)
        fsa1.aux_labels = k2.RaggedTensor('[ [1 0 2] [3 5] ]')
        fsa2 = k2.Fsa.from_str(s2)
        fsa2.aux_labels = k2.RaggedTensor('[ [5 8 9] ]')
        fsa = k2.create_fsa_vec([fsa1, fsa2])
        assert fsa.aux_labels == k2.RaggedTensor('[ [ 1 0 2 ] [ 3 5 ] [ 5 8 9 ] ]')  # noqa

        fsa = k2.Fsa.from_fsas([fsa1, fsa2])
        assert fsa.aux_labels == k2.RaggedTensor(
            '[ [ 1 0 2 ] [ 3 5 ] [ 5 8 9 ] ]')

    def test_index_fsa(self):
        for device in self.devices:
            s1 = '''
                0 1 1 0.1
                1 2 -1 0.2
                2
            '''
            s2 = '''
                0 1 -1 1.0
                1
            '''
            fsa1 = k2.Fsa.from_str(s1)
            fsa1.tensor_attr = torch.tensor([10, 20], dtype=torch.int32)
            fsa1.ragged_attr = k2.RaggedTensor([[11, 12], [21, 22, 23]])

            fsa2 = k2.Fsa.from_str(s2)
            fsa2.tensor_attr = torch.tensor([100], dtype=torch.int32)
            fsa2.ragged_attr = k2.RaggedTensor([[111]])

            fsa1 = fsa1.to(device)
            fsa2 = fsa2.to(device)

            fsa_vec = k2.create_fsa_vec([fsa1, fsa2])

            single1 = k2.index_fsa(
                fsa_vec, torch.tensor([0], dtype=torch.int32, device=device))
            assert torch.all(torch.eq(fsa1.tensor_attr, single1.tensor_attr))
            assert str(single1.ragged_attr) == str(fsa1.ragged_attr)
            assert single1.device == device

            single2 = k2.index_fsa(
                fsa_vec, torch.tensor([1], dtype=torch.int32, device=device))
            assert torch.all(torch.eq(fsa2.tensor_attr, single2.tensor_attr))
            assert str(single2.ragged_attr) == str(fsa2.ragged_attr)
            assert single2.device == device

            multiples = k2.index_fsa(
                fsa_vec,
                torch.tensor([0, 1, 0, 1, 1], dtype=torch.int32,
                             device=device))
            assert multiples.shape == (5, None, None)
            assert torch.all(
                torch.eq(
                    multiples.tensor_attr,
                    torch.cat(
                        (fsa1.tensor_attr, fsa2.tensor_attr, fsa1.tensor_attr,
                         fsa2.tensor_attr, fsa2.tensor_attr))))
            assert str(multiples.ragged_attr) == str(
                k2.ragged.cat([
                    fsa1.ragged_attr, fsa2.ragged_attr, fsa1.ragged_attr,
                    fsa2.ragged_attr, fsa2.ragged_attr
                ],
                              axis=0))  # noqa
            assert multiples.device == device

    def test_clone(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                1 2 2 0.2
                2 3 -1 0.3
                3
            '''
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.non_tensor_attr1 = [10]
            fsa.tensor_attr1 = torch.tensor([10, 20, 30]).to(device)
            fsa.ragged_attr1 = k2.RaggedTensor('[[100] [] [-1]]').to(device)

            fsa._cache['abc'] = [100]

            cloned = fsa.clone()

            fsa.non_tensor_attr1[0] = 0
            fsa.tensor_attr1[0] = 0
            fsa.ragged_attr1 = k2.RaggedTensor('[[] [] [-1]]').to(device)
            fsa._cache['abc'][0] = 1

            # we assume that non-tensor attributes are readonly
            # and are shared
            self.assertEqual(cloned.non_tensor_attr1, [0])
            self.assertEqual(cloned._cache['abc'], [1])

            assert torch.all(
                torch.eq(cloned.tensor_attr1,
                         torch.tensor([10, 20, 30]).to(device)))

            assert cloned.ragged_attr1 == \
                    k2.RaggedTensor('[[100] [] [-1]]', device=device)

    def test_detach_more_attributes(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                1 2 2 0.2
                2 3 -1 0.3
                3
            '''

            fsa = k2.Fsa.from_str(s).to(device).requires_grad_(True)
            fsa.non_tensor_attr1 = [10]
            fsa.tensor_attr1 = torch.tensor([10., 20, 30],
                                            device=device,
                                            requires_grad=True)
            fsa.ragged_attr1 = k2.RaggedTensor('[[100] [] [-1]]').to(device)
            fsa._cache['abc'] = [100]

            detached = fsa.detach()
            fsa._cache['abc'][0] = 1
            fsa.non_tensor_attr1[0] = 0

            assert id(detached.non_tensor_attr1) == id(fsa.non_tensor_attr1)
            assert detached.tensor_attr1.requires_grad is False
            assert torch.all(torch.eq(fsa.tensor_attr1, detached.tensor_attr1))
            assert str(fsa.ragged_attr1) == str(detached.ragged_attr1)

            self.assertEqual(detached.non_tensor_attr1, [0])
            self.assertEqual(detached._cache['abc'], [1])

    def test_convert_attr_to_ragged(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                1 2 2 0.2
                2 3 -1 0.3
                3
            '''
            fsa = k2.Fsa.from_str(s).to(device)
            fsa.tensor_attr1 = torch.tensor([1, 2, 3, 4, 0, 6],
                                            dtype=torch.int32,
                                            device=device)[::2]
            fsa.convert_attr_to_ragged_(name='tensor_attr1', remove_eps=False)
            expected = k2.RaggedTensor('[ [1] [3] [0] ]', device=device)
            assert fsa.tensor_attr1 == expected

            fsa.tensor_attr2 = torch.tensor([1, 0, -1],
                                            dtype=torch.int32,
                                            device=device)
            fsa.convert_attr_to_ragged_(name='tensor_attr2', remove_eps=True)
            expected = k2.RaggedTensor('[ [1] [] [-1] ]', device=device)
            assert fsa.tensor_attr2 == expected

    def test_invalidate_cache(self):
        s = '''
            0 1 1 0.1
            1 2 -1 0.2
            2
        '''
        fsa = k2.Fsa.from_str(s)
        fsa = k2.create_fsa_vec([fsa])
        fsa.get_tot_scores(True, True)

        assert 'forward_scores_double_log' in fsa._cache
        assert 'state_batches' in fsa._cache

        fsa.scores *= 2

        assert 'forward_scores_double_log' not in fsa._cache
        assert 'state_batches' in fsa._cache

    def test_modify_fsa_label(self):
        s = """
            0 1 1 0.1
            1 2 2 0.2
            2 3 -1 0.3
            3
        """
        fsa = k2.Fsa.from_str(s)
        fsa.labels[0] = 4
        with self.assertRaises(RuntimeError):
            k2.arc_sort(fsa)


if __name__ == '__main__':
    unittest.main()
