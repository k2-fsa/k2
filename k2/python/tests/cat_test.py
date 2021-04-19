#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R cat_test_py

import unittest

import k2
import torch


class TestCat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))

    def test_cat_fsa_vec(self):
        for device in self.devices:
            s = '''
                0 1 1 0.1
                0 1 2 0.2
                1 2 -1 0.3
                2
            '''
            fsa1 = k2.Fsa.from_str(s).to(device)
            fsa1.tensor_attr1 = torch.tensor([1, 2, 3]).to(device)
            fsa1.tensor_attr2 = torch.tensor([4, 5, 6]).to(device)
            fsa1.non_tensor_attr1 = 'fsa1'

            fsa1.ragged_tensor_attr1 = \
                    k2.RaggedInt('[[1 2] [] [3 4 5]]').to(device)
            fsa1.ragged_tensor_attr2 = \
                    k2.RaggedInt('[[1 20] [30] [5]]').to(device)

            fsa2 = k2.Fsa.from_str(s).to(device)
            fsa2.tensor_attr1 = torch.tensor([10, 20, 30]).to(device)
            fsa2.tensor_attr3 = torch.tensor([40, 50, 60]).to(device)
            fsa2.non_tensor_attr1 = 'fsa'
            fsa2.non_tensor_attr2 = 'fsa2'

            fsa2.ragged_tensor_attr1 = \
                    k2.RaggedInt('[[3] [4 5] [6 7]]').to(device)
            fsa2.ragged_tensor_attr3 = \
                    k2.RaggedInt('[[1 0] [0] [-1]]').to(device)

            fsa_vec1 = k2.create_fsa_vec([fsa1])
            fsa_vec2 = k2.create_fsa_vec([fsa2])
            fsa_vec = k2.cat([fsa_vec1, fsa_vec2])

            assert str(fsa_vec[0].arcs) == str(fsa1.arcs)
            assert str(fsa_vec[1].arcs) == str(fsa2.arcs)
            assert not hasattr(fsa_vec, 'tensor_attr2')
            assert not hasattr(fsa_vec, 'tensor_attr3')

            assert fsa_vec.non_tensor_attr1 == fsa1.non_tensor_attr1
            assert fsa_vec.non_tensor_attr2 == fsa2.non_tensor_attr2
            assert torch.all(
                torch.eq(fsa_vec.tensor_attr1,
                         torch.tensor([1, 2, 3, 10, 20, 30]).to(device)))

            assert str(fsa_vec.ragged_tensor_attr1) == \
                    str(k2.RaggedInt('[[1 2] [] [3 4 5] [3] [4 5] [6 7]]'))

            assert not hasattr(fsa_vec, 'ragged_tensor_attr2')
            assert not hasattr(fsa_vec, 'ragged_tensor_attr3')


if __name__ == '__main__':
    unittest.main()
