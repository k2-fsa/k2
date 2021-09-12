#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation      (authors: Fangjun Kuang)
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
#  ctest --verbose -R fsa_v2_test_py

import unittest

import k2
import k2.ragged as k2r
import torch


class TestFsa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test_init_acceptor(self):
        s = """
            0 1 1  0.1
            1 2 2  0.2
            2 3 -1  0.3
            3
        """
        fsa = k2r.Fsa(s)

        arcs = fsa.arcs
        expected_arcs = torch.tensor([[0, 1, 1], [1, 2, 2], [2, 3, -1]]).to(fsa.arcs)
        assert torch.all(torch.eq(arcs[:, :3], expected_arcs))

        scores = fsa.scores
        expected_scores = torch.tensor([0.1, 0.2, 0.3]).to(scores)
        assert torch.allclose(scores, expected_scores)

        # test attribute

        with self.assertRaises(AttributeError):
            attr1 = fsa.attr1

        # Non-tensor attribute
        assert hasattr(fsa, "attr1") is False
        fsa.attr1 = 10
        fsa.attr2 = 2.5

        assert hasattr(fsa, "attr1") is True
        assert fsa.attr1 == 10
        assert fsa.attr2 == 2.5

        with self.assertRaises(AttributeError):
            del fsa.attr3

        del fsa.attr2
        assert hasattr(fsa, "attr2") is False

        # tensor attribute
        with self.assertRaises(RuntimeError):
            # Number of elements does not match the number of arcs,
            # so it throws a RuntimeError exception
            fsa.t1 = torch.tensor([1, -1.5])

        # 1-D
        fsa.t1 = torch.tensor([1, -1.5, 2.5])
        fsa.t2 = torch.rand(fsa.arcs.size(0), 10)

        expected_t1 = torch.tensor([1, -1.5, 2.5])
        assert torch.allclose(fsa.t1, expected_t1)

        del fsa.t2
        assert hasattr(fsa, "t2") is False

        # ragged tensors
        with self.assertRaises(RuntimeError):
            # Number of elements does not match the number of arcs,
            # so it throws a RuntimeError exception
            fsa.r1 = k2.RaggedTensor([[], [10]])

        # Now the tensor attribute 't1' is replaced with a ragged tensor
        fsa.t1 = k2.RaggedTensor([[], [10], [2, 3, 5.5]])
        expected_t1 = k2.RaggedTensor([[], [10], [2, 3, 5.5]])
        assert fsa.t1 == expected_t1


if __name__ == "__main__":
    unittest.main()
