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
        expected_arcs = torch.tensor([[0, 1, 1], [1, 2, 2], [2, 3,
                                                             -1]]).to(fsa.arcs)
        assert torch.all(torch.eq(arcs[:, :3], expected_arcs))

        scores = fsa.scores
        expected_scores = torch.tensor([0.1, 0.2, 0.3]).to(scores)
        assert torch.allclose(scores, expected_scores)

        # test attribute

        with self.assertRaises(AttributeError):
            attr1 = fsa.attr1
            del attr1

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

    def test_get_forward_scores_simple_fsa_case_1(self):
        # see https://git.io/JtttZ
        s = """
            0 1 1 0.0
            0 1 2 0.1
            0 2 3 2.2
            1 2 4 0.5
            1 2 5 0.6
            1 3 -1 3.0
            2 3 -1 0.8
            3
        """
        for device in self.devices:
            if device.type == "cuda":
                continue  # no implmented yet
            for use_double_scores in (True, False):
                #  fsa = k2r.Fsa(s).to(device).requires_grad_(True)
                # TODO(fangjun): Implement `To` and autograd
                fsa = k2r.Fsa(s).requires_grad_(True)
                fsa_vec = k2r.Fsa.from_fsas([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=False)
                expected_forward_scores = torch.tensor([
                    0,  # start state
                    0.1,  # state 1, arc: 0 -> 1 (2/0.1)
                    2.2,  # state 2, arc: 0 -> 2 (3/2.2)
                    3.1,  # state 3, arc: 1 -> 3 (-1/3.0)
                ]).to(forward_scores)
                assert torch.allclose(forward_scores, expected_forward_scores)
                scale = torch.arange(forward_scores.numel()).to(device)
                (scale * forward_scores).sum().backward()
                expected_grad = torch.tensor([0, 4, 2, 0, 0, 3,
                                              0]).to(fsa.grad)
                assert torch.allclose(fsa.grad, expected_grad)

                # now for log semiring
                fsa.grad = None
                fsa_vec = k2r.Fsa.from_fsas([fsa])
                forward_scores = fsa_vec.get_forward_scores(
                    use_double_scores=use_double_scores, log_semiring=True)
                scores = fsa.scores.detach().clone().requires_grad_(True)
                expected_forward_scores = torch.empty_like(forward_scores)
                expected_forward_scores[0] = 0
                expected_forward_scores[1] = scores[:2].exp().sum().log()
                expected_forward_scores[2] = (
                    scores[2].exp() +
                    (expected_forward_scores[1] + scores[3]).exp() +
                    (expected_forward_scores[1] + scores[4]).exp()).log()
                expected_forward_scores[3] = (
                    (expected_forward_scores[1] + scores[5]).exp() +
                    (expected_forward_scores[2] + scores[6]).exp()).log()
                assert torch.allclose(forward_scores, expected_forward_scores)

                (scale * forward_scores).sum().backward()
                (scale * expected_forward_scores).sum().backward()
                assert torch.allclose(fsa.grad, scores.grad)


if __name__ == "__main__":
    unittest.main()
