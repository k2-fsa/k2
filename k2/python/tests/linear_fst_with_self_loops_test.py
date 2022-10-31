#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corporation      (authors: Liyong Guo)
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
#  ctest --verbose -R linear_fst_self_loops_test_py

import torch
import k2
import unittest


class TestLinearFstSelfLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test(self):
        s0 = """
          0 1 1 10 -0.1
          1 2 2 20 -0.2
          2 3 3 0 -0.5
          3 4 0 40 -0.1
          4 5 0 50 -0.1
          5 6 0 0 -0.1
          6 7 -1 0 -0.1
          7
        """

        s1 = """
          0 1 1 10 -0.1
          1 2 2 20 -0.2
          2 3 0 0 -0.5
          3 4 0 40 -0.1
          4 5 0 50 -0.1
          5 6 0 0 -0.1
          6 7 -1 0 -0.1
          7
        """

        s2 = """
          0 1 0 10 -0.5
          1 2 20 20 -0.2
          2 3 30 0 -0.5
          3 4 0 40 -0.1
          4 5 0 50 -0.1
          5 6 6 0 -0.1
          6 7 -1 0 -0.1
          7
        """

        s3 = """
            0 1 0 1 0.1
            1 2 0 2 0.2
            2 3 0 3 0.3
            3 4 0 4 0.4
            4 5 0 5 0.5
            5 6 -1 6 0.6
            6
        """

        for device in self.devices:
            fst0 = k2.Fsa.from_str(s0, acceptor=False).to(device)
            fst1 = k2.Fsa.from_str(s1, acceptor=False).to(device)
            fst2 = k2.Fsa.from_str(s2, acceptor=False).to(device)
            fst3 = k2.Fsa.from_str(s3, acceptor=False).to(device)
            fstv = k2.create_fsa_vec([fst0, fst1, fst2, fst3])
            for src_fst in [fst0, fst1, fst2, fst3, fstv]:
                expected_fst = k2.add_epsilon_self_loops(
                    k2.remove_epsilon(src_fst.to("cpu"))
                ).to(device)
                out_fst = k2.linear_fst_with_self_loops(src_fst)
                assert torch.all(torch.eq(out_fst.labels, expected_fst.labels))
                assert out_fst.aux_labels == expected_fst.aux_labels
                assert torch.allclose(out_fst.scores, expected_fst.scores)


if __name__ == "__main__":
    unittest.main()
