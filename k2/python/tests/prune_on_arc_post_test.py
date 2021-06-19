#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R prune_on_arc_post_test_py

import unittest

import k2
import torch


class TestPruneOnArcPost(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        for device in self.devices:
            s = '''
                0 1 1 0.2
                0 1 2 0.1
                1 2 2 0.1
                1 2 3 0.2
                2 3 -1 0
                3
            '''
            fsa = k2.Fsa.from_str(s)
            fsa_vec = k2.create_fsa_vec([fsa])
            threshold_prob = 0.5
            ans = k2.prune_on_arc_post(fsa_vec,
                                       threshold_prob,
                                       use_double_scores=True)
            expected = k2.Fsa.from_str('''
                0 1 1 0.2
                1 2 3 0.2
                2 3 -1 0
                3
            ''')
            assert str(ans[0]) == str(expected)


if __name__ == '__main__':
    unittest.main()
