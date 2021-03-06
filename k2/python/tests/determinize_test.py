#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
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
#  ctest --verbose -R determinize_test_py

import unittest

import k2


class TestDeterminize(unittest.TestCase):

    def test1(self):
        s = '''
            0 4 1 1
            0 1 1 1
            1 2 2 2
            1 3 3 3
            2 7 1 4
            3 7 1 5
            4 6 1 2
            4 6 1 3
            4 5 1 3
            4 8 -1 2
            5 8 -1 4
            6 8 -1 3
            7 8 -1 5
            8
        '''
        fsa = k2.Fsa.from_str(s)
        prop = fsa.properties
        self.assertFalse(
            prop & k2.fsa_properties.ARC_SORTED_AND_DETERMINISTIC != 0)
        dest = k2.determinize(fsa)
        log_semiring = False
        self.assertTrue(k2.is_rand_equivalent(fsa, dest, log_semiring))
        arc_sorted = k2.arc_sort(dest)
        prop = arc_sorted.properties
        self.assertTrue(
            prop & k2.fsa_properties.ARC_SORTED_AND_DETERMINISTIC != 0)


if __name__ == '__main__':
    unittest.main()
