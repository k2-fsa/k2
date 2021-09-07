#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu, Wei Kang)
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
        # test weight pushing tropical
        dest_max = k2.determinize(
            fsa, k2.DeterminizeWeightPushingType.kTropicalWeightPushing)
        self.assertTrue(k2.is_rand_equivalent(dest, dest_max, log_semiring))
        # test weight pushing log
        dest_log = k2.determinize(
            fsa, k2.DeterminizeWeightPushingType.kLogWeightPushing)
        self.assertTrue(k2.is_rand_equivalent(dest, dest_log, log_semiring))

    def test_random(self):
        while True:
            fsa = k2.random_fsa(max_symbol=20,
                                min_num_arcs=50,
                                max_num_arcs=500)
            fsa = k2.arc_sort(k2.connect(k2.remove_epsilon(fsa)))
            prob = fsa.properties
            # we need non-deterministic fsa
            if not prob & k2.fsa_properties.ARC_SORTED_AND_DETERMINISTIC:
                break
        log_semiring = False
        # test weight pushing tropical
        dest_max = k2.determinize(
            fsa, k2.DeterminizeWeightPushingType.kTropicalWeightPushing)
        self.assertTrue(
            k2.is_rand_equivalent(fsa, dest_max, log_semiring, delta=1e-3))
        # test weight pushing log
        dest_log = k2.determinize(
            fsa, k2.DeterminizeWeightPushingType.kLogWeightPushing)
        self.assertTrue(
            k2.is_rand_equivalent(fsa, dest_log, log_semiring, delta=1e-3))


# TODO(fangjun): add more tests to test autograd use simple cases

if __name__ == '__main__':
    unittest.main()
