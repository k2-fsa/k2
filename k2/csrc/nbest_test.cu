/**
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "k2/csrc/nbest.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"


namespace k2 {
TEST(AlgorithmsTest, TestSuffixArray) {
  ContextPtr cpu = GetCpuContext();

  for (int i = 0; i < 100; i++) {
    int array_len = RandInt(1, 50),  // 1 is min, due to termination symbol.
        max_symbol = RandInt(10, 500);

    Array1<int32_t> array(cpu, array_len + 3);
    int32_t *array_data = array.Data();
    for (int i = 0; i + 1 < array_len; i++)
      array_data[i] = RandInt(1, max_symbol - 1);  // termination symbol must
                                                   // be larger than all
                                                   // others, don't allow
    array_data[array_len - 1] = max_symbol;  // Termination symbol

    for (int i = array_len; i < array_len + 3; i++)
      array_data[i] = 0;

    // really array_len, extra elem is to test that it doesn't write past
    // the end.
    Array1<int32_t> suffix_array(cpu, array_len + 1);
    int32_t *suffix_array_data = suffix_array.Data();
    suffix_array_data[array_len] = -10;  // should not be changed.
    CreateSuffixArray(array_data, array_len,
                      max_symbol, suffix_array_data);
    assert(suffix_array_data[array_len] == -10);  // should be unchanged.
    Array1<int32_t> seen_indexes(cpu, array_len, 0);
    int32_t *seen_indexes_data = seen_indexes.Data();
    for (int32_t i = 0; i < array_len; i++)
      seen_indexes_data[suffix_array_data[i]] = 1;

    for (int32_t i = 0; i < array_len; i++)
      assert(seen_indexes_data[i] == 1);  // make sure all integers seen.
    for (int32_t i = 0; i + 1 < array_len; i++) {
      int32_t *suffix_a = array_data + suffix_array_data[i],
              *suffix_b = array_data + suffix_array_data[i + 1];
      // checking that each suffix is lexicographically less than the next one.
      // None are identical, because the terminating zero is always in different
      // positions.
      while (true) {
        if (*suffix_a < *suffix_b)
          break;  // correct order
        assert(!(*suffix_a > *suffix_b));  // order is wrong!
        // past array end without correct comparison order.
        assert(!(suffix_a > array_data + array_len ||
                 suffix_b > array_data + array_len));
        suffix_a++;
        suffix_b++;
      }
    }
  }
}

TEST(AlgorithmsTest, TestCreateLcpArray) {
  ContextPtr cpu = GetCpuContext();

  for (int i = 0; i < 100; i++) {
    int array_len = RandInt(1, 50),  // at least 1 due to termination symbol
        max_symbol = RandInt(2, 5);

    Array1<int32_t> array(cpu, array_len + 3);
    int32_t *array_data = array.Data();
    for (int i = 0; i + 1 < array_len; i++)
      array_data[i] = RandInt(1, max_symbol - 1);
    array_data[array_len - 1] = max_symbol;  // Termination symbol
    for (int i = array_len; i < array_len + 3; i++)
      array_data[i] = 0;

    Array1<int32_t> suffix_array(cpu, array_len);
    int32_t *suffix_array_data = suffix_array.Data();
    CreateSuffixArray(array_data, array_len,
                      max_symbol, suffix_array_data);

    Array1<int32_t> lcp(cpu, array_len);
    int32_t *lcp_data = lcp.Data();
    CreateLcpArray(array_data, suffix_array_data, array_len,
                   lcp_data);
    if (array_len > 0)
      assert(lcp_data[0] == 0);
    for (int32_t i = 1; i < array_len; i++) {
      int32_t lcp = lcp_data[i],
          prev_pos = suffix_array_data[i - 1],
          this_pos = suffix_array_data[i];
      for (int32_t j = 0; j < lcp; j++)
        assert(array_data[prev_pos + j] == array_data[this_pos + j]);
      assert(array_data[prev_pos + lcp] != array_data[this_pos + lcp]);
    }
  }
}

TEST(AlgorithmsTest, TestCreateLcpIntervalArray) {
  ContextPtr cpu = GetCpuContext();

  for (int i = 0; i < 100; i++) {
    int array_len = RandInt(1, 50),   // at least 1 due to termination symbol
        max_symbol = RandInt(3, 5);

    Array1<int32_t> array(cpu, array_len + 3);
    int32_t *array_data = array.Data();
    for (int i = 0; i + 1 < array_len; i++)
      array_data[i] = RandInt(1, max_symbol - 1);
    array_data[array_len - 1] = max_symbol;  // Termination symbol
    for (int i = array_len; i < array_len + 3; i++)
      array_data[i] = 0;

    Array1<int32_t> suffix_array(cpu, array_len);
    int32_t *suffix_array_data = suffix_array.Data();
    CreateSuffixArray(array_data, array_len,
                      max_symbol, suffix_array_data);

    Array1<int32_t> lcp(cpu, array_len);
    int32_t *lcp_data = lcp.Data();
    CreateLcpArray(array_data, suffix_array_data, array_len,
                   lcp_data);

    Array1<LcpInterval<int32_t> > lcp_intervals;
    Array1<int32_t> leaf_parent_intervals;

    CreateLcpIntervalArray(GetCpuContext(),
                           array_len, lcp_data,
                           &lcp_intervals,
                           &leaf_parent_intervals);

    LcpInterval<int32_t> *lcp_intervals_data = lcp_intervals.Data();
    int32_t *leaf_parent_intervals_data = leaf_parent_intervals.Data();
    int32_t num_intervals = lcp_intervals.Dim();
    for (int32_t i = 0; i < array_len; i++) {
      int32_t lcp_interval = leaf_parent_intervals_data[i];
      assert(lcp_interval >= 0 && lcp_interval < num_intervals);
      assert(i >= lcp_intervals_data[lcp_interval].lb &&
             i <= lcp_intervals_data[lcp_interval].rb);
      // the lcp value / height
      int32_t lcp = lcp_intervals_data[lcp_interval].lcp;

      for (int32_t j = 0; j < num_intervals; j++) {
        // The interval that i is a member of should be the tightest enclosing
        // interval, this loop checks that.
        if (lcp_intervals_data[j].lcp >= lcp && j != lcp_interval) {
          assert(!(i >= lcp_intervals_data[j].lb &&
                   i <= lcp_intervals_data[j].rb));
        }
      }
    }

    for (int32_t i = 0; i < num_intervals; i++) {
      LcpInterval<int32_t> interval = lcp_intervals_data[i];
      if (!(interval.lb == 0 && interval.rb + 1 == array_len &&
            interval.parent == -1)) {
        assert(interval.parent > i);
        LcpInterval<int32_t> parent = lcp_intervals_data[interval.parent];
        assert(interval.lb >= parent.lb &&
               interval.rb <= parent.rb &&
               interval.lcp > parent.lcp);
      }
      // Now check the basic requirements/definition of lcp interval...
      assert(interval.lb >= 0 &&
             (interval.rb > interval.lb || array_len == 1) &&
             interval.rb < array_len);
      assert(lcp_data[interval.lb] < interval.lcp ||
             (interval.lb == 0 && interval.lcp == 0));
      assert(interval.rb == array_len - 1 ||
             lcp_data[interval.rb + 1] < interval.lcp);
      if (array_len != 1) {
        int32_t min_lcp = 1000000;
        for (int32_t j = interval.lb + 1; j <= interval.rb; ++j)
          if (lcp_data[j] < min_lcp)
            min_lcp = lcp_data[j];
        assert(min_lcp == interval.lcp);  // Check lcp value is correct.  This
                                          // test does not work if array_len ==
                                          // 1 so we skip it in that case.
      }
    }
  }
}

TEST(AlgorithmsTest, TestFindTightestNonemptyIntervals) {
  ContextPtr cpu = GetCpuContext();

  for (int i = 0; i < 100; i++) {
    int array_len = RandInt(1, 50),   // at least 1 due to termination symbol
        max_symbol = RandInt(3, 5);

    Array1<int32_t> array(cpu, array_len + 3),
        counts(cpu, array_len);
    int32_t *array_data = array.Data();
    for (int i = 0; i + 1 < array_len; i++)
      array_data[i] = RandInt(1, max_symbol - 1);
    array_data[array_len - 1] = max_symbol;  // Termination symbol
    for (int i = array_len; i < array_len + 3; i++)
      array_data[i] = 0;

    int32_t *counts_data = counts.Data();
    for (int i = 0; i < array_len; i++)
      counts_data[i] = RandInt(0, 1);

    Array1<int32_t> suffix_array_plusone(cpu, array_len + 1, 0),
        suffix_array = suffix_array_plusone.Range(0, array_len);
    int32_t *suffix_array_data = suffix_array.Data();
    CreateSuffixArray(array_data, array_len,
                      max_symbol, suffix_array_data);

    Array1<int32_t> lcp(cpu, array_len);
    int32_t *lcp_data = lcp.Data();
    CreateLcpArray(array_data, suffix_array_data, array_len,
                   lcp_data);

    Array1<LcpInterval<int32_t> > lcp_intervals;
    Array1<int32_t> leaf_parent_intervals;  // dim will be seq_len

    CreateLcpIntervalArray(GetCpuContext(),
                           array_len, lcp_data,
                           &lcp_intervals,
                           &leaf_parent_intervals);
    // we get one extra don't-care element at the end of `counts_reordered`,
    // which is required by ExclusiveSum().
    Array1<int32_t> counts_reordered = counts[suffix_array_plusone],
        counts_reordered_sum(cpu, array_len + 1);
    ExclusiveSum(counts_reordered, &counts_reordered_sum);


    Array1<int32_t> leaf_parent_intervals_mod(leaf_parent_intervals.Clone());

    FindTightestNonemptyIntervals(array_len,
                                  &lcp_intervals,
                                  &counts_reordered_sum,
                                  &leaf_parent_intervals_mod);

    LcpInterval<int32_t> *lcp_intervals_data = lcp_intervals.Data();
    int32_t *leaf_parent_intervals_data = leaf_parent_intervals.Data(),
        *leaf_parent_intervals_mod_data = leaf_parent_intervals_mod.Data();

    int32_t num_intervals = lcp_intervals.Dim();
    for (int32_t i = 0; i < array_len; i++) {
      int32_t lcp_interval = leaf_parent_intervals_data[i],
          nonempty_lcp_interval = leaf_parent_intervals_mod_data[i];
      assert(lcp_interval >= 0 && lcp_interval < num_intervals);
      assert(nonempty_lcp_interval >= 0 &&
             nonempty_lcp_interval < num_intervals);
      if (counts_reordered_sum[array_len] == 0) {
        // If the total count is zero, everything should go to the top of the
        // tree, but we won't otherwise test this.
        assert(nonempty_lcp_interval == num_intervals - 1);
      } else {
        int32_t lcp = lcp_intervals_data[lcp_interval].lcp;
        K2_CHECK_EQ((lcp_interval == nonempty_lcp_interval),
                    (counts_reordered_sum[lcp_intervals_data[lcp_interval].lb] !=      // NOLINT
                     counts_reordered_sum[lcp_intervals_data[lcp_interval].rb + 1]));  // NOLINT
        K2_CHECK(i >= lcp_intervals_data[nonempty_lcp_interval].lb &&
                 i <= lcp_intervals_data[nonempty_lcp_interval].rb);

        for (int32_t j = 0; j < num_intervals; j++) {
          // nonempty_lcp_interval  should be the tightest enclosing
          // interval that has nonzero count, this loop checks that.
          if (lcp_intervals_data[j].lcp >= lcp && j != nonempty_lcp_interval) {
            // Check that this is not a tighter enclosing interval than
            // nonempty_lcp_interval, with nonzero count, that encloses i.
            K2_CHECK(!(i >= lcp_intervals_data[j].lb &&
                       i <= lcp_intervals_data[j].rb &&
                       counts_reordered_sum[lcp_intervals_data[j].lb] !=
                       counts_reordered_sum[lcp_intervals_data[j].rb + 1]));
          }
        }
      }
    }
  }
}

TEST(AlgorithmTest, TestGetBestMatchingStatsEmpty) {
  Ragged<int32_t> tokens(GetCpuContext(), "[ [ [ ] ] ]");
  Array1<float> scores(GetCpuContext(), "[ ]");
  Array1<int32_t> counts(GetCpuContext(), "[ ]");
  Array1<float> mean, var;
  Array1<int32_t> counts_out, ngram_order;
  int32_t eos = 8,
          min_token = 1,
          max_token = 8,
          max_order = 2;
  GetBestMatchingStats(tokens, scores, counts, eos, min_token, max_token,
                       max_order, &mean, &var, &counts_out, &ngram_order);

  K2_CHECK_EQ(mean.Dim(), 0);
  K2_CHECK_EQ(var.Dim(), 0);
  K2_CHECK_EQ(counts_out.Dim(), 0);
  K2_CHECK_EQ(ngram_order.Dim(), 0);
}

TEST(AlgorithmTest, TestGetBestMatchingStatsSingle) {
  // There are 20 tokens, index with [0, 20)
  // keys' positions are [0, 10), queries positions are [10, 20)
  // The best matching positions(include the token itself) are as follows
  // index 0  : (0, 5, 10) with lcp "84", we add eos(8)
  // index 1  : (1, 16,) with lcp "6"
  // index 2  : (2, 17,) with lcp "76"
  // index 3  : (3, 18,) with lcp "671"
  // index 4  : (4, 19,) with lcp "5718"
  // index 5  : (5, 10,) with lcp "7184"
  // index 6  : (6, 11,) with lcp "43"
  // index 7  : (2, 7, 17,) with lcp "7"
  // index 8  : (3, 8, 18,) with lcp "71"
  // index 9  : (4, 9, 19,) with lcp "718"
  // index 10 : (5, 10,) with lcp "7184"
  // index 11 : (6, 11,) with lcp "43"
  // index 12 : (12,) with no matching
  // index 13 : (3, 8, 13, 18,) with lcp "1"
  // index 14 : (4, 9, 14, 19,) with lcp "18"
  // index 15 : (15,) with no matching
  // index 16 : (1, 16,) with lcp "6"
  // index 17 : (2, 17,) with lcp "67"
  // index 18 : (3, 18,) with lcp "671"
  // index 19 : (4, 19,) with lcp "6718"
  Ragged<int32_t> tokens(GetCpuContext(), "[ [ 4 6 7 1 8 ] [ 4 3 7 1 8 ] "
                                          "  [ 4 3 2 1 8 ] [ 5 6 7 1 8 ] ]");
  Array1<float> scores(GetCpuContext(), "[ 1 2 3 4 5 6 7 8 9 10 "
                                        "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<int32_t> counts(GetCpuContext(), "[ 1 1 1 1 1 1 1 1 1 1 "
                                          "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<float> mean, var;
  Array1<int32_t> counts_out, ngram_order;
  int32_t eos = 8,
          min_token = 1,
          max_token = 8,
          max_order = 2;
  GetBestMatchingStats(tokens, scores, counts, eos, min_token, max_token,
                       max_order, &mean, &var, &counts_out, &ngram_order);
  Array1<float> mean_ref(GetCpuContext(), "[ 3.5 2 3 4 5 6 7 5.5 6.5 7.5 "
                                          "  6 7 5.5 6.5 7.5 5.5 2 3 4 5 ]");
  Array1<float> var_ref(GetCpuContext(), "[ 6.25 0 0 0 0 0 0 6.25 6.25 6.25 "
                                      "  0 0 8.25 6.25 6.25 8.25 0 0 0 0 ]");
  Array1<int32_t> counts_out_ref(GetCpuContext(), "[ 2 1 1 1 1 1 1 2 2 2 "
                                                  "  1 1 0 2 2 0 1 1 1 1 ]");
  Array1<int32_t> ngram_order_ref(GetCpuContext(), "[ 2 1 2 2 2 2 2 1 2 2 "
                                                   "  2 2 0 1 2 0 1 2 2 2 ]");
  K2_CHECK(Equal(mean, mean_ref));
  K2_CHECK(Equal(var, var_ref));
  K2_CHECK(Equal(counts_out, counts_out_ref));
  K2_CHECK(Equal(ngram_order, ngram_order_ref));
}

TEST(AlgorithmTest, TestGetBestMatchingStatsSpecial) {
  Ragged<int32_t> tokens(GetCpuContext(), "[ [ 4 6 7 1 8 ] [ 4 3 7 1 8 ] "
                                          "  [ 4 3 2 1 8 ] [ 5 6 7 1 8 ] ]");
  Array1<float> scores(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                        "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<int32_t> counts(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                          "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<float> mean, var;
  Array1<int32_t> counts_out, ngram_order;
  int32_t eos = 8,
          min_token = 1,
          max_token = 8,
          max_order = 2;
  GetBestMatchingStats(tokens, scores, counts, eos, min_token, max_token,
                       max_order, &mean, &var, &counts_out, &ngram_order);
  Array1<float> mean_ref(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                          "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<float> var_ref(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                      "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<int32_t> counts_out_ref(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                                  "  0 0 0 0 0 0 0 0 0 0 ]");
  Array1<int32_t> ngram_order_ref(GetCpuContext(), "[ 0 0 0 0 0 0 0 0 0 0 "
                                                   "  0 0 0 0 0 0 0 0 0 0 ]");
  K2_CHECK(Equal(mean, mean_ref));
  K2_CHECK(Equal(var, var_ref));
  K2_CHECK(Equal(counts_out, counts_out_ref));
  K2_CHECK(Equal(ngram_order, ngram_order_ref));
}

TEST(AlgorithmTest, TestGetBestMatchingStatsSingleMulti) {
  Ragged<int32_t> tokens(GetCpuContext(), "[ [ [ 4 6 7 1 8 ] [ 4 3 7 1 8 ] "
                                          "    [ 4 3 2 1 8 ] [ 5 6 7 1 8 ] ] "
                                          "  [ [ 5 1 4 8 ] [ 5 1 2 8 ] "
                                          "    [ 5 3 4 8 ] ] ]");
  Array1<float> scores(GetCpuContext(), "[ 1 2 3 4 5 6 7 8 9 10 "
                                        "  0 0 0 0 0 0 0 0 0 0 "
                                        "  1 2 3 4 5 7 8 6 0 0 0 0 ]");
  Array1<int32_t> counts(GetCpuContext(), "[ 1 1 1 1 1 1 1 1 1 1 "
                                          "  0 0 0 0 0 0 0 0 0 0 "
                                          "  1 1 1 1 1 1 1 1 0 0 0 0 ]");
  Array1<float> mean, var;
  Array1<int32_t> counts_out, ngram_order;
  int32_t eos = 8,
          min_token = 0,
          max_token = 10,
          max_order = 5;
  GetBestMatchingStats(tokens, scores, counts, eos, min_token, max_token,
                       max_order, &mean, &var, &counts_out, &ngram_order);
  Array1<float> mean_ref(GetCpuContext(), "[ 3.5 2 3 4 5 6 7 5.5 6.5 7.5 "
                                          "  6 7 5.5 6.5 7.5 5.5 2 3 4 5 "
                                          "  3 4.5 3 4 3 4.5 4.5 5 "
                                          "  3 4.5 3 4 ]");
  Array1<float> var_ref(GetCpuContext(), "[ 6.25 0 0 0 0 0 0 6.25 6.25 6.25 "
                                      "  0 0 8.25 6.25 6.25 8.25 0 0 0 0 "
                                      "  4 6.25 0 0 4 6.25 5.25 1 "
                                      "  4 5.25 0 0 ]");
  Array1<int32_t> counts_out_ref(GetCpuContext(), "[ 2 1 1 1 1 1 1 2 2 2 "
                                                  "  1 1 0 2 2 0 1 1 1 1 "
                                                  "  2 2 1 1 2 2 0 2 "
                                                  "  2 0 1 1 ]");
  Array1<int32_t> ngram_order_ref(GetCpuContext(), "[ 5 1 2 3 4 5 5 1 2 3 "
                                                   "  5 5 0 1 2 0 1 2 3 4 "
                                                   "  5 5 1 2 5 5 0 1 "
                                                   "  5 0 1 2 ]");
  K2_CHECK(Equal(mean, mean_ref));
  K2_CHECK(Equal(var, var_ref));
  K2_CHECK(Equal(counts_out, counts_out_ref));
  K2_CHECK(Equal(ngram_order, ngram_order_ref));
}

}  // namespace k2
