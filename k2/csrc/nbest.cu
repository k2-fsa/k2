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

#include <algorithm>
#include <vector>

#include "k2/csrc/nbest.h"

// This is not really a CUDA file but for build-system reasons I'm currently
// leaving it with the .cu extension.

namespace k2 {

template <typename T>
inline bool Leq(T a1, T a2, T b1, T b2) {
  // lexicographic order for pairs, used in CreateSuffixArray()
  return(a1 < b1 || a1 == b1 && a2 <= b2);
}
template <typename T>
inline bool Leq(T a1, T a2, T a3, T b1, T b2, T b3) {
  // lexicographic order for triples, used in CreateSuffixArray()
  return(a1 < b1 || a1 == b1 && Leq(a2, a3, b2, b3));
}

/*
  Helper function for CreateSuffixArray().

  Stably sorts a[0..n-1] to b[0..n-1] with keys in 0..K from r;
  i.e. the values in a are interpreted as indexes into the array
  `r` and the values in `r` are used for comparison, so that
  at exit, r[b[i]] <= r[b[i+1]].
*/
template <typename T>
static void RadixPass(const T* a, T* b, const T* r, T n, T K) {
  std::vector<T> c(K + 1, 0);  // counter array
  for (T i = 0; i < n; i++) c[r[a[i]]]++;  // count occurrences
  for (T i = 0, sum = 0; i <= K; i++) {  // exclusive prefix sums
    T t = c[i]; c[i] = sum; sum += t;
  }
  for (T i = 0; i < n; i++) b[c[r[a[i]]]++] = a[i];  // sort
}

// See documentation in nbest.h, where we use different names
// for the arguments (here, we leave the names the same as in
// https://algo2.iti.kit.edu/documents/jacm05-revised.pdf.
template <typename T>
void CreateSuffixArray(const T* text, T n, T K, T* SA) {
  // assert(text[0] <= text[n-1]);    // spot check that termination symbol is
                                   // larger than other symbols;
                                   // <= in case n==1.
  if (n == 1) {  // The paper's code didn't seem to handle n == 1 correctly.
    SA[0] = 0;
    return;
  }
  T n0 = (n + 2) / 3, n1 = (n+1) / 3, n2 = n / 3, n02 = n0 + n2;
  std::vector<T> R(n02 + 3, 0);
  std::vector<T> SA12(n02 + 3, 0);
  std::vector<T> R0(n0, 0);
  std::vector<T> SA0(n0, 0);

  //******* Step 0: Construct sample ********
  // generate positions of mod 1 and mod 2 suffixes
  // the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
  for (T i = 0, j = 0; i < n + (n0 - n1); i++) if (i % 3 != 0) R[j++] = i;
  //******* Step 1: Sort sample suffixes ********
  // lsb radix sort the mod 1 and mod 2 triples
  RadixPass(R.data(), SA12.data(), text + 2, n02, K);
  RadixPass(SA12.data(), R.data() , text + 1, n02, K);
  RadixPass(R.data(), SA12.data(), text, n02, K);

  // find lexicographic names of triples and
  // write them to correct places in R
  T name = 0, c0 = -1, c1 = -1, c2 = -1;
  for (T i = 0; i < n02; i++) {
    if (text[SA12[i]] != c0 || text[SA12[i] + 1] != c1 ||
        text[SA12[i] + 2] != c2) {
      name++;
      c0 = text[SA12[i]];
      c1 = text[SA12[i] + 1];
      c2 = text[SA12[i] + 2];
    }
    if (SA12[i] % 3 == 1) { R[SA12[i] / 3] = name; }  // write to R1
    else { R[SA12[i] / 3 + n0] = name; }  // write to R2
  }
  // recurse if names are not yet unique
  if (name < n02) {
    CreateSuffixArray(R.data(), n02, name, SA12.data());
    // store unique names in R using the suffix array
    for (T i = 0; i < n02; i++) R[SA12[i]] = i + 1;
  } else  // generate the suffix array of R directly
    for (T i = 0; i < n02; i++) SA12[R[i] - 1] = i;
  //******* Step 2: Sort nonsample suffixes ********
  // stably sort the mod 0 suffixes from SA12 by their first character
  for (T i = 0, j = 0; i < n02; i++)
    if (SA12[i] < n0) R0[j++] = 3 * SA12[i];
  RadixPass(R0.data(), SA0.data(), text, n0, K);
  //******* Step 3: Merge ********
  // merge sorted SA0 suffixes and sorted SA12 suffixes
  for (T p = 0, t = n0 - n1, k = 0; k < n; k++) {
    // i is pos of current offset 12 suffix
    T i = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2);
    T j = SA0[p];  // pos of current offset 0 suffix
    if (SA12[t] < n0 ?  // different compares for mod 1 and mod 2 suffixes
        Leq(text[i], R[SA12[t] + n0], text[j], R[j / 3]) :
        Leq(text[i], text[i + 1], R[SA12[t] - n0 + 1], text[j],
            text[j + 1], R[j / 3 + n0])) {  // suffix from SA12 is smaller
      SA[k] = i; t++;
      if (t == n02)  // done --- only SA0 suffixes left
        for (k++; p < n0; p++, k++) SA[k] = SA0[p];
    } else {  // suffix from SA0 is smaller
      SA[k] = j; p++;
      if (p == n0)  // done --- only SA12 suffixes left
        for (k++; t < n02; t++, k++)
          SA[k] = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2);
    }
  }
}

// Instantiate template for int32_t and int16_t
template void CreateSuffixArray(const int32_t* text, int32_t n,
                                int32_t K, int32_t* SA);
template void CreateSuffixArray(const int16_t* text, int16_t n,
                                int16_t K, int16_t* SA);

// This implements Kasai's algorithm, as summarized here
// https://people.csail.mit.edu/jshun/lcp.pdf (Fig. 3)
// (Note: there seem to be some wrong implementations of
// Kasai's algorithm online).
/*
template <typename T>
void CreateLcpArray(const T *array,
                    const T *suffix_array,
                    T seq_len,
                    T *lcp_array) {
  Array1<T> inv_suffix_array(GetCpuContext(), seq_len);
  T *inv_suffix_array_data = inv_suffix_array.Data();
  for (T i = 0; i < seq_len; i++) {
    inv_suffix_array_data[suffix_array[i]] = i;
  }
  T k = 0;
  if (seq_len > 0)
    lcp_array[0] = 0;
  for (T i = 0; i < seq_len; ++i) {
    T cur_rank = inv_suffix_array[i];
    if (cur_rank != 0) {
      T j = suffix_array[cur_rank - 1];
      while (array[i + k] == array[j + k])
        ++k;
      lcp_array[cur_rank] = k;
      if (k > 0)
        --k;
    }
  }
}
*/

// This implements a modification of Kasai's algorithm by Karkkainen et al.,
// as summarized in https://people.csail.mit.edu/jshun/lcp.pdf (Fig. 4).
// The result is identical to Kasai's but is shown to be faster in practice.
// (Note: pseudocodes in Fig 3. and Fig 4. generate slightly different results.
// Here we slightly modified Fig 4. algorithm to get identical results as
// Fig 3 (Kasai's algorithm).)
template <typename T>
void CreateLcpArray(const T *array,
                    const T *suffix_array,
                    T seq_len,
                    T *lcp_array) {
  Array1<T> plcp(GetCpuContext(), seq_len);
  T *plcp_data = plcp.Data();

  Array1<T> phi(GetCpuContext(), seq_len);  // the Phi array
  T *phi_data = phi.Data();
  phi_data[suffix_array[0]] = -1;
  for (T i = 1; i < seq_len; i++)
    phi_data[suffix_array[i]] = suffix_array[i-1];

  T k = 0;
  for (T i = 0; i < seq_len; i++) {
    if (phi_data[i] == -1)
      k = 0;
    else {
      while (array[i + k] == array[phi_data[i] + k])
        k++;
    }
    plcp_data[i] = k;
    if (k > 0)
      k--;
  }

  for (T i = 0; i < seq_len; i++)
    lcp_array[i] = plcp_data[suffix_array[i]];
}

// Instantiate template for int32_t and int16_t
template void CreateLcpArray(const int32_t *array, const int32_t *suffix_array,
                             int32_t seq_len, int32_t *lcp_array);
template void CreateLcpArray(const int16_t *array, const int16_t *suffix_array,
                             int16_t seq_len, int16_t *lcp_array);

template <typename T>
void CreateLcpIntervalArray(ContextPtr c,
                            T seq_len,
                            T *lcp_array,
                            Array1<LcpInterval<T> > *lcp_intervals,
                            Array1<T> *leaf_parent_intervals) {
  *lcp_intervals = Array1<LcpInterval<T> >(c, seq_len);
  LcpInterval<T> *lcp_intervals_data = lcp_intervals->Data();

  Array1<T> intervals_order(c, seq_len);
  T *intervals_order_data = intervals_order.Data();

  Array1<T> leaf_parent(c, seq_len);
  T *leaf_parent_data = leaf_parent.Data();

  // This is the stack from Algorithm 1 and Algorithm 2 of
  // http://www.mi.fu-berlin.de/wiki/pub/ABI/RnaSeqP4/enhanced-suffix-array.pdf
  // (you can refer to the papers mentioned in the documentation in nbest.h
  //  if this link goes dead).
  //
  // The 'begin', 'last' and 'lcp' members correspond to the 'lb', 'rb' and
  // 'lcp' members mentioned there; the 'parent' member is used temporarily
  // on the stack to refer to the index of this LcpInterval in
  // `lcp_intervals_data`, i.e. it can be interpreted as a 'self' pointer.
  std::vector<LcpInterval<T> > stack;

  // A separate stack, of leaves of suffix tree; we maintain this so that
  // we can assign the `leaf_parent` array.
  std::vector<T> leaf_stack;

  // lcp=0; begin=0; last=undefined; self=0  (interpreting the 'parent' member
  // as index-of-self
  // Will always store the next free index into `lcp_intervals_data`
  T next = 0;
  // Will always store the next free index into `intervals_order_data`;
  // this is an ordering of the indexes into `lcp_intervals_data` that
  // corresponds to depth-first search.
  T dfs_next = 0;
  T last_interval = -1;  // Will store an index into `lcp_intervals`; this
                         // comes from Algorithm 2 mentioned above
  stack.push_back({0, 0, T(seq_len - 1), next++ });
  // We are using a numbering in which the terminating symbol $ is included
  // in the array length, which is why we do "i < seq_len" and not
  // "i <= seq_len" as in
  // http://www.mi.fu-berlin.de/wiki/pub/ABI/RnaSeqP4/enhanced-suffix-array.pdf
  for (T i = 1; i < seq_len; ++i) {
    T lb = i - 1, lcp_array_i = lcp_array[i];
    leaf_stack.push_back(lb);

    while (lcp_array_i < stack.back().lcp) {
      last_interval = stack.back().parent;  // actually, the .parent field
                                            // currently represents 'self',
                                            // i.e. the index of the
                                            // lcp-interval stack.back().
      T last_interval_dfsorder = dfs_next++;
      lb = stack.back().lb;
      while (!leaf_stack.empty() && leaf_stack.back() >= lb) {
        leaf_parent_data[leaf_stack.back()] = last_interval_dfsorder;
        leaf_stack.pop_back();
      }
      // process(last_interval):
      lcp_intervals_data[last_interval_dfsorder] = stack.back();
      // Previously tried doing:
      //   stack.back().rb = i - 1;
      // a bit further above, but hit some kind of compiler problem,
      // the assignment had no effect (back() is supposed to return a
      // reference).
      lcp_intervals_data[last_interval_dfsorder].rb = i - 1;
      intervals_order_data[last_interval] = last_interval_dfsorder;
      stack.pop_back();
      if (lcp_array_i <= stack.back().lcp) {
        // lcp_intervals_data[last_interval_dfsorder].parent represents
        // the parent of `last_interval`; `stack.back().parent` currently
        // represents the intended position of stack.back() itself,
        // not of its parent.
        lcp_intervals_data[last_interval_dfsorder].parent =
            stack.back().parent;
        last_interval = -1;
      }
    }
    if (lcp_array_i > stack.back().lcp) {
      if (last_interval >= 0) {
        lcp_intervals_data[intervals_order_data[last_interval]].parent = next;
        last_interval = -1;
      }
      stack.push_back({lcp_array_i, lb, -1, next++});
    }
  }
  assert(stack.size() == 1);
  T top_node_dfsorder = dfs_next++;
  lcp_intervals_data[top_node_dfsorder] = stack.back();
  lcp_intervals_data[top_node_dfsorder].parent = -1;
  intervals_order_data[0] = top_node_dfsorder;
  leaf_stack.push_back(seq_len - 1);
  while (!leaf_stack.empty()) {
    leaf_parent_data[leaf_stack.back()] = top_node_dfsorder;
    leaf_stack.pop_back();
  }
  assert(dfs_next == next);
  for (T i = 0; i + 1 < next; i++) {
    // for each lpc-interval, except the last (top) node which has -1 as its
    // parent field..  Change from pushing order (order in which we pushed them
    // onto the stack) to dfs post order (order in which they were popped).
    lcp_intervals_data[i].parent =
      intervals_order_data[lcp_intervals_data[i].parent];
  }

  *lcp_intervals = lcp_intervals->Range(0, next);
  for (T i = 0; i < next; i++)
    intervals_order_data[i] = i;  // We output in dfs post order now.. will
                                  // remove this output arg.
  if (leaf_parent_intervals)
    *leaf_parent_intervals = leaf_parent;
}

// Instantiate template
template
void CreateLcpIntervalArray(ContextPtr c,
                            int32_t seq_len,
                            int32_t *lcp_array,
                            Array1<LcpInterval<int32_t> > *lcp_intervals,
                            Array1<int32_t> *leaf_parent_intervals);
template
void CreateLcpIntervalArray(ContextPtr c,
                            int16_t seq_len,
                            int16_t *lcp_array,
                            Array1<LcpInterval<int16_t> > *lcp_intervals,
                            Array1<int16_t> *leaf_parent_intervals);

template <typename T>
void FindTightestNonemptyIntervals(T seq_len,
                                   Array1<LcpInterval<T> > *lcp_intervals,
                                   Array1<T> *counts_exclusive_sum,
                                   Array1<T> *leaf_parent_intervals) {
  ContextPtr c = lcp_intervals->Context();
  K2_CHECK(counts_exclusive_sum->Dim() == seq_len + 1);
  K2_CHECK(leaf_parent_intervals->Dim() == seq_len);

  const LcpInterval<T> *lcp_intervals_data = lcp_intervals->Data();
  const T *counts_exclusive_sum_data = counts_exclusive_sum->Data();
  int32_t num_intervals = lcp_intervals->Dim();
  // `tightest_nonempty_intervals` gives, for each interval
  // 0 <= i < num_intervals, the index j >= i of the tightest enclosing
  // interval that has a nonzero count.  As a special case, if all counts
  // are zero, it will return the top (last) interval.
  Array1<T> tightest_nonempty_interval(c, num_intervals);
  T *tightest_nonempty_interval_data = tightest_nonempty_interval.Data();
  for (T i = num_intervals - 1; i >= 0; --i) {
    T j;
    LcpInterval<T> cur_interval = lcp_intervals_data[i];
    if (cur_interval.parent < 0 ||  // top node
        counts_exclusive_sum_data[cur_interval.rb + 1] >
        counts_exclusive_sum_data[cur_interval.lb]) {
      j = i;
    } else {
      // j > i, we will have already set tightest_nonempty_interval_data
      // at this location.
      j = tightest_nonempty_interval_data[cur_interval.parent];
    }
    tightest_nonempty_interval_data[i] = j;
  }
  T *leaf_parent_intervals_data = leaf_parent_intervals->Data();
  for (T i = 0; i < seq_len; ++i)
    leaf_parent_intervals_data[i] = tightest_nonempty_interval_data[
        leaf_parent_intervals_data[i]];
}

// Instantiate template
template
void FindTightestNonemptyIntervals(int32_t seq_len,
                                   Array1<LcpInterval<int32_t> > *lcp_intervals,  // NOLINT
                                   Array1<int32_t> *counts_exclusive_sum,
                                   Array1<int32_t> *leaf_parent_intervals);
template
void FindTightestNonemptyIntervals(int16_t seq_len,
                                   Array1<LcpInterval<int16_t> > *lcp_intervals,  // NOLINT
                                   Array1<int16_t> *counts_exclusive_sum,
                                   Array1<int16_t> *leaf_parent_intervals);

// Internal implementation of GetBestMatchingStats(), that handles the case
// where tokens.NumAxes() == 2 and tokens.NumElements() > 0.  It will
// be instantiated with int16_t if the size of the problem permits, and
// int32_t otherwise (this size is used for
template <typename T>
void GetBestMatchingStatsInternal(Ragged<int32_t> &tokens,
                                  Array1<float> &scores,
                                  Array1<int32_t> &counts,
                                  T eos,
                                  T min_token,
                                  T max_token,
                                  int32_t max_order,
                                  Array1<float> *mean,
                                  Array1<float> *var,
                                  Array1<int32_t> *counts_out,
                                  Array1<int32_t> *ngram_order) {
  // Outline:
  //  First construct an array of type T which contains values as follows:
  //    [ tokens.values[-1]+offset, ..., tokens.values[1]+offset,
  //      tokens.values[0]+offset, eos+offset, terminator, 0, 0, 0 ]
  //  where offset is 1-min_token, and terminator is max_token+1+offset.
  //  The 3 terminating zeros are required by CreateSuffixArray().
  //
  //  Call CreateSuffixArray (seq_len == tokens.Dim() + 2, we include the
  //  eos and terminator).
  //
  //  Create the reordered counts array `counts_reordered`, in the same order
  //  as the suffix array, then its exclusive sum,
  //  e.g. `counts_reordered_excsum`. At this point we can also create similar
  //  reordered exclusive-sums of `scores` and scores-squared;
  //  do these as double or roundoff will be a problem.
  //
  //  Call CreateLcpArray, CreateLcpIntervalArray,
  //  FindTightestNonemptyIntervals
  //
  //  By this point we should have enough information to directly create the
  //  outputs : mean, var, counts_out, ngram_order.  We need to be a bit
  //  careful about ngram_order at positions when the suffix goes up to the
  //  next eos (i.e. it goes to the beginning of the sentence) because the
  //  correct ngram order to output here is `max_order`.  You will have to
  //  create an array containing the distance from the beginning of the
  //  sentence (can be constructed from the row_ids and row_splits of `tokens`)
  //
  //  Note: we only really care about the output at the query positions, but
  //  try to make it so you don't need to treat keys as a special case.
  //
  //  Special cases/conditions to consider include:
  //    - make sure the `count` in the position of the eos and terminator
  //      are zero
  //    - various code may break if the total count over all these sentences is
  //      zero, so you could just detect that and treat it as a special case.
  //      If the total count is nonzero, it should be guaranteed that you never
  //      have to process an interval with zero count;
  //      FindTightestNonemptyIntervals() should guarantee that.
  ContextPtr &c = tokens.Context();
  T num_elements = tokens.NumElements();
  K2_CHECK_EQ(mean->Dim(), num_elements);
  K2_CHECK_EQ(var->Dim(), num_elements);
  K2_CHECK_EQ(counts_out->Dim(), num_elements);
  K2_CHECK_EQ(ngram_order->Dim(), num_elements);

  T offset = 1 - min_token,
    terminator = max_token + 1 + offset;
  // we include the eos and terminator, so plus 2 here.
  T  seq_len = num_elements + 2;
  // 3 zeros are required by CreateSuffixArray3
  Array1<T> text_array(c, seq_len + 3);
  T *text_array_data = text_array.Data();
  const int32_t *tokens_values_data = tokens.values.Data();
  // we want to match the longest common prefix of the word and the words
  // preceding it, so we need to reverse the sequence before constructing
  // suffix array.
  for (T i = 0; i < num_elements; ++i) {
    text_array_data[i] =
      tokens_values_data[num_elements - i - 1] + offset;
  }
  T eos_offset = eos + offset;
  // fill eos, terminator and required zeros
  std::vector<T> tail({eos_offset, terminator, 0, 0, 0});
  for (T i = num_elements; i < text_array.Dim(); ++i)
    text_array_data[i] = tail[i - num_elements];

  Array1<T> suffix_array(c, seq_len);
  CreateSuffixArray(text_array.Data(), seq_len, terminator,
                    suffix_array.Data());

  // we need extra one position for `ExclusiveSum`
  Array1<T> reorder_counts(c, seq_len + 1);
  Array1<float> reorder_scores(c, seq_len + 1);
  Array1<double> reorder_scores_squre(c, seq_len + 1);
  T *reorder_counts_data = reorder_counts.Data();
  float *reorder_scores_data = reorder_scores.Data();
  double *reorder_scores_squre_data = reorder_scores_squre.Data();

  const int32_t *counts_data = counts.Data();
  const float *scores_data = scores.Data();
  const T *suffix_array_data = suffix_array.Data();
  for (int32_t i = 0; i < suffix_array.Dim(); ++i) {
    // we reverse the sequence above, the order of counts and scores should be
    // reversed accordingly, and make sure that the counts and scores be zero
    // in the positions of eos and terminator.
    int32_t rindex = seq_len - 2 - suffix_array_data[i] - 1;
    reorder_counts_data[i] = rindex < 0 ? 0 : counts_data[rindex];
    reorder_scores_data[i] = rindex < 0 ? 0 : scores_data[rindex];
    reorder_scores_squre_data[i] = rindex < 0 ? 0 :
                                   scores_data[rindex] * scores_data[rindex];
  }
  ExclusiveSum(reorder_counts, &reorder_counts);
  ExclusiveSum(reorder_scores, &reorder_scores);
  ExclusiveSum(reorder_scores_squre, &reorder_scores_squre);

  // total count of all these sentences is zero means that there is no **keys**
  // we can not match anything, return as special case.
  if (reorder_counts_data[reorder_counts.Dim() - 1] == 0) {
    *mean = 0;
    *var = 0;
    *counts_out = 0;
    *ngram_order = 0;
    return;
  }

  Array1<T> lpc_array(c, seq_len);
  CreateLcpArray(text_array.Data(), suffix_array.Data(), seq_len,
                 lpc_array.Data());

  Array1<T> leaf_parent_interval;
  Array1<LcpInterval<T> > lcp_intervals;
  CreateLcpIntervalArray(c, seq_len, lpc_array.Data(),
                         &lcp_intervals, &leaf_parent_interval);

  FindTightestNonemptyIntervals(seq_len, &lcp_intervals,
                                &reorder_counts, &leaf_parent_interval);
  const LcpInterval<T> *lcp_intervals_data = lcp_intervals.Data();
  const T *leaf_parent_interval_data = leaf_parent_interval.Data();

  Array1<T> dist_to_begin(c, num_elements);
  T *dist_to_begin_data = dist_to_begin.Data();
  const int32_t *tokens_row_ids1_data = tokens.RowIds(1).Data(),
                *tokens_row_splits1_data = tokens.RowSplits(1).Data();
  K2_EVAL(
    c, num_elements, lambda_set_dist_to_begin, (int32_t idx01)
    -> void {
      int32_t idx0 = tokens_row_ids1_data[idx01],
              idx0x = tokens_row_splits1_data[idx0],
              idx1 = idx01 - idx0x;
      dist_to_begin_data[idx01] = idx1 + 1;
    });

  // mapping original order to suffix array order
  Array1<T> inv_suffix_array(c, seq_len);
  T *inv_suffix_array_data = inv_suffix_array.Data();
  for (T i = 0; i < seq_len; i++) {
    inv_suffix_array_data[suffix_array_data[i]] = i;
  }
  float *mean_data = mean->Data(),
        *var_data = var->Data();
  int32_t *counts_out_data = counts_out->Data(),
          *ngram_order_data = ngram_order->Data();

  // loop in the original order
  for (T i = 0; i < num_elements; ++i) {
    // we reverse `tokens.values` above, minus 2 here to remove eos and
    // terminator that not belongs to tokens.
    T text_array_index = seq_len - 2 - i - 1;
    T suffix_index = inv_suffix_array_data[text_array_index];
    // leaf_parent_interval, reorder_counts, reorder_scores are all index with
    // suffix array order.
    T interval_index = leaf_parent_interval_data[suffix_index];
    LcpInterval<T> interval = lcp_intervals_data[interval_index];
    float scores_sum = reorder_scores_data[interval.rb + 1] -
                       reorder_scores_data[interval.lb];
    double scores_squre_sum = reorder_scores_squre_data[interval.rb + 1] -
                              reorder_scores_squre_data[interval.lb];
    int32_t counts_out_interval = reorder_counts_data[interval.rb + 1] -
                                  reorder_counts_data[interval.lb];
    if (interval.lcp == 0) {  // tightest interval is root interval
      K2_CHECK_EQ(interval.parent, -1);
      counts_out_data[i] = 0;
      ngram_order_data[i] = 0;
    } else {
      counts_out_data[i] = counts_out_interval;
      ngram_order_data[i] = min(interval.lcp, (T)max_order);
      // handle the sentence boundary
      if (dist_to_begin_data[i] <= interval.lcp)
        ngram_order_data[i] = max_order;
    }
    mean_data[i] = counts_out_interval == 0 ? 0 :
                   (scores_sum / counts_out_interval);
    if (counts_out_interval == 0 || counts_out_interval == 1) {
      var_data[i] = 0;
    } else {
      double numerator = scores_squre_sum - 2 * mean_data[i] * scores_sum +
                         counts_out_interval * mean_data[i] * mean_data[i];
      var_data[i] = numerator / counts_out_interval;
    }
  }
}

void GetBestMatchingStats(Ragged<int32_t> &tokens,
                          Array1<float> &scores,
                          Array1<int32_t> &counts,
                          int32_t eos,
                          int32_t min_token,
                          int32_t max_token,
                          int32_t max_order,
                          Array1<float> *mean,
                          Array1<float> *var,
                          Array1<int32_t> *counts_out,
                          Array1<int32_t> *ngram_order) {
  ContextPtr &c = tokens.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);

  int32_t num_elements = tokens.NumElements();
  K2_CHECK(mean);
  if (mean->Dim() != num_elements) {
    *mean = Array1<float>(c, num_elements);
  } else {
    K2_CHECK_EQ(mean->Dim(), num_elements);
  }
  K2_CHECK(var);
  if (var->Dim() != num_elements) {
    *var = Array1<float>(c, num_elements);
  } else {
    K2_CHECK_EQ(var->Dim(), num_elements);
  }
  K2_CHECK(counts_out);
  if (counts_out->Dim() != num_elements) {
    *counts_out = Array1<int32_t>(c, num_elements);
  } else {
    K2_CHECK_EQ(counts_out->Dim(), num_elements);
  }
  K2_CHECK(ngram_order);
  if (ngram_order->Dim() != num_elements) {
    *ngram_order = Array1<int32_t>(c, num_elements);
  } else {
    K2_CHECK_EQ(ngram_order->Dim(), num_elements);
  }

  K2_CHECK(eos >= min_token && eos <= max_token);
  K2_CHECK_GE(max_order, 2);
  K2_CHECK_EQ(num_elements, scores.Dim());
  K2_CHECK_EQ(num_elements, counts.Dim());

  if (tokens.NumAxes() == 3) {
    int32_t num_collections = tokens.Dim0();
    for (int32_t i = 0; i < num_collections; i++) {
      Ragged<int32_t> this_tokens = tokens.Index(0, i);
      int32_t begin = this_tokens.values.Data() - tokens.values.Data(),
          end = begin + this_tokens.NumElements();
      Array1<float> this_scores = scores.Arange(begin, end),
          this_mean = mean->Arange(begin, end),
          this_var = var->Arange(begin, end);
      Array1<int32_t> this_counts = counts.Arange(begin, end),
          this_counts_out = counts_out->Arange(begin, end),
          this_ngram_order = ngram_order->Arange(begin, end);
      GetBestMatchingStats(this_tokens, this_scores, this_counts, eos,
                           min_token, max_token, max_order,
                           &this_mean, &this_var,
                           &this_counts_out, &this_ngram_order);
    }
    return;
  }
  K2_CHECK_EQ(tokens.NumAxes(), 2);  // Only 2 or 3 axes are allowed.

  if (num_elements == 0) {
    return;  // Nothing to do.
  } else if (num_elements + 10 < (1 << 15) &&
             (max_token - min_token + 10 < (1 << 15))) {
    GetBestMatchingStatsInternal<int16_t>(tokens, scores, counts, eos,
                                          min_token, max_token, max_order,
                                          mean, var, counts_out, ngram_order);
  } else {
    GetBestMatchingStatsInternal<int32_t>(tokens, scores, counts, eos,
                                          min_token, max_token, max_order,
                                          mean, var, counts_out, ngram_order);
  }
}

}  // namespace k2
