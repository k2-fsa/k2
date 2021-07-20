/**
 * Copyright      2021     Xiaomi Corporation (authors: Daniel Povey)
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

#ifndef K2_CSRC_NBEST_H_
#define K2_CSRC_NBEST_H_

#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

namespace k2 {

// This header contains certain utility functions that are used in rescoring
// n-best lists: specifically, functions that help us select which among a set
// of n-best candidates to select for rescoring.  The selection scheme is a
// little complex. It is intended to be used in a context where we do multiple
// successive rounds of n-best list rescoring, and we use the results of the
// 1st round to guide selection of candidates in the second round.
// So for each word in each n-best path that we are considering, we find the
// best-matching positions among those that we evaluated in the first round and
// we use those as inputs to a model that predicts the scores of words after
// n-best-list rescoring.
//
// Some of these functions may seem a little unrelated to n-best lists, they
// are algorithms involving suffix arrays, which we use internally in some
// algorithms we use to process n-best lists.

/*
  This function creates a suffix array; it is based on the
  code in
  https://algo2.iti.kit.edu/documents/jacm05-revised.pdf,
  "Linear Work Suffix Array construction" by J. Karkkainen.

   Template args: T should be a signed integer type, we
   plan to instantiate this for int32_t and int16_t only.

    @param [in] text_array  Pointer to the input array of symbols,
           including the termination symbol ($) which must be larger
           than the other symbols.
           All pointers must be CPU pointers only, for now.
           The suffixes of this array are to be sorted.  Logically this
           array has length `seq_len`, and symbols are required
           to be in the range [1..max_symbol].
           text_array is additionally required to be terminated by 3 zeros,
           for purposes of this algorithm, i.e.
             text_array[seq_len] == text_array[seq_len+1] == text_array[seq_len+2] == 0
    @param [in] seq_len  Length of the symbol sequence (`text_array`
            must be longer than this by at least 3, for termination.)
            Require seq_len >= 0
    @param [out] suffix_array   A pre-allocated array of length
             `seq_len`.  At exit it will contain a permutation of
             the list [ 0, 1, ... seq_len  - 1 ], interpreted
             as the start indexes of the nonempty suffixes of `text_array`,
             with the property that the sub-arrays of `text_array`
             starting at these positions are lexicographically sorted.
             For example, as a trivial case, if seq_len = 3
             and text_array contains [ 3, 2, 1, 10, 0, 0, 0 ], then
             `suffix_array` would contain [ 2, 1, 0, 3 ] at exit.
    @param [in] max_symbol  A number that must be >= the largest
             number that might be in `text_array`, including the
             termination symbol.  The work done
             is O(seq_len + max_symbol), so it is not advisable
             to let max_symbol be too large.
    Caution: this function allocates memory internally (although
    not much more than `text_array` itself).
 */
template <typename T>
void CreateSuffixArray(const T *text_array,
                       T seq_len,
                       T max_symbol,
                       T *suffix_array);

/*
  Create the LCP array, which is the array of lengths of longest common prefixes
  (of successive pairs of suffixes).

   Template args: T should be a signed integer type, we plan to instantiate this
   for int32_t and int16_t only.

     @param [in] text_array  The array of symbols, of length `seq_len` plus at least
                    one terminating zero.  The symbols should be positive
                    (this may not be required here, but it is rqeuired by
                    CreateSuffixArray()).
     @param [in] suffix_array  The suffix array, as created by CreateSuffixArray();
                    it is a permutation of the numbers 0, 1, ... seq_len - 1.
     @param [out] lcp_array   An array of length `seq_len` is output to here;
                    it is expected to be pre-allocated.  At exit, lcp_array[0]
                    will be 0 and lcp_array[i] for i>0 will equal the length
                    of the longest common prefix of
                    (text_array+suffix_array[i-1]) and (text_array+suffix_array[i]).
*/
template <typename T>
void CreateLcpArray(const T *text_array,
                    const T *suffix_array,
                    T seq_len,
                    T *lcp_array);

/*
   Template args: T is a signed type, intended to be int16_t or int32_t

  lcp-intervals correspond to the nodes in the suffix trie; they are a concept
  used with suffix arrays, and are derived from the LCP table (see lcp_array
  output of CreateLcpArray).  Take care with the notation here: intervals are
  "closed intervals" so [i,j] means i,i+1,...,j, i.e. the RHS is the index of
  the last element, not one past the last element.

  Notation: [i,j] is an lcp-interval with lcp-value l, if:
     0 <= i < j < seq_len
     lcptab[i] < l
     lcptab[j+1] < l
     l is the minimum of (lcptab[i+1], lcptab[i+2], ..., lcptab[j])
  lcp-intervals correspond to the internal nodes of the suffix trie, so
  they always contain at least two children (where children can be
  leaves, corresponding indexes into the suffix array, or other
  lcp-intervals).

  SPECIAL CASE: if seq_len is 1, which is a rather degenerate case, the above
  definitions do not quite work; and we treat [0,0] as an lcp-interval with
  lcp-value 0 although it does not meet the above definitions.

  Type LcpInterval is used to store information about the lcp interval,
  which we'll later use in algorithms that traverse the suffix tree.
 */
template <typename T>
struct LcpInterval {
  // Represents the lcp-interval [begin,last] with lcp-value `lcp`
  T lcp;    // The lcp-value of the lcp-interval, which is the length of the
            // longest prefix shared by all suffixes in this interval.
  T lb;     // Index of the first element (left boundary)
  T rb;     // Index of the last elemen (right boundary)
  T parent;  // The parent of this lcp-interval
             // (-1 if this is the top interval),
             // in the order in which it appears in this array (of
             // lcp-intervals).  Note: this order is neither top-down or
             // bottom-up; you can treat it as arbitrary.
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const LcpInterval<T> &interval) {
  static constexpr char kSep = ' ';
  os << interval.lcp << kSep << interval.lb << kSep << interval.rb << kSep
    << interval.parent;
  return os;
}


/*
  Create an array of struct LcpInterval which describes the Lcp intervals
  corresponding to the internal nodes of the suffix tree, and allows you
  to easily run algorithms on this tree.  This data structure is not
  very memory-optimized and doesn't correspond to anything in the literature,
  although the basic tree traversal algorithm comes from
  [Mohamed Ibrahim Abouelhoda, Stefan Kurtz, Enno Ohlebusch: Replacing suffix
  trees with enhanced suffix arrays. Journal of Discrete Algorithms 2 (2004)
  53-86.]  and was originally adapted from [Kasai, Lee, Arimura, Arikawa, Park:
  Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its
   Applications, CPM 2001].

  The motivation here is that we are likely limited more by time than memory,
  and we want a data structure that is relatively simple to use.

   Template args: T is a signed type, intended to be int16_t or int32_t

     @param [in] c  Context pointer, used to create arrays.  Required to
                     be a CPU context pointer for now.
     @param [in] seq_len  The length of the text for which we have a suffix
                   array
     @param [in] lcp_array  The LCP array, as computed by CreateLcpArray()
     @param [out] lcp_intervals    A *newly created* array of LcpInterval<T>
                   will be written to here, of length no greater than seq_len.
                   They will be in dfs post order.  Children precede their
                   parents.
     @param [out] leaf_parent_intervals  If this is non-NULL, a newly
                   created array of size seq_len will be written to here,
                   saying, for each leaf in the suffix tree (corresponding to
                   positions in the suffix array) which lcp-interval
                   is its parent.  Indexes into this array correspond to
                   indexes into the suffix array, and values correspond
                   to indexes into `lcp_intervals`.
 */
template <typename T>
void CreateLcpIntervalArray(ContextPtr c,
                            T seq_len,
                            T *lcp_array,
                            Array1<LcpInterval<T> > *lcp_intervals,
                            Array1<T> *leaf_parent_intervals);

/*
  Modifies `leaf_parent_intervals` to give us, for each position in the suffix
  array (i.e. each suffix), the tightest enclosing lcp-interval that has
  nonzero count.  This is used in finding the highest-order match of
  a position in a text (i.e. the longest matching history).

   Template args: T is a signed type, intended to be int16_t or int32_t

     @param [in] seq_len  The length of the sequence,
                          including the terminating $.
     @param [in] lcp_intervals  The array of lcp intervals, as returned
                     by CreateLcpIntervalArray
     @param [in] counts_exclusive_sum  The exclusive-sum of counts of symbols in
                   the original text array, in the order given by the suffix
                   array, e.g. the original counts would have satisfied
                   suffix_counts[i] = counts[suffix_array[i]], and then
                   counts_exclusive_sum is the exclusive-sum of suffix_counts.
                   Must satisfy counts_exclusive_sum->Dim() == seq_len + 1.
                   The original counts would have been 1 for "keys" and 0 for
                   "queries", so an interval with nonzero difference in
                   counts_exclusive_sum is an interval containing at least
                   one key.
     @param [in,out] leaf_parent_intervals  At entry, this will contain,
                   for each leaf of the suffix tree (i.e. each position
                   in the suffix array) the index of the tightest enclosing
                   lcp-interval, i.e. an index into `lcp_intervals`.
                   At exit, it will contain the index of the tightest
                   enclosing *nonempty* lcp-interval.
 */
template <typename T>
void FindTightestNonemptyIntervals(T seq_len,
                                   Array1<LcpInterval<T> > *lcp_intervals,
                                   Array1<T> *counts_exclusive_sum,
                                   Array1<T> *leaf_parent_intervals);

/*
    For "query" sentences, this function gets the mean and variance of scores
    from the best matching words-in-context in a set of provided "key"
    sentences.  This matching process matches the word and the words preceding
    it, looking for the highest-order match it can find (it's intended for
    approximating the scores of models that see only left-context, like language
    models).  It is an efficient implementation using suffix arrays (done on CPU
    for now, since the implementation is not very trivial).  The intended
    application is in estimating the scores of hypothesized transcripts, when we
    have actually computed the scores for only a subset of the hypotheses.

      @param [in] tokens  A ragged tensor of int32_t with 2 or 3 axes (this
                  function recurses).  If 2 axes, this represents a collection of
                  key and query sequences (keys have count==1, query count==0).
                  If 3 axes, this represents a set of such collections
                  and retrieval should be done independently.

               2-axis example:
                 [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ]
               3-axis example:
               [ [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ],
                 [ [ hi, my, name, is, eos ], [ bye, my, name, is, eos ] ], ... ]

                 .. where the words would actually be represented as integers,
                 and the eos might be -1.  The eos symbol is required if this
                 code is to work as intended (otherwise this code will not
                 be able to recognize when we have reached the beginnings
                 of sentences when comparing histories).  bos symbols are
                 allowed but not required.

     @param [in] scores  An array with scores.Dim() == tokens.NumElements();
                 this is the item for which we are requesting best-matching
                 values (as means and variances in case there are multiple
                 best matches).  In our anticipated use, these would represent
                 scores of words in the sentences, but they could represent
                 anything.
     @param [in] counts  An array with counts.Dim() == tokens.NumElements(),
                 containing 1 for words that are considered "keys" and 0 for
                 words that are considered "queries".  Typically some entire
                 sentences will be keys and others will be queries.
     @param [in] eos The value of the eos (end of sentence) symbol; internally, this
               is used as an extra padding value before the first sentence in each
               collection, so that it can act like a "bos" symbol.
     @param [in] min_token  The lowest possible token value, including the bos
               symbol (e.g., might be -1).
     @param [in] max_token  The maximum possible token value.  Be careful not to
               set this too large the implementation contains a part which
               takes time and space O(max_token - min_token).
     @param [in] max_order  The maximum n-gram order to ever return in the
              `ngram_order` output; the output will be the minimum of max_order
              and the actual order matched; or max_order if we matched all the
              way to the beginning of both sentences.  The main reason this is
              needed is that we need a finite number to return at the
              beginning of sentences.

     @param [out] mean  For query positions, will contain the mean of the
              scores at the best matching key positions, or zero if that is
              undefined because there are no key positions at all.  For key positions,
              you can treat the output as being undefined (actually they
              are treated the same as queries, but won't match with only
              themselves because we don't match at singleton intervals).  This array
              will be allocated if it did not have the correct size at
              entry.
     @param [out] var  Like `mean`, but contains the (centered) variance
              of the best matching positions.
     @param [out] counts_out  The number of key positions that contributed
              to the `mean` and `var` statistics.  This should only
              be zero if `counts` was all zero.  Will be allocated
              if it did not have the correct size at entry.
     @param [out] ngram_order  The n-gram order corresponding to the
             best matching positions found at each query position, up
             to a maximum of `max_order`; will be `max_order` if we matched all
             the way to the beginning of a sentence.  Will be allocated if it
             did not have the correct size at entry.
*/
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
                          Array1<int32_t> *ngram_order);
}  // namespace k2
#endif  // K2_CSRC_NBEST_H_
