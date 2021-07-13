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
// of n-best candidates to select for rescoring.  The selection scheme is a little
// complex.  It is intended to be used in a context where we do multiple successive
// rounds of n-best list rescoring, and we use the results of the 1st round
// to guide selection of candidates in the second round.  So for each word
// in each n-best path that we are considering, we find the best-matching
// positions among those that we evaluated in the first round and we use those
// as inputs to a model that predicts the scores of words after n-best-list
// rescoring.
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

    @param [in] text_array  Pointer to the input array of symbols
           (all pointers must be CPU pointers only, for now),
            whose suffixes are to be sorted.  Logically this
            has length `seq_len`, and symbols are required
            to be in the range [1..max_symbol].  It is required
            to be terminated by 3 zeros, i.e.
            text_array[seq_len] == text_array[seq_len+1] == text_array[seq_len+2] == 0
    @param [in] seq_len  Length of the symbol sequence (`text_array`
            must be longer than this by at least 3, for termination.)
            Require seq_len >= 0
    @param [out] suffix_array   A pre-allocated array of length
             `seq_len + 1`.  At exit it will contain a permutation of
             the list [ 0, 1, ... seq_len ], interpreted
             as the start indexes of suffixes of `text_array`,
             with the property that the sub-arrays of `text_array`
             starting at these positions are lexicographically sorted.
             For example, as a trivial case, if seq_len = 3
             and text_array contains [ 3, 2, 1, 0, 0, 0 ], then
             `suffix_array` would contain [ 2, 1, 0 ] at exit.
    @param [in] max_symbol  A number that must be >= the largest
             number that might be in `text_array`.  The work done
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

   Template args: T should be a signed integer type, we
   plan to instantiate this for int32_t and int16_t only.

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
  lcp-intervals correspond to the nodes in the suffix trie; they are a concept
  used with suffix arrays, and are derived from the LCP table (see lcp_array
  output of CreateLcpArray).  Take care with the notation here: intervals are
  "closed intervals" so [i,j] means i,i+1,...,j, i.e. the RHS is the index of
  the last element, not one past the last element.

  Notation: [i,j] is an lcp-interval with lcp-value l, if:
     0 <= i < j < seq_len
     lcptab[i] < l
     lcptab[j+1] < l
     l is the minimum of (lcptab[i], lcptab[i+1], ..., lcptab[j])
  lcp-intervals correspond to the internal nodes of the suffix trie, so
  they always contain at least two children (where children can be
  leaves, corresponding indexes into the suffix array, or other
  lcp-intervals).

  Type LcpInterval is used to store information about the lcp interval,
  which we'll later use in algorithms that traverse the suffix tree.


 */
template <typename T>
struct LcpInterval {
  // Represents the lcp-interval [begin,last] with lcp-value `lcp`
  T lcp;    // The lcp-value of the lcp-interval, which is the length of the
            // longest prefix shared by all suffixes in this interval.
  T begin;  // Index of the first element
  T last;   // Index of the last element; we don't call this 'end' because that
            // is generally used to mean one past the end.
  T parent; // The parent of this lcp-interval (-1 if this is the top interval),
            // in the order in which it appears in this array (of
            // lcp-intervals).  Note: this order is neither top-down or
            // bottom-up; you can treat it as arbitrary.
};


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

     @param [in] c  Context pointer, used to create arrays.  Required to
                     be a CPU context pointer for now.
     @param [in] seq_len  The length of the text for which we have a suffix
                   array
     @param [in] lcp_array  The LCP array, as computed by CreateLcpArray()
     @param [out] lcp_intervals    A *newly created* array of LcpInterval<T>
                   will be written to here, of length no greater than seq_len.
     @param  [out] lcp_intervals_order  If this is non-NULL, a newly
                   created array will be written to here, giving a bottom-up
                   order of the lcp-intervals so that each child comes before
                   its parent.  This is a permutation of the numbers
                   [0,1,...lcp_intervals->Dim()-1].
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
                            Array1<T> *lcp_intervals_order,
                            Array1<T> *leaf_parent_intervals);







}
#endif  // K2_CSRC_NBEST_H_
