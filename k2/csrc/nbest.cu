/**
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey)
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

#include "k2/csrc/nbest.h"

// This is not really a CUDA file but for build-system reasons I'm currently leaving it
// with the .cu extension.

namespace k2 {

template <typename T>
inline bool Leq(T a1, T a2, T b1, T b2) {
  // lexicographic order for pairs, used in CreateSuffixArray()
  return(a1 < b1 || a1 == b1 && a2 <= b2);
}
template <typename T>
inline bool Leq(T a1, T a2, T a3, T b1, T b2, T b3) {
  // lexicographic order for triples, used in CreateSuffixArray()
  return(a1 < b1 || a1 == b1 && Leq(a2,a3, b2,b3));
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
  T* c = new T[K + 1]; // counter array
  for (T i = 0; i <= K; i++) c[i] = 0; // reset counters
  for (T i = 0; i < n; i++) c[r[a[i]]]++; // count occurrences
  for (T i = 0, sum = 0; i <= K; i++) {// exclusive prefix sums
    T t = c[i]; c[i] = sum; sum += t;
  }
  for (T i = 0; i < n; i++) b[c[r[a[i]]]++] = a[i]; // sort
  delete [] c;
}


// See documentation in nbest.h, where we use different names
// for the arguments (here, we leave the names the same as in
// https://algo2.iti.kit.edu/documents/jacm05-revised.pdf.
template <typename T>
void CreateSuffixArray(const T* text, T n, T K, T* SA) {
  //assert(text[0] <= text[n-1]);  // spot check that termination symbol is larger
                                 // than other symbols; <= in case n==1.
  if (n == 1) {  // The paper's code didn't seem to handle n == 1 correctly.
    SA[0] = 0;
    return;
  }
  T n0=(n+2)/3, n1=(n+1)/3, n2=n/3, n02=n0+n2;
  T* R = new T[n02 + 3]; R[n02]= R[n02+1]= R[n02+2]=0;
  T* SA12 = new T[n02 + 3]; SA12[n02]=SA12[n02+1]=SA12[n02+2]=0;
  T* R0 = new T[n0];
  T* SA0 = new T[n0];
  //******* Step 0: Construct sample ********
  // generate positions of mod 1 and mod 2 suffixes
  // the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
  for (T i=0, j=0; i < n+(n0-n1); i++) if (i%3 != 0) R[j++] = i;
  //******* Step 1: Sort sample suffixes ********
  // lsb radix sort the mod 1 and mod 2 triples
  RadixPass(R, SA12, text+2, n02, K);
  RadixPass(SA12, R , text+1, n02, K);
  RadixPass(R, SA12, text, n02, K);

  // find lexicographic names of triples and
  // write them to correct places in R
  T name = 0, c0 = -1, c1 = -1, c2 = -1;
  for (T i = 0; i < n02; i++) {
    if (text[SA12[i]] != c0 || text[SA12[i]+1] != c1 || text[SA12[i]+2] != c2)
    { name++; c0 = text[SA12[i]]; c1 = text[SA12[i]+1]; c2 = text[SA12[i]+2]; }
    if (SA12[i] % 3 == 1) { R[SA12[i]/3] = name; } // write to R1
    else { R[SA12[i]/3 + n0] = name; } // write to R2
  }
  // recurse if names are not yet unique
  if (name < n02) {
    CreateSuffixArray(R, n02, name, SA12);
    // store unique names in R using the suffix array
    for (T i = 0; i < n02; i++) R[SA12[i]] = i + 1;
  } else // generate the suffix array of R directly
    for (T i = 0; i < n02; i++) SA12[R[i] - 1] = i;
  //******* Step 2: Sort nonsample suffixes ********
  // stably sort the mod 0 suffixes from SA12 by their first character
  for (T i=0, j=0; i < n02; i++) if (SA12[i] < n0) R0[j++] = 3*SA12[i];
  RadixPass(R0, SA0, text, n0, K);
  //******* Step 3: Merge ********
  // merge sorted SA0 suffixes and sorted SA12 suffixes
  for (T p=0, t=n0-n1, k=0; k < n; k++) {
    // i is pos of current offset 12 suffix
    T i = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2) ;
    T j = SA0[p]; // pos of current offset 0 suffix
    if (SA12[t] < n0 ? // different compares for mod 1 and mod 2 suffixes
        Leq(text[i], R[SA12[t] + n0], text[j], R[j/3]) :
        Leq(text[i],text[i+1],R[SA12[t]-n0+1], text[j],text[j+1],R[j/3+n0]))
    { // suffix from SA12 is smaller
      SA[k] = i; t++;
      if (t == n02) // done --- only SA0 suffixes left
        for (k++; p < n0; p++, k++) SA[k] = SA0[p];
    } else { // suffix from SA0 is smaller
      SA[k] = j; p++;
      if (p == n0) // done --- only SA12 suffixes left
        for (k++; t < n02; t++, k++)
          SA[k] = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2);
    }
  }
  delete [] R; delete [] SA12; delete [] SA0; delete [] R0;
}

// Instantiate template for int32_t and int16_t
template void CreateSuffixArray(const int32_t* text, int32_t n,
                                int32_t K, int32_t* SA);
template void CreateSuffixArray(const int16_t* text, int16_t n,
                                int16_t K, int16_t* SA);



// This implements Kasai's algorithm, as summarized here
// https://people.csail.mit.edu/jshun/lcp.pdf
// (Note: there seem to be some wrong implementations of
// Kasai's algorithm online).
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
                            Array1<T> *lcp_intervals_order,
                            Array1<T> *leaf_parent_intervals) {


  *lcp_intervals = Array1<LcpInterval<T> >(c, seq_len);
  LcpInterval<T> *lcp_intervals_data = lcp_intervals->Data();

  Array1<T> intervals_order(c, seq_len);
  T *intervals_order_data = intervals_order.Data();

  Array1<T> leaf_parent(c, seq_len);
  T *leaf_parent_data = leaf_parent.Data();


  // This is the stack from Algorithm 1 and Algorithm 2 of
  // http://www.mi.fu-berlin.de/wiki/pub/ABI/RnaSeqP4/enhanced-suffix-array.pdf
  // (you can refer to the papers mentioned in the documentation in nbest.h if this link goes
  // dead).
  //
  // The 'begin', 'last' and 'lcp' members correspond to the 'lb', 'rb' and 'lcp'
  // members mentioned there; the 'parent' member is used temporarily on the stack
  // to refer to the index of this LcpInterval in `lcp_intervals_data`, i.e.
  // it can be interpreted as a 'self' pointer.
  std::vector<LcpInterval<T> > stack;

  // A separate stack, of leaves of suffix tree; we maintain this so that
  // we can assign the `leaf_parent` array.
  std::vector<T> leaf_stack;

  // lcp=0; begin=0; last=undefined; self=0  (interpreting the 'parent' member
  // as index-of-self
  T next = 0;  // Will always store the next free index into `lcp_intervals_data`
  T dfs_next = 0;  // Will always store the next free index into `intervals_order_data`;
                   // this is an ordering of the indexes into `lcp_intervals_data`
                   // that corresponds to depth-first search.
  T last_interval = -1;  // Will store an index into `lcp_intervals`; this comes
                         // from Algorithm 2 mentioned above
  stack.push_back({0, 0, T(seq_len - 1), next++ });
  lcp_intervals_data[0] = stack.back();
  lcp_intervals_data[0].parent = -1;
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
      lb = stack.back().lb;
      while (!leaf_stack.empty() && leaf_stack.back() >= lb) {
        leaf_parent_data[leaf_stack.back()] = last_interval;
        leaf_stack.pop_back();
      }

      // process(last_interval):
      lcp_intervals_data[last_interval] = stack.back();
      //  Previously tried doing:
      //   stack.back().rb = i - 1;
      // a bit further above, but hit some kind of compiler problem, the assignment
      // had no effect (back() is supposed to return a reference).
      lcp_intervals_data[last_interval].rb = i - 1;
      intervals_order_data[dfs_next++] = last_interval;
      stack.pop_back();
      if (lcp_array_i <= stack.back().lcp) {
        // lcp_intervals_data[last_interval].parent represents the parent
        // of `last_interval`; `stack.back().parent` currently represents
        // the intended position of stack.back() itself, not of its parent.
        lcp_intervals_data[last_interval].parent = stack.back().parent;
        last_interval = -1;
      }
    }
    if (lcp_array_i > stack.back().lcp) {
      if (last_interval >= 0) {
        lcp_intervals_data[last_interval].parent = next;
        last_interval = -1;
      }
      stack.push_back({lcp_array_i, lb, -1, next++});
    }
  }
  assert(stack.size() == 1);
  intervals_order_data[dfs_next++] = 0;
  leaf_stack.push_back(seq_len - 1);
  while (!leaf_stack.empty()) {
    leaf_parent_data[leaf_stack.back()] = 0;
    leaf_stack.pop_back();
  }
  assert(dfs_next == next);


  *lcp_intervals = lcp_intervals->Range(0, next);
  if (lcp_intervals_order)
    *lcp_intervals_order = intervals_order.Range(0, next);
  if (leaf_parent_intervals)
    *leaf_parent_intervals = leaf_parent;

}

// Instantiate template
template
void CreateLcpIntervalArray(ContextPtr c,
                            int32_t seq_len,
                            int32_t *lcp_array,
                            Array1<LcpInterval<int32_t> > *lcp_intervals,
                            Array1<int32_t> *lcp_intervals_order,
                            Array1<int32_t> *leaf_parent_intervals);
template
void CreateLcpIntervalArray(ContextPtr c,
                            int16_t seq_len,
                            int16_t *lcp_array,
                            Array1<LcpInterval<int16_t> > *lcp_intervals,
                            Array1<int16_t> *lcp_intervals_order,
                            Array1<int16_t> *leaf_parent_intervals);


}  // namespace k2
