/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_UTILS_H_
#define K2_CSRC_UTILS_H_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/eval.h"
#include "k2/csrc/math.h"

namespace k2 {

// Some quite low-level utilities.
// CAUTION: this is not up to date, I will simplify this probably.

/*
  sizes concept

  A vector of sizes is a vector of non-negative integers, like [ 5 4 0 1 3 ].
  This just establishes a naming convention.  In most cases the 'sizes' will
  be the lengths of sub-lists in a list of lists.

  Relation to other concepts:
   A vector of sizes can be seen as the difference of successive elements of
  a vector of row_splits, i.e. sizes[i] = row_splits[i+1] - row_splits[i].
*/

/*
  row_splits concept / row-splits concept
  (Note, this has been named for compatibility with TensorFlow's RaggedTensor).

  A row_splits vector is a vector of the form, say, [ 0 5 9 9 10 13 ].
  i.e. it starts with 0 and is non-decreasing.  It will often be encountered
  as the exclusive-sum of a vector of 'sizes' (see 'sizes concept' above),
  with a size one greater than that of the corresponding 'sizes'.  It
  will represent the positions in a single linearized list, of where we
  put the elements of a list of sub-lists.  So in the example above,
  sub-list 0 occupies positions 0,1,2,3,4, sub-list 1 occupies positions
  5,6,7,8, and so on.  Caution: the number of elements of the row_splits vector
  equals the number of sub-lists PLUS ONE.

  Relation to other concepts:
    See 'row_ids concept' where its relation to 'row_splits' is described.
    If row_splits = [ 0 2 3 5 ], the corresponding 'row_ids' is:
    row_ids = [ 0 0 1 2 2 ].
*/

/*
  row_ids concept / row-ids concept
  (Note, this has been named for compatibility with TensorFlow's RaggedTensor).

  A vector of row_ids is a vector of the form
    [ 0 0 0 1 1 2 2 2 ]
  or in general any nonnegative, non-decreasing list of integers.  Each
  value represents the index of the sub-list to which that position belongs; for
  instance, if we had a list-of-lists like [ a b c ] [ d e ] [ f g h ], the
  above vector of row_ids would describe its structure.

  Relation to other concepts:
    A vector of row_ids can arise as the cumulative sum of a vector of tails.
    A vector of row_ids and a vector of row_splits represent the same
  information in different ways, satisfying row_splits[row_ids[i]] <= i <
  row_splits[row_ids[i] + 1] and row_splits[row_splits.size() - 1] ==
  row_ids.size().

*/

/*
  tails concept

  A vector of tails is a vector containing zeros and ones; each '1' represents
  the last element of a sub-list.  For example [ 0 1 0 0 0 1 1 ] to represent
  a list of sub-lists like [ x x ] [ y y y y ] [ z ].  The last element will
  always be 1.

  Relation to other concepts:
    The exclusive cumulative sum of a vector of tails is a vector of row_ids.
  E.g. the above example, with exclusive cumulative sum, is:
    [ 0 0 1 1 1 1 2 ].
 */

/*

  Index naming scheme

  In a ragged tensor t with n axes (say, 3) the actual elements will be written
  in a linear array we'll have various levels of indexes that allow us to
  look up an element given the hierarchical indexes and vice versa.  A 3-d
  ragged tensor will have t.RowIds(1), t.RowSplits(1), t.RowIds(2),
  t.RowSplits(2), and the actual elements.  We have a naming scheme that
  expresses what information is packed into a single integer.

  Some entry-level facts about the naming scheme are:

     - The hierarchical indexes into the tensor (3 of them for a tensor with 3
       axes), we call idx0, idx1 and idx2
     - The linear index into the elements, we call idx012 because it includes
       all 3 values.
     - t.RowSplits(1) would map from an idx0 to an idx0x.  The x here
       takes the place of a 1 and that replacement means "actually the index
       here is definitely zero".  Any specific idx0x that we have will be
       for a particular idx0.

   For more details, it's best to use an example.

     Ragged<int32_t> t('[ [ [ 1 2 ] [ 5 ] ] [ [ 7 8 9 ] ] ]')

     # which will give us:
     t.RowSplits(1) == [ 0 2 3 ]    # indexed by idx0, elements are idx0x
     t.RowIds(1) == [ 0 0 1 ]       # indexed by idx01, elements are idx0
     t.RowSplits(2) == [ 0 2 3 6 ]  # indexed by idx01, elements are idx01x
     t.RowIds(2) == [ 0 0 1 2 2 2 ] # indexed by idx012, elements are idx01
     t.values == [ 1 2 5 7 8 9 ]   # indexed by idx012, elements are whatever
                                   # values we're storing.

     Sometimes we'll want to know the number of elements in sub-lists, and we
     have a notation for the computations involved in that.  Suppose we want to
     know the number of elements in T[0].  We'll compute:
       int32_t idx0 = 0,
           idx0x = t.RowSplits(1)[idx0],
           idx0x_next = t.RowSplits(1)[idx0 + 1],
           idx0xx = t.RowSplits(2)[idx0x],
           idx0xx_next = t.RowSplits(2)[idx0x_next],
           len12 = idx0xx_next - idx0xx
     (The _next suffix is used when we're querying the most specific known index
     plus one, in this case index 0 but for instance, idx01x_next would mean
     that we were querying idx01x after incrementing the index on axis 1.)

     We also might sometimes want to know an offset of an element within the
     part of the array that starts with a particular prefix of that index.
     E.g. suppose we want the offset of element t[ idx0, idx1, idx2 ]
     relative to the start of the sub-array t[idx0].  We'd do this as
     follows:
        int32_t idx0, idx1, idx2;  # provided
        int32_t idx0x = t.RowSplits(1)[idx0],
            idx01 = idx0x + idx1,
            idx01x = t.RowSplits(2)[idx01],
            idx012 = idx01x + idx2,
            idx0xx = t.RowSplits(2)[idx0x],
            idx12 = idx012 - idx0xx;
     In the last line above, when we subtract idx012 - idx0xx we lose the
     leading "0" because the zeroth index was the same in the two things being
     subtracted.  Note: in an expression like idx0xx_next - idx0xx we don't get
     idxxxx because index zero is *different*.  However, the result of
     idx01x_next - idx01x would be written idx1x because index zero would be
     the same.

     The advantage of this naming scheme is that the 'type' that operations give
     is intuitively obvious and any mismatches will tend to be obvious in
     an individual line of code once you have understood the naming scheme
     and its rules.

  Note: We also have a naming convention according to the index naming scheme
        above. For a Ragged array with index `[fsa][state][arc]` (say, an FsaVec),
        we usually call `idx0` as `fsa_idx0`, `idx01` as `state_idx01`, `idx012`
        as `arc_idx012`, that is, `theThingWeIndex_idx0[123]`.
*/

/**
   Perform exclusive cumulative sum: dest[i] = 0 + src[0] + src[1] + ...
  src[i-1] for 0 <= i < n.  Note: although the input for 0 <= i < n-1 is all
  that affects the output, the last element src[n-1] may still be accessed in
  the CUDA kernel so you should make sure it was allocated as part of the array
  being summed even if the value was not set.

      @param [in] c     Context object, specifies CPU or GPU
      @param [in] n     Number of elements in the input and output arrays
                        (although only items up to n-1 in the input array will
                        affect the result).  Must be >= 0
      @param [in] src    Array from which to read the input data
      @param [out] dest    Array to which to write the exclusive sum
                           (may be the same as src)
 */
template <typename SrcPtr, typename DestPtr>
void ExclusiveSum(ContextPtr c, int32_t n, SrcPtr src, DestPtr dest);

/**
  Perform inclusive cumulative sum: dest[i] = 0 + src[0] + src[1] + ...
  src[i] for 0 <= i < n.

      @param [in] c     Context object, specifies CPU or GPU
      @param [in] n     Number of elements in the input and output arrays
      @param [in] src    Array from which to read the input data
      @param [out] dest    Array to which to write the inclusive sum
                           (may be the same as src)
 */
template <typename SrcPtr, typename DestPtr>
void InclusiveSum(ContextPtr c, int32_t n, SrcPtr src, DestPtr dest);


/* Return the maximum value of the device array 't'.  Note: the sum will be
   initialized with T(0).

   Implementation notes: similar to ExclusiveSum.  We might at some point
   combine this with ExclusiveSum to (optionally) get the max value with little
   additional cost.
 */
template <typename T>
T MaxValue(ContextPtr c, int32_t nelems, const T *t);

/*
  This is a rather special purpose function that is used in RaggedShape.

  It sets row_ids[i] to the index j to which position i 'belongs' according to
  the array `row_splits`.  `row_splits` is expected to be an array containing
  the exclusive sum of a sequence of nonnegative integers corresponding to sizes
  of sub-lists, so suppose there was a original sequence sizes = [ 2 1 0 4 ] and
  row_splits = [ 0 2 3 3 7 ] then we would fill row_ids with: row_ids = [ 0 0 1
  3 3 3 3 ]

       @param [in] c   ContextPtr, points to the context to which the
                       data belongs (e.g. CPU or GPU).
       @param [in] num_rows
                       Number of rows in the ragged matrix; must be >= 0.
       @param [in] row_splits
                       Start of row_splits vector, must be non-decreasing and
                       start from zero.  Length is num_rows + 1.
                       row_splits[0] must equal 0 and row_splits[num_rows] must
                       equal num_elems.
       @param [in] num_elems  Number of elements, in all the rows together.
                       Note: the length of row_ids equals num_elems.
       @param [out] row_ids   Start of row_ids vector, we write the output to
                              here. Length is num_elems.

   Note: there is another function of the same name using the Array1 interface,
   declared in array_ops.h, that may be more convenient.
*/
void RowSplitsToRowIds(ContextPtr c, int32_t num_rows,
                       const int32_t *row_splits, int32_t num_elems,
                       int32_t *row_ids);

/*
  This function works out the row_id of `this` index from row-splits, using
  binary search.  Specifically, it returns i such that row_splits[i] <= index <
  row_splits[i+1]. row_splits should be a vector with at least num_rows+1
  elements.
       @param [in] num_rows      Number of rows (row-id will be less than this)
       @param [in] row_splits    Row-splits vector, of size num_rows + 1 (search
                                 for `row_splits concept` near the top of
                                 utils.h for more info)
       @param [in] index         Linear index (e.g. idx01) for which we're
                                 querying which row it is from
       @param [in] num_indexes   Total number of indexes (should equal
                                 row_splits[num_rows]); right now it's not used,
                                 but in future it might be used for a heuristic,
                                 for the initial guess of where to start the
                                 binary search.
       @return                   Returns i such that row_splits[i] <= index <
                                 row_splits[i+1] and 0 <= i < num_rows;
                                 will die with assertion in debug mode if such
                                 an i does not exist.
   TODO(dan): make this compile, apparently std::lower_bound won't work on GPU
   so we should manually do the binary search.
 */
__forceinline__ __host__ __device__ int32_t
RowIdFromRowSplits(int32_t num_rows, const int32_t *row_splits, int32_t index,
                   int32_t num_indexes) {
  // lower_bound gives the first i in row_splits that's greater than `index`.
  // That implies the previous one is <= index.
  //
  // auto i =
  // std::lower_bound(row_splits + 1, row_splits + num_rows + 1, index) - 1;
  // K2_DCHECK(static_cast<uint32_t>(i) < static_cast<uint32_t>(num_rows));
  // TODO:  Implement std::lower_bound ourselves.
  // return *i;
  return 0;  // TODO:  Does not work.
}

/*
  See above for 'row_ids concept' and 'row_splits' concept.
  This function turns a vector of row_ids into a vector of row_splits,
  e.g. given [ 0 0 1 1 1 2 ] it would produce [ 0 2 5 6 ].

   @param [in] c    ContextPtr, points to the context to which the
                    data belongs (e.g. CPU or GPU).
   @param [in] num_elems   The number of elements in the irregular array
   @param [in] row_ids   row_ids vector of length num_elems (
                    row_ids[num_elems - 1] + 1 must equal num_rows). Must be
                    non-decreasing.
   @param [in] no_empty_rows   If the caller happens to know that no rows
                    of the irregular array are empty
                    (i.e. that row_ids[i] +1 >= row_ids[i+1]) they may
                    set this to true, which will improve speed.
                    If you are not sure, set this to false.
   @param [in] num_rows   Number of rows in the irregular array, must
                    be greater than any element of row_ids
   @param [out] row_splits  Row-splits vector that this function
                    writes to, of length num_rows + 1.  row_splits[num_rows]
                    will equal num_elems.

   Note: there is another function of the same name using the Array1 interface,
   declared in array_ops.h, that may be more convenient.
 */
void RowIdsToRowSplits(ContextPtr c, int32_t num_elems, const int32_t *row_ids,
                       bool no_empty_rows, int32_t num_rows,
                       int32_t *row_splits);

__host__ __device__ __forceinline__ int32_t FloatAsInt(float f) {
  union {
    float f;
    int32_t i;
  } u;
  u.f = f;
  return u.i;
}

__host__ __device__ __forceinline__ float IntAsFloat(int32_t i) {
  union {
    float f;
    int32_t i;
  } u;
  u.i = i;
  return u.f;
}

/* Atomically decrement *i and return true if it is zero after the decrement (it
   is an error if it becomes less than zero).
*/
__host__ __device__ __forceinline__ bool AtomicDecAndCompareZero(int32_t *i) {
#ifdef __CUDA_ARCH__
  int32_t old = atomicAdd(i, -1);
  K2_CHECK_GT(old, 0);
  return old == 1;
#else
  // For host code, we assume single-threaded for now).
  int32_t i_value = *i;
  *i = i_value - 1;
  K2_CHECK_GE(i_value - 1, 0);
  return i_value - 1 == 0;
#endif
}

/* Add a value to a memory address atomically.

   It implements `*address += value`.

   CAUTION: For host code, we assume single-threaded for now.

   @param  [inout]  address  The memory address.
   @param  [in]      value    The value to be added.
   @return    The value before added.
 */
template <typename T>
__host__ __device__ __forceinline__ T AtomicAdd(T *address, T value) {
#ifdef __CUDA_ARCH__
  return atomicAdd(address, value);
#else
  // For host code, we assume single-threaded for now).
  T res = *address;
  *address += value;
  return res;
#endif
}

// atomicAdd() for double-precision floating-point numbers is not available on
// devices with compute capability lower than 6.0.
// The following implementation is copied from
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__host__ __device__ __forceinline__ double AtomicAdd(double *address,
                                                     double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#elif defined(__CUDA_ARCH__)
  double res = *address;
  // clang-format off
  unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull;  // NOLINT
  unsigned long long int assumed;  // NOLINT
  // clang-format on
  do {
    assumed = old;
    old =
        atomicCAS(address_as_ull, assumed,
                  __double_as_longlong(value + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
  return res;
#else
  // For host code, we assume single-threaded for now.
  double res = *address;
  *address += value;
  return res;
#endif
}

/*
 1:1 Conversion float <---> sortable int32_t We convert floats to sortable ints
 in order to use native atomics operation, which are way faster than looping
 over atomicCAS
*/
__host__ __device__ __forceinline__ int32_t FloatToOrderedInt(float f) {
  int32_t i = FloatAsInt(f);
  return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__host__ __device__ __forceinline__ float OrderedIntToFloat(int32_t i) {
  return IntAsFloat((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

/*
  host version of Cuda's atomicMax function, marked __host__ (the default) for
  clarity.  So we can use this in lambdas that run on both host and device.
 */
__host__ __device__ __forceinline__ int32_t AtomicMax(int32_t *address,
                                                      int32_t val) {
#if defined(__CUDA_ARCH__)
  return atomicMax(address, val);
#else
  int32_t old = *address;
  if (old < val) *address = val;
  return old;
#endif
}

// have to figure out if there's a better place to put this
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[ ";
  for (auto iter = vec.begin(); iter != vec.end(); ++iter) os << *iter << ' ';
  os << ']';
  return os;
}

template <typename T>
struct MaxOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return (a > b) ? a : b;
  }
};

template <typename T>
struct MinOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return (a > b) ? b : a;
  }
};

template <typename T>
struct PlusOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a + b;
  }
};

template <typename T>
struct MinusOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a - b;
  }
};

template <typename T>
struct TimesOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a * b;
  }
};

template <typename T>
struct BitAndOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a & b;
  }
};

template <typename T>
struct BitOrOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a | b;
  }
};

template <typename T>
struct LessThan {
  __host__ __device__ __forceinline__ bool operator()(const T &a,
                                                      const T &b) const {
    return a < b;
  }
};

template <typename T>
struct GreaterThan {
  __host__ __device__ __forceinline__ bool operator()(const T &a,
                                                      const T &b) const {
    return a > b;
  }
};

#define K2_MIN_LOG_DIFF_FLOAT -15.9423847198486328125f  // logf(FLT_EPSILON)
#define K2_MIN_LOG_DIFF_DOUBLE \
  -36.0436533891171535515240975655615329742431640625  // log(DBL_EPSILON)

template <typename T>
struct LogAdd;

template <>
struct LogAdd<double> {
  __host__ __device__ __forceinline__ double operator()(double x,
                                                        double y) const {
    double diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= K2_MIN_LOG_DIFF_DOUBLE) {
      double res;
      res = x + log1p(exp(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

template <>
struct LogAdd<float> {
  __host__ __device__ __forceinline__ float operator()(float x, float y) const {
    float diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= K2_MIN_LOG_DIFF_DOUBLE) {
      float res;
      res = x + log1pf(expf(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

}  // namespace k2

#include "k2/csrc/utils_inl.h"

#endif  // K2_CSRC_UTILS_H_
