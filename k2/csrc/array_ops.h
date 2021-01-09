/**
 * @brief
 * array_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_OPS_H_
#define K2_CSRC_ARRAY_OPS_H_

#include <cassert>
#include <type_traits>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

// Note, I'm not sure about the name of this file, they are not ops like in
// TensorFlow, but procedures..

namespace k2 {
/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().  This is not the only way to transpose a matrix, you can also
  do: dest = Array2<T>(src.ToTensor().Transpose(0,1)), which will likely call
  this function

     @param [in] c   Context to use, must satisfy
                     `c.IsCompatible(src.Context())` and
                     `c.IsCompatible(dest->Context())`.
     @param [in] src  Source array to transpose
     @param [out] dest  Destination array; must satisfy
                        `dest->Dim1() == src.Dim0()` and
                        `dest->Dim0() == src.Dim1()`.
                        At exit, we'll have dest[i,j] == src[j,i].
*/
template <typename T>
void Transpose(ContextPtr &c, const Array2<T> &src, Array2<T> *dest);

/*
  Sets 'dest' to exclusive prefix sum of 'src'.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data (possibly &src).  Must satisfy
                       dest.Dim() == src.Dim() or
                       dest.Dim() == src.Dim() + 1,
                       but in the latter case we require that the memory
                       region inside src be allocated with at least one extra
                       element, because the exclusive-sum code may read from
                       it even though it doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j]
                       for i > 0. dest[0] is always 0. Must be pre-allocated and
                       on the same device as src.
 */
template <typename S, typename T>
void ExclusiveSum(const Array1<S> &src, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(src, *dest));
  int32_t src_dim = src.Dim();
  int32_t dest_dim = dest->Dim();
  K2_CHECK(dest_dim == src_dim || dest_dim == src_dim + 1);
  if (dest_dim == src_dim + 1) {
    const RegionPtr &region = src.GetRegion();
    ssize_t byte_offset = static_cast<ssize_t>(src.ByteOffset());
    K2_CHECK_GE(region->num_bytes - byte_offset, dest_dim * src.ElementSize());
  }
  ExclusiveSum(src.Context(), dest_dim, src.Data(), dest->Data());
}

/*  wrapper for the ExclusiveSum above.  Will satisfy
     ans[i] = sum_{j=0}^{i-1} src[j] for i > 0.
     ans[0] is always 0.
 */
template <typename T>
Array1<T> ExclusiveSum(const Array1<T> &src) {
  NVTX_RANGE(K2_FUNC);
  Array1<T> ans(src.Context(), src.Dim());
  ExclusiveSum(src, &ans);
  return ans;
}

/*
  Sets 'dest' to exclusive prefix sum of the result of dereferencing the
  elements of 'src'.
    @param [in] src    Source data, to be dereferenced and then summed.
    @param [out] dest  Destination data.  Must satisfy dest.Dim() == src.Dim()
                       or dest.Dim() == src.Dim() + 1, but in the latter case
                       we require that the memory region inside src be allocated
                       with at least one extra element. The extra element
                       (its value type is T*) should be assigned with a valid
                       address, because the exclusive-sum code may dereference
                       it even though it doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].
                       Must be on same device as src.
 */
template <typename T>
void ExclusiveSumDeref(Array1<const T *> &src, Array1<T> *dest);

/*
  Sets 'dest' to exclusive prefix sum of 'src', along a specified axis.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data; allowed to be the same as src.
                       For axis==1, for example, at exit it will satisfy
                       dest[i][j] == sum_{k=0}^{j-1} src[i][k].
                       On the axis being summed, must be either the same size
                       as src, or one greater than src (in such case, the
                       memory region inside src must be allocated with at least
                       one extra element as exclusive-sum may read from it even
                       though it doesn't affect the result);
                       Must have the same size with src on the other axis.
    @param [in] axis   Determines in what direction we sum, e.g. axis = 0 means
                       summation is over row axis (slower because we have to
                       transpose), axis = 1 means summation is over column axis.
 */
template <typename T>
void ExclusiveSum(const Array2<T> &src, Array2<T> *dest, int32_t axis);

//  wrapper for the ExclusiveSum above with axis = 1
template <typename T>
void ExclusiveSum(Array2<T> &src, Array2<T> *dest) {
  ExclusiveSum(src, dest, 1);
}

/*
  Append a list of Array1<T> to create a longer array.

  For now we can just use a simple loop; later there are lots of opportunities
  to optimize this, including multiple streams and using a single kernel making
  use of RaggedShape.
      @param [in] src_size  Number of arrays to append.  Must be > 0.
      @param [in] src     Array of pointers to arrays, of size `src_size`.
      @return       Returns the appended array
 */
template <typename T>
Array1<T> Append(int32_t src_size, const Array1<T> **src);

// Wrapper for Append() that has one fewer levels of indirection.
template <typename T>
Array1<T> Append(int32_t src_size, const Array1<T> *src);

/*
   This is a little like Append(), but with special treatment of the last
   elements (it's intended for use with row_splits vectors, which
   have a single "extra" last element).

   It appends the arrays with an offset.  Define:
        offset[i] = (sum of last element of src[j] for j < i).
        offset[i] = 0 for i = 0.
   This function appends the arrays, while leaving out the last element
   of all but the last of the arrays in `src`, and also adding the
   offsets mentioned above for each array.

      @param [in] src_size  Number of arrays to append
      @param [in] src     Array of pointers to arrays, of size `src_size`.
                          `src[i]` should be valid row_splits, i.e.
                          src[i]->Dim() >= 1 and the elements in it start
                          with 0 and are non-decreasing.
      @return       Returns the appended array

 */
Array1<int32_t> SpliceRowSplits(int32_t src_size, const Array1<int32_t> **src);

/*
  Get the reduction value from the array `src` with a binary operator `Op`,
  initialized with `default_value`. Will be used to implement
  `Max`, `And` and `Or` below.
      @param [in] src             Array to find the reduction of
      @param [in] default_value   Value to initialize the reduction with, and
                                  to use if src is empty.
      @param [out] dest           Output array, which must have dim == 1.
*/
template <typename T, typename Op>
void ApplyOpOnArray1(Array1<T> &src, T default_value, Array1<T> *dest);

/*
  Get the maximum value from the array `src`, or `default_value`, whichever is
  greater.
      @param [in] src             Array to find the max reduction of
      @param [in] default_value   Value to initialize the reduction with, and
                                  to use if src is empty.  Would typically be
                                  the most negative T possible.
      @param [out] dest           Output array, which must have dim == 1.
                                  Note: it is allowable for the output array
                                  to be an element of `src`.
 */
template <typename T>
void Max(Array1<T> &src, T default_value, Array1<T> *dest) {
  ApplyOpOnArray1<T, MaxOp<T>>(src, default_value, dest);
}

template <typename T>
T MaxValue(const Array1<T> &src) {
  return MaxValue(src.Context(), src.Dim(), src.Data());
}

/*
  Get the bitwise and reduction of the array `src`, using `default_value` (e.g.
  all ones) to initialize the reduction.

      @param [in] src             Array to find the bitwise & reduction of
      @param [in] default_value   Value to initialize the reduction with, and
                                  to use if src is empty.  Would typically be
                                  the most negative T possible.
      @param [out] dest           Output array, which must have dim == 1.
                                  Note: it is allowable for the output array
                                  to be an element of `src`.
 */
template <typename T>
void And(Array1<T> &src, T default_value, Array1<T> *dest) {
  ApplyOpOnArray1<T, BitAndOp<T>>(src, default_value, dest);
}

// As And, but bitwise Or. Note it is allowable for the output array to be an
// element of `src`.
template <typename T>
void Or(Array1<T> &src, T default_value, Array1<T> *dest) {
  ApplyOpOnArray1<T, BitOrOp<T>>(src, default_value, dest);
}

/*
  Call BinaryOp on each element in `src1` and `src2`, then write the result
  to the corresponding element in `dest`, i.e.
  for 0 <= i < dest_dim = src1_dim == src2_dim.
    dest[i] = BinaryOp(src1[i], src2[i])
  Noted `src1`, `src2` and `dest` must have the same Dim() and on the same
  device. It is allowable for &src1 == &src2 == dest.
*/
template <typename T, typename BinaryOp>
void ApplyBinaryOpOnArray1(Array1<T> &src1, Array1<T> &src2, Array1<T> *dest);

// Call PlusOp on `src1` and `src2` and save the result to `dest`,
// i.e. dest[i] = src1[i] + src2[i]
template <typename T>
void Plus(Array1<T> &src1, Array1<T> &src2, Array1<T> *dest) {
  ApplyBinaryOpOnArray1<T, PlusOp<T>>(src1, src2, dest);
}

// A wrapper function for Plus above, ans[i] = src1[i] + src2[i].
template <typename T>
Array1<T> Plus(Array1<T> &src1, Array1<T> &src2) {
  K2_CHECK_EQ(src1.Dim(), src2.Dim());
  Array1<T> ans(GetContext(src1, src2), src1.Dim());
  Plus(src1, src2, &ans);
  return ans;
}

// Same with `Plus`, but with `MinusOp`, i.e. dest[i] = src1[i] - src2[i].
template <typename T>
void Minus(Array1<T> &src1, Array1<T> &src2, Array1<T> *dest) {
  ApplyBinaryOpOnArray1<T, MinusOp<T>>(src1, src2, dest);
}

// A wrapper function for Minus above, ans[i] = src1[i] - src2[i].
template <typename T>
Array1<T> Minus(Array1<T> &src1, Array1<T> &src2) {
  K2_CHECK_EQ(src1.Dim(), src2.Dim());
  Array1<T> ans(GetContext(src1, src2), src1.Dim());
  Minus(src1, src2, &ans);
  return ans;
}

// Same with `Plus`, but with `TimesOp`, i.e. dest[i] = src1[i] * src2[i].
template <typename T>
void Times(Array1<T> &src1, Array1<T> &src2, Array1<T> *dest) {
  ApplyBinaryOpOnArray1<T, TimesOp<T>>(src1, src2, dest);
}

// A wrapper function for Times above, ans[i] = src1[i] * src2[i].
template <typename T>
Array1<T> Times(Array1<T> &src1, Array1<T> &src2) {
  K2_CHECK_EQ(src1.Dim(), src2.Dim());
  Array1<T> ans(GetContext(src1, src2), src1.Dim());
  Times(src1, src2, &ans);
  return ans;
}

/*
  Returns a random Array1, uniformly distributed betwen `min_value` and
  `max_value`.  CAUTION: for now, this will be randomly generated on CPU and
  then transferred to other devices if c is not a CPU context, so it will be
  slow if c is not a CPU context.
  Note T should be floating-pointer type or integral type.

    @param[in] c  Context for this array; note, this function will be slow
                  if this is not a CPU context
    @param [in] dim    Dimension, must be > 0
    @param [in] min_value  Minimum value allowed in the array
    @param [in] max_value  Maximum value allowed in the array;
                           require max_value >= min_value.
    @param [in] seed  The seed for the random generator. 0 to
                      use the default seed. Set it to a non-zero
                      value for reproducibility.
    @return    Returns the randomly generated array
 */
template <typename T>
Array1<T> RandUniformArray1(ContextPtr c, int32_t dim, T min_value, T max_value,
                            int32_t seed = 0);

/*
  Returns a random Array2, uniformly distributed betwen `min_value` and
  `max_value`.  CAUTION: for now, this will be randomly generated on CPU and
  then transferred to other devices if c is not a CPU context, so it will be
  slow if c is not a CPU context.
  Note: T should be floating-pointer type or integral type.

  The resulting array will be randomly contiguous or not (for better testing
  of bugs that depend on this property).

    @param[in] c  Context for this array; note, this function will be slow
                  if this is not a CPU context
    @param [in] dim0    Dimension 0 of answer, must be >= 0.
    @param [in] dim1    Dimension 1 of answer, must be >= 0.
    @param[in] min_value  Minimum value allowed in the array
    @param[in] max_value  Maximum value allowed in the array;
                           require max_value >= min_value.
    @return    Returns the randomly generated array
 */
template <typename T>
Array2<T> RandUniformArray2(ContextPtr c, int32_t dim0, int32_t dim1,
                            T min_value, T max_value);

/*
  Return a newly allocated Array1 whose values form a linear sequence,
   so ans[i] = first_value + i * inc.
*/
template <typename T = int32_t>
Array1<T> Range(ContextPtr c, int32_t dim, T first_value, T inc = 1);

template <typename T = int32_t>
Array1<T> Arange(ContextPtr c, T begin, T end, T inc = 1);

/*
  This is a convenience wrapper for the function of the same name in utils.h.
   @param [in] row_splits  Input row_splits vector, of dimension num_rows + 1
   @param [out] row_ids    row_ids vector to whose data we will write,
                           of dimension num_elems (which must equal
                           row_splits[num_rows].
 */
void RowSplitsToRowIds(const Array1<int32_t> &row_splits,
                       Array1<int32_t> *row_ids);

/*
  Given a vector of row_splits, return a vector of sizes.

     @param [in] row_splits   Row-splits array, should be non-decreasing,
                            and have Dim() >= 1
     @return   Returns array of sizes, satisfying `ans.Dim() ==
               row_splits.Dim() - 1` and
              `ans[i] = row_splits[i+1] - row_splits[i]`.
 */
Array1<int32_t> RowSplitsToSizes(const Array1<int32_t> &row_splits);

/*
  This is a convenience wrapper for the function of the same name in utils.h.
   @param [in] row_ids     Input row_ids vector, of dimension num_elems
   @param [out] row_splits  row_splits vector to whose data we will write,
                            of dimension num_rows + 1; we require
                           (but do not necessarily check!) that `row_ids` is
                           non-negative, non-decreasing, and all elements are
                           less than num_rows.
 */
void RowIdsToRowSplits(const Array1<int32_t> &row_ids,
                       Array1<int32_t> *row_splits);

/*
  Creates a merge-map from a vector of sizes.  A merge-map is something we
  sometimes create when we want to combine elements from a fixed number of
  sources.  If there are `num_srcs` sources, `merge_map[i] % num_srcs`
  gives the index of the source that the i'th element came from,
  and `merge_map[i] / num_srcs` is the index within the i'th source.

  This function is for when the sources are to be appended together.

     @param [in] c      Context which we want the result to use
     @param [in] sizes  Sizes of source arrays, which are to be appended.
                        Must have `sizes[i] > 0`.  `ans.Dim()` will equal
                        the sum of `sizes`.
     @return            Returns the merge map.


   NOTE: this function makes the most sense to call when you won't be
   needing the row-splits or row-ids that can be obtained from `sizes`.
   If you need those, it would be easier to create the merge_map directly
   from the row_ids and row_splits.

   EXAMPLE.  Suppose sizes is [ 3, 5, 1 ].  Then merge_map will be:
    [ 0, 3, 6, 1, 4, 7, 10, 13, 2 ].
 */
Array1<uint32_t> SizesToMergeMap(ContextPtr c,
                                 const std::vector<int32_t> &sizes);

/*
  Returns a new Array1<T> whose elements are this array's elements plus t.
 */
template <typename T>
Array1<T> Plus(const Array1<T> &src, T t);

template <typename T>
Array1<T> Minus(const Array1<T> &src, T t) {
  return Plus(src, -t);
}

/*
  Return true if all elements of the two arrays are equal.
  Will crash if the sizes differ.
*/
template <typename T>
bool Equal(const Array1<T> &a, const Array1<T> &b);

/*
  Return true if array `a` is monotonically increasing, i.e.
  a[i+1] >= a[i].
 */
template <typename T>
bool IsMonotonic(const Array1<T> &a);
/*
  Return true if array `a` is monotonically decreasing, i.e.
  a[i+1] >= a[i].
 */
template <typename T>
bool IsMonotonicDecreasing(const Array1<T> &a);

/*
  Generalized function inverse for an array viewed as a function which is
  monotonically decreasing.

     @param [in] src   Array which is monotonically decreasing (not necessarily
                      strictly) and whose elements are all positive, e.g.
                      [ 5 5 4 2 1 ]
     @return          Returns an array such that ans.Dim() == src[0],
                      such that ans[i] = min(j >= 0 : src[j] <= i).
                      We pretend values past the end of `src` are zeros.
                      In this case the result would be:
                      [ 5 4 3 3 2].
                      Notice ans[0] = 5 here, we get it because we pretend
                      `src` is [5 5 4 2 1 0]

    Note:             InvertMonotonicDecreasing(InvertMonotonicDecreasing(x))
                      will always equal x if x satisfies the preconditions.

   Implementation notes: allocate ans as zeros; run lambda {
   if (i + 1 == src_dim || src[i+1] < src[i])
       ans[src[i] - 1] = i + 1 } -> ans = [ 5 4 0 3 2]
   in this example; call MonotonicDecreasingUpperBound(ans, &ans).
 */
Array1<int32_t> InvertMonotonicDecreasing(const Array1<int32_t> &src);

/*
  Assuming `src` is a permutation of Range(0, src.Dim()), returns the inverse of
  that permutation, such that ans[src[i]] = i.  It is an error, and may cause a
  segfault or undefined results, if `src` was not a permutation of Range(0,
  src.Dim()).
 */
Array1<int32_t> InvertPermutation(const Array1<int32_t> &src);

/*
   Validate a row_ids vector; this just makes sure its elements are nonnegative
   and non-decreasing.

     @param [in] row_ids  row_ids to validate
     @param [in] temp     The user may supply a nonempty array on the same
                          device (or host) as `row_ids` that can be used
                          temporarily (just the first element is needed).
                          This saves an allocation.
     @return   Returns true if `row_ids` is a plausible row_ids vector.
*/
bool ValidateRowIds(const Array1<int32_t> &row_ids,
                    Array1<int32_t> *temp = nullptr);

/*
   Validate a row_splits vector; this just makes sure its elements are
   non-decreasing, its dimension is at least 1 and row_splits[0] == 0.

     @param [in] row_splits  row_splits to validate
     @param [in] temp     The user may supply a nonempty array on the same
                          device (or host) as `row_splits` that can be used
                          temporarily (just the first element is needed).
                          This saves an allocation.
     @return   Returns true if `row_splits` is a plausible row_splits vector.
*/
bool ValidateRowSplits(const Array1<int32_t> &row_splits,
                       Array1<int32_t> *temp = nullptr);

/*
  Jointly validate row_splits and row_ids vectors, making sure they are
  plausible and consistent with each other.

     @param [in] row_splits  row_splits to validate
     @param [in] row_ids     row_ids to validate
     @param [in] temp     The user may supply a nonempty array on the same
                          device (or host) as `row_splits` that can be used
                          temporarily (just the first element is needed).
                          This saves an allocation.
     @return   Returns true if the vectors are plausible and agree with each
  other.
*/
bool ValidateRowSplitsAndIds(const Array1<int32_t> &row_splits,
                             const Array1<int32_t> &row_ids,
                             Array1<int32_t> *temp = nullptr);

/*
  Compute a monotonically increasing lower bound on the array `src`,
  putting the result in `dest` (which may be the same array as `src`).

      @param [in] src  Source array (may be empty)
      @param [out] dest   Destination array; must be on the same device
                       as `src` and have the same dimension; may be the
                       same as `src`.

  At exit, `d = *dest` will be the largest sequence that is monotonically
  increasing (i.e. `d[i] <= d[i+1]`) and for which `d[i] <= src[i]`.  We
  compute this using an inclusive scan using a min operator on the
  reverse of the array `src` with output to the reverse of the
  array `dest`.
 */
template <typename S, typename T>
void MonotonicLowerBound(const Array1<S> &src, Array1<T> *dest);

/*
  Compute a monotonically decreasing upper bound on the array `src`,
  putting the result in `dest` (which may be the same array as `src`).

      @param [in] src  Source array (may be empty)
      @param [out] dest   Destination array (already allocated);
                       must be on the same device as `src` and have the same
                       dimension; may be the same as `src`.

  At exit, `d = *dest` will be the smallest sequence that is monotonically
  decreasing (i.e. `d[i] >= d[i+1]`) and for which `d[i] >= src[i]`.  We
  compute this using an inclusive scan using a max operator on the reverse
  of array `src` with output to the reverse of array `dest`.
 */
template <typename S, typename T>
void MonotonicDecreasingUpperBound(const Array1<S> &src, Array1<T> *dest);

/*
   Returns counts of numbers in the array
         @param [in] src   Source array whose elements are to be counted
         @param [in] n     Number of counts; we require `0 <= src[i] < n`.
         @return          Returns an array of size n, with ans[i] being
                          equal to the number of times i appeared in `src`.

   See also GetCountsPartitioned in ragged.h.
*/
Array1<int32_t> GetCounts(const Array1<int32_t> &src, int32_t n);

/* Returns counts of numbers in the array.

    @param [in] c  Context of `src_data`.
    @param [in] src_data  The source array.
    @param [in] src_dim   The dimension of the src array.
    @param [in] n         Number of counts; we require `0 <= src_data[i] < n`.

    See also GetCounts above.
 */
Array1<int32_t> GetCounts(ContextPtr c, const int32_t *src_data,
                          int32_t src_dim, int32_t n);

template <typename T>
Array2<T> ToContiguous(const Array2<T> &src);

/*
  Return true if all elements of the two arrays are equal.
  Will crash if the sizes differ.
*/
template <typename T>
bool Equal(const Array2<T> &a, const Array2<T> &b);

/*
  Index `src` with `indexes`, as in src[indexes].
     @param [in] src   Array whose elements are to be read
     @param [in] indexes  Indexes into `src`; must satisfy
                       `0 <= indexes[i] < src.Dim()` if
                       `allow_minus_one == false`,
                       else -1 is also allowed and the corresponding
                       output element will be zero.
     @return  Returns an `Array1<T>` of dimension indexes.Dim(),
               with `ans[i] = src[indexes[i]]` (or zero if
               `allow_minus_one == true` and `indexes[i] == -1`).
 */
template <typename T>
Array1<T> Index(const Array1<T> &src, const Array1<int32_t> &indexes,
                bool allow_minus_one);

/*
  Index src's rows with `indexes` which contains the row indexes.
     @param [in] src   Array whose elements are to be read
     @param [in] indexes  Indexes into `src`; must satisfy
                       `0 <= indexes[i] < src.Dim0()` if
                       `allow_minus_one == false`,
                       else -1 is also allowed and the corresponding
                       output element will be zero.
     @return  Returns an `Array2<T>` of shape (indexes.Dim(), src.Dim1()),
               with `ans[i,j] = src[indexes[i], j]` (or zero if
               `allow_minus_one == true` and `indexes[i] == -1`).
 */
template <typename T>
Array2<T> IndexRows(const Array2<T> &src, const Array1<int32_t> &indexes,
                    bool allow_minus_one);

/* Sort an array **in-place**.

   @param [inout]   array        The array to be sorted.
   @param [out]     index_map    If non-null, it will be set to
                          a new array that maps the index of the returned array
                          to the original unsorted array. That is, out[i] =
                          unsorted[index_map[i]] for i in [0, array->Dim()) if
                          `unsorted` was the value of `array` at input and `out`
                          is the value after the function call.
 */
template <typename T, typename Compare = LessThan<T>>
void Sort(Array1<T> *array, Array1<int32_t> *index_map = nullptr);

/*
  Assign elements from `src` to `dest`; they must have the same shape.  For now
  this only supports cross-device copy if the data is contiguous.
 */
template <typename T>
void Assign(Array2<T> &src, Array2<T> *dest);

/*
  Assign elements from `src` to `dest`; they must have the same Dim().
 */
template <typename S, typename T>
void Assign(Array1<S> &src, Array1<T> *dest);


/*
  Merge an array of Array1<T> with a `merge_map` which indicates which items
  to get from which positions (doesn't do any checking of the merge_map values!)

    @param [in] merge_map   Array which is required to have the same dimension
                            as the sum of src[i]->Dim().
                            If merge_map[i] == m, it indicates that the i'th
                            position in the answer should come from
                            element `m / num_srcs` within `*src[m % num_srcs]`.
    @param [in] num_srcs  Number of Array1's in the source array
    @param [in] src       Array of sources; total Dim() must equal
                          merge_map.Dim()
    @return               Returns array with elements combined from those in
                          `src`.

   CAUTION: may segfault if merge_map contains invalid values.
 */
template <typename T>
Array1<T> MergeWithMap(const Array1<uint32_t> &merge_map, int32_t num_srcs,
                       const Array1<T> **src);

}  // namespace k2

#define IS_IN_K2_CSRC_ARRAY_OPS_H_
#include "k2/csrc/array_ops_inl.h"
#undef IS_IN_K2_CSRC_ARRAY_OPS_H_

#endif  // K2_CSRC_ARRAY_OPS_H_
