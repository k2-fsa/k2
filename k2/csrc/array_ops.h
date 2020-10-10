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

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
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

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].
                       Must be on same device as src.
 */
template <typename S, typename T>
void ExclusiveSum(Array1<S> &src, Array1<T> *dest) {
  K2_CHECK(IsCompatible(src, *dest));
  int32_t src_dim = src.Dim();
  int32_t dest_dim = dest->Dim();
  K2_CHECK(dest_dim == src_dim || dest_dim == src_dim + 1);
  if (dest_dim == src_dim + 1) {
    const RegionPtr &region = src.GetRegion();
    int32_t byte_offset = src.ByteOffset();
    K2_CHECK_GE(region->num_bytes - byte_offset, dest_dim * src.ElementSize());
  }
  ExclusiveSum(src.Context(), dest_dim, src.Data(), dest->Data());
}

/*  wrapper for the ExclusiveSum above.  Will satisfy
     ans[i] = sum_{j=0}^{i-1} src[j].
 */
template <typename T>
Array1<T> ExclusiveSum(Array1<T> &src) {
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
void ExclusiveSum(Array2<T> &src, Array2<T> *dest, int32_t axis);

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
  Output to an array `dst` the result of reducing each sub-list along
  the last axis of `src` with a binary operator `Op`, will be called to
  implement `MaxPerSublist`, `AndPerSublist` and `OrPerSublist`

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] default_value  Value to initialize the reduction with;
     @param [out] dst           Array to which the reduction values will be
                                written. Must satisfy
                                dst->Dim() == rows along the last axis in src,
                                i.e. src.RowSplits(src.NumAxes() - 1).Dim() - 1.
*/

template <typename T, typename Op>
void ApplyOpPerSublist(Ragged<T> &src, T default_value, Array1<T> *dst);
/*
  Output to an array `max_values` the maximum of each sub-list along the last
  axis of `src` i.e. the max taken over the last axis), or `default_value`,
  whichever was larger.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] default_value  Value to use for maximum operation as a default
                                so max is taken over this and the elements
                                of sub-lists in `src`.
     @param [out] max_values    Array to which the maximum values will be
                                written. Must satisfy
                                max_values->Dim() == rows along the last axis in
                                src, i.e.
                                src.RowSplits(src.NumAxes() - 1).Dim() - 1.
 */
template <typename T>
void MaxPerSublist(Ragged<T> &src, T default_value, Array1<T> *max_values) {
  ApplyOpPerSublist<T, MaxOp<T>>(src, default_value, max_values);
}

/*
  Output to an array `and_values` the result of reducing each sub-list along
  the last axis of `src` with operator &, i.e. bit-wise and.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] default_value  Value to initialize the reduction with; should
                                probably be all-ones.
     @param [out] and_values    Array to which the bitwise-and values will be
                                written. Must satisfy
                                and_values->Dim() == src.TotSize(src.NumAxes() -
  2), i.e. the total size on the second-to-last axis of `src`.
*/
template <typename T>
void AndPerSublist(Ragged<T> &src, T default_value, Array1<T> *and_values) {
  ApplyOpPerSublist<T, BitAndOp<T>>(src, default_value, and_values);
}

// bitwise or
template <typename T>
void OrPerSublist(Ragged<T> &src, T default_value, Array1<T> *or_values) {
  ApplyOpPerSublist<T, BitOrOp<T>>(src, default_value, or_values);
}

/*
  Sort each sub-list in `src`, with operator `<`, and output the order to
  `order`. CAUTION: don't rely on this being a stable sort for now. Will
  eventually make the operator customizable, in which case this would become a
  wrapper.

      @param [in] src   Ragged array with 2 axes.
      @param [out] order   List of indexes that we'll use to give `src`
                      a sorted order; will be resized if its size is
                      not src.values.Dim().  If you do
                        src.values = src.values[*order]
                      then src.values will be sorted.
 */
template <typename T, typename Op>
void SortSublists(Ragged<T> &src, Array1<int32_t> *order);

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
  Returns a random Array1, uniformly distributed betwen `min_value` and
  `max_value`.  CAUTION: for now, this will be randomly generated on CPU and
  then transferred to other devices if c is not a CPU context, so it will be
  slow if c is not a CPU context.
  Note T should be floating-pointer type or integral type.

    @param[in] c  Context for this array; note, this function will be slow
                  if this is not a CPU context
    @param [in] dim    Dimension
    @param[in] min_value  Minimum value allowed in the array
    @param[in] max_value  Maximum value allowed in the array;
                           require max_value >= min_value.
    @return    Returns the randomly generated array
 */
template <typename T>
Array1<T> RandUniformArray1(ContextPtr &c, int32_t dim, T min_value,
                            T max_value);

/*
  Return a newly allocated Array1 whose values form a linear sequence,
   so ans[i] = first_value + i * inc.
*/
template <typename T>
Array1<T> Range(ContextPtr &c, int32_t dim, T first_value, T inc = 1);

/*
  This is a convenience wrapper for the function of the same name in utils.h.
   @param [in] row_splits  Input row_splits vector, of dimension num_rows + 1
   @param [out] row_ids    row_ids vector to whose data we will write,
                           of dimension num_elems (which must equal
                           row_splits[num_rows].
 */
void RowSplitsToRowIds(const Array1<int32_t> &row_splits,
                       Array1<int32_t> &row_ids);

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
                       Array1<int32_t> &row_splits);

/*
  Return true if all elements of the two arrays are equal
*/
template <typename T>
bool Equal(const Array1<T> &a, const Array1<T> &b);

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
   Returns counts of numbers in the array
         @param [in] src   Source array whose elements are to be counted
         @param [in] n     Number of counts; we require `0 <= src[i] < n`.
         @return          Returns an array of size n, with ans[i] being
                          equal to the number of times i appeared in `src`.

   See also GetCountsPartitioned in ragged.h.
*/
Array1<int32_t> GetCounts(const Array1<int32_t> &src, int32_t n);

template <typename T>
Array2<T> ToContiguous(const Array2<T> &src);

}  // namespace k2

#define IS_IN_K2_CSRC_ARRAY_OPS_H_
#include "k2/csrc/array_ops_inl.h"
#undef IS_IN_K2_CSRC_ARRAY_OPS_H_

#endif  // K2_CSRC_ARRAY_OPS_H_
