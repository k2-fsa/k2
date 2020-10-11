/**
 * @brief
 * ragged_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_OPS_H_
#define K2_CSRC_RAGGED_OPS_H_

#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

namespace k2 {

/*
  Stack a list of RaggedShape to create a RaggedShape with one more axis.
  Similar to TF/PyTorch's Stack. The result will have Dim0 == src_size.
  All the source RaggedShapes must have the same NumAxes().

     @param [in] src_size  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes to be stacked
     @param [in] axis   The new axis whose dimension will equal src_size.
                        CAUTION: only axis == 0 and axis == 1 are supported
                        right now, and for the axis==1 case we have a
                        requirement that all the src[i]->Dim0() have the
                        same value.

     @return  The appended result.
        Viewing the source and result as the shapes of n-dimensional arrays,
        if axis==0 we will have:
            result[i,j,k,l] = (*src[i])[j,k,l]
        and if axis==1 we will have:
            result[i,j,k,l] = (*src[j])[i,k,l]
        (although of course no such operator actually exists at the C++ level,
        and these are just the shapes of arrays..).
        See also the version of Stack for class Ragged.
 */
RaggedShape Stack(int32_t axis, int32_t src_size, const RaggedShape **src);

/*
  Insert a new axis at position `axis`, with 0 <= axis <= src.NumAxes(), for
  which the only allowed index will be 0 (which is another way of saying: all
  list sizes on that axis will be 1).

     @param [in] src   Source shape.  Row-ids and Row-splits of answer will
                      share memory with those in `src` (although it goes
                      without saying).
     @param [in] axis  Axis to insert.

  Note: you probably shouldn't be using this very often; if you are using
  this, you may be thinking in a PyTorch-y way but you should be relying on
  things like Eval() with custom lambdas more.  Read algorithms like in
  compose.cc to understand why.  Also: axis==0 is probably the only really
  useful case. See more useful notes in comments in the implementation.
 */
RaggedShape Unsqueeze(const RaggedShape &src, int32_t axis);

/* Remove an axis; if it is not the last axis, this is done by appending lists
   (effectively the axis is combined with the following axis).  If it is the
   last axis it is just removed and the number of elements will be affected.

          @param [in] src Ragged shape to remove axis of (`src` is conceptually
                      unchanged by this operation but non-const because row-ids
                      may need to be generated).
                      We require src.NumAxes() > 2, since the minimum number of
                      axes for a RaggedShape is 2.
          @param [in] axis  Axis to remove; must satisfy
                            0 <= axis < src.NumAxes()
          @return      Returns the modified shape with one fewer axis; will
                       satisfy ans.TotSize(axis) == src.TotSize(axis + 1).
                       if axis < src.NumAxes() - 1.
*/
RaggedShape RemoveAxis(RaggedShape &src, int32_t axis);

/*
  Returns a CPU array of shape (src[0]->NumAxes() + 1) by (num_srcs + 1), where
  each row is the exclusive-sum of the TotSize() of the respective sources,
  on the previous axis (or 1 for axis 0).  Specifically: it's the same
  as setting ans(i,j) to (i == 0 ? 1 : src[j]->TotSize(i-1)), and then
  doing an exclusive-sum on each row of i.

     @param [in] num_srcs  The number of `RaggedShape`s in `src`
     @param [in] src       The shapes whose sizes we want. Must all have the
                           same NumAxes().
     @return   Returns a freshly allocated CPU Array2<int32_t> of dimension
               (src[0]->NumAxes() + 1) by (num_srcs + 1), where each
               row is the exclusive-sum of the TotSize() of the respective
               sources, on that axis. Its last column contains the totals.

 */
Array2<int32_t> GetOffsets(int32_t num_srcs, RaggedShape **src);

/*
  Transpose a RaggedShape: namely, axes 0 and 1.  Requires that the sizes
  of lists on axis 1 all be the same, i.e. that src.RowSplits(1) have
  equally spaced elements.

     @param [in] src   Shape to be transposed.  We require src.NumAxes() > 2.
                       (this is because the implementation would be slightly
                       different, and because if you had a ragged array
                       with 2 axes and a regular shape, you should really
                       be using an Array2 or Tensor).
     @return           Returns the transposed shape, with axes 0 and 1
                       swapped.  Will satisfy
                       ans.Dim0() == src.TotSize(1) / src.Dim0()
                       and ans[i,j,k] = src[j,i,k] (Note, this is not actual C++
                       code, it represents a conceptual indexing operator).
 */
RaggedShape Transpose(RaggedShape &src);

/*
   Append a list of RaggedShape to form a single RaggedShape

      @param [in] axis     Axis to append them on. Previous axes must
                           have the same shape! CAUTION: currently
                           we only support axis == 0.
      @param [in] num_srcs Number of source shapes to append
      @param [in] src      Array of sources to append
      @return      Returns the appended RaggedShape.
*/
RaggedShape Append(int32_t axis, int32_t num_srcs, RaggedShape **src);

/*
    Gets an array of pointers to the row_splits of `src`, on the same
    device as `src`.
       @param [in] src  Source RaggedShape
       @return        Returns an array of size src.NumAxes() - 1 containing
                      pointers to the starts of the row_splits vectors.
*/
Array1<int32_t *> GetRowSplitsPtr(RaggedShape &src);

/*
  Renumber(/Reorder) axis 0 of a ragged shape
     @param [in] src      Shape to renumber
     @param [in] new2old  Mapping from new to old numbering of array, of
                          length src.Dim0(), on the same device as `src`;
                          must contain the numbers
                          0 through src.Dim0() - 1 in some order.
     @return              Returns the renumbered shape.  Will satisfy:
                          ret[i,j,k] = src[new2old[i],j,k].  (Note, this is
                          not actual C++ code, it represents a conceptual
                          indexing operator).
*/
RaggedShape Renumber(RaggedShape &src, const Array1<int32_t> &new2old);

/*
  Return a random RaggedShape, with a CPU context.  Intended for testing.

     @param [in] set_row_ids    If false, row_ids in the returned RaggedShape
                                will be empty; If true, row_ids would be filled.
     @param [in] min_num_axes   Minimum number of axes (must be at least 2)
     @param [in] max_num_axes   Maximum number of axes, must be
                                >= min_num_axes
     @param [in] min_num_elements  Minimum number of elements; must be at
                                   least 0.
     @param [in] max_num_elements  Maximum number of elements; must be
                                    >= min_num_elements.
 */
RaggedShape RandomRaggedShape(bool set_row_ids = false,
                              int32_t min_num_axes = 2,
                              int32_t max_num_axes = 4,
                              int32_t min_num_elements = 0,
                              int32_t max_num_elements = 2000);

/*
  Return ragged shape with only a subset of the bottom-level elements
  kept.  Require renumbering.NumOldElems() == src.TotSize(src.NumAxes()-1).
  Note: all dimensions and tot-sizes preceding that will remain the
  same, which might give rise to empty lists.
 */
RaggedShape SubsampleRaggedShape(RaggedShape &src, Renumbering &renumbering);

/*
  Stack a list of Ragged arrays to create a Ragged array with one more axis.
  Similar to TF/PyTorch's Stack.  The result will have Dim0 == num_srcs.  All
  the source Ragged arrays' shapes must have the same NumAxes().

     @param [in] axis   The new axis whose dimension will equal num_srcs.
                        CAUTION: only axis == 0 is supported right now.
     @param [in] num_srcs  The number of `RaggedShape`s in `src`
     @param [in] src       The shapes to be stacked

     @return  The appended result.
       Assuming as an example that the input had 3 axes:
       if axis==0, the result would have:
          result[i,j,k,l] = (*src[i])[j,k,l]
       and if axis==1 we would have:
          result[i,j,k,l] = (*src[j])[i,k,l]
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, const Ragged<T> **src);

/*
  This version of Stack() has one fewer levels of pointer indirection,
  it is just a wrapper for the version above.
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, const Ragged<T> *src);

/*
  Construct a RaggedShape with 2 axes.
     @param [in] row_splits   row_splits, or NULL (at least one of this and
                     row_ids must be non-NULL).  Note: the dimension of row
                     splits must equal the number of rows plus one;
                     row_splits[0] must be zero and the array must be
                     non-decreasing; and the last element of row_splits
                     is the total number of elements in the ragged matrix.
     @param [in] row_ids      row_ids, or NULL (at least one of this and
                     row_splits must be non-NULL). The array must be
                     non-negative and non-decreasing; If both row_ids
                     and row_splits are supplied, we require
                     row_splits[row_ids[i]] <= i < row_splits[row_ids[i]+1]
                     and
                     row_splits[row_splits.Dim() - 1] == row_ids.Dim().
                     If row_splits is NULL, then we suppose the number of rows
                     equals row_ids[row_ids.Dim() - 1] + 1;
     @param [in] cached_tot_size   Total number of elements in the ragged
                     matrix, or -1 if the user does not wish to supply
                     it now. (If >= 0, must equal
                     `(row_splits)[row_splits.Dim()-1]` if row_splits
                     is non-NULL, or row_ids->Dim() if row_ids is non-NULL.
*/
RaggedShape RaggedShape2(Array1<int32_t> *row_splits, Array1<int32_t> *row_ids,
                         int32_t cached_tot_size);

/*
  This is a general method of creating higher-dimensional ragged shapes.
     @param [in] a   RaggedShape describing the top level (first indexes)
                     of the returned shape
     @param [in] b   RaggedShape describing the bottom level (later
                     indexes) of the returned shape.  We require
                     a.NumElements() == b.Dim0().
     @return     Returns the combined ragged shape.  Its number of axes
                 will be equal to a.NumAxes() + b.NumAxes() - 1 (the last
                 axis of a and the first axis of b are combined).
 */
RaggedShape ComposeRaggedShapes(const RaggedShape &a, const RaggedShape &b);

/*
  Construct a RaggedShape with 3 axes.  For N=1 and 2 respectively:
  either row_splitsN or row_idsN or both must be non-NULL.
  If cached_tot_sizeN is not -1, it must equal the total size on
  that axis which will equal the last element of row_splitsN (if
  provided) and must equal the row_idsN.Dim(), if provided. See
  documentation above for RagggedShape2 for details.

  We also require that (supposing both row_splitsN and row_idsN are non-NULL):
  row_splits1[row_splits1.Dim() - 1] == row_ids1.Dim()
     == (row_splits2.Dim() - 1)
     >= (row_ids2[row_ids2.Dim() - 1] + 1)
*/
RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2);

/*
  Returns a RaggedShape with 2 axes, with Dim0() == 1 and
  TotSize(1) = num_elems.
 */
RaggedShape TrivialShape(ContextPtr &c, int32_t num_elems);

/*
  Allocates an *invalid* RaggedShape given the TotSize() values for each axis.
  This allocates space for all the row_ids and row_splits, but does not write
  any data to them.  View this as an internal function, because such a
  RaggedShape would not be usable directly.

     @param [in] c          The context with which we'll allocate memory for
                            row_splits and row_ids.
     @param [in] num_axes   Number of axes of ragged shape desired; must be at
                            least 2.
     @param [in] tot_sizes  Total size on each axis. `tot_sizes[0]` through
                            `tot_sizes[num_axes]` must be valid which also
                            implies the number of rows on axis i is
                            tot_sizes[i-1].

     @return           The returned RaggedShape satisfies
                       ans.TotSize(axis) == tot_sizes[axis], and has all
                       its row_splits and row_ids allocated but not
                       filled in, and its cached_tot_size elements
                       set.
 */
RaggedShape RaggedShapeFromTotSizes(ContextPtr &c, int32_t num_axes,
                                    int32_t *tot_sizes);

inline RaggedShape RaggedShapeFromTotSizes(ContextPtr &c,
                                           std::vector<int32_t> &tot_sizes) {
  return RaggedShapeFromTotSizes(c, static_cast<int32_t>(tot_sizes.size()),
                                 tot_sizes.data());
}

/*
  Creates a random ragged array (with a CPU context!).  Note: you may want to
  explicitly use the template arg, e.g. invoke as RandomRagged<int32_t>(...).

     @param [in] min_value  Minimum element value allowed
     @param [in] max_value  Maximum element value allowed
     @param [in] min_num_axes   Minimum number of axes (must be at least 2)
     @param [in] max_num_axes   Maximum number of axes, must
                                be >= min_num_axes
     @param [in] min_num_elements  Minimum number of elements;
                                   must be at least 0.
     @param [in] max_num_elements  Maximum number of elements;
                                   must be >= min_num_elements.

     @return  Returns a random ragged array, with a CPU context.
 */
template <typename T>
Ragged<T> RandomRagged(T min_value = static_cast<T>(0),
                       T max_value = static_cast<T>(100),
                       int32_t min_num_axes = 2, int32_t max_num_axes = 4,
                       int32_t min_num_elements = 0,
                       int32_t max_num_elements = 2000);

/*
  Sort a ragged array in-place.

     @param [inout]   The input array to be sorted.
                      CAUTION: it is sorted in-place.
     @param [out]     The indexes mapping from the sorted
                      array to the input array. If not NULL,
                      the caller has to pre-allocate memory for
                      it on the same device as `src`.
 */
template <typename T, typename Op = LessThan<T>>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order = nullptr);

// Caution: you need the template argument to invoke this,
// e.g. RaggedFromTotSizes<int32_t>(c, { 10, 100 })
// Also caution: the Ragged object this returns is *not valid* in the sense
// that ans.shape.Check() would fail because the row_splits and row_ids
// have not been set up.  The caller is expected to do that.
template <typename T>
inline Ragged<T> RaggedFromTotSizes(ContextPtr &c,
                                    std::vector<int32_t> &tot_sizes) {
  return Ragged<T>(RaggedShapeFromTotSizes(c, tot_sizes),
                   Array1<T>(c, tot_sizes.back()));
}

/*
  Transpose a ragged tensor as if it were the index information of a CSR-format
  sparse matrix (but with possibly repeated elements!).  This is easiest to
  explain if we assume `src` has 2 axes.  We view `src` as a list of nonzero
  elements of a matrix, indexed first by row, and containing column-indexes
  (but possibly repeated column indexes, which violates the assumptions of
  the cusparse library).  This function returns an array of dimension
  src.values.Dim() which tells us the order in which these elements would
  appear if sorted by column.  (TODO: we can decide later whether to require
  sorting secondarily by row).  So `src.values[ans]` will be in sorted
  order at exit, and `ans` will contain all numbers from 0 to `src.values.Dim()
  - 1`.

  If `src` has more than 2 axes, the earlier-numbered axes do not affect
  the result, except for an efficiency modification: we require that the
  required reordering does not cross the boundaries fixed by the earlier
  axes, so we can if necessary implement this by sorting sub-lists instead
  of sorting a single long list.  That is: for i < j,  if idx0(i) < idx0(j)
  then we require src.values(i) < src.values(j).

  TODO(dan): we may at some point make, as an optional output, row-splits and/or
  row-ids of the rearranged matrix.

  This problem has some relationship to the cusparse library, specifically the
  csr2csc functions
  https://docs.nvidia.com/cuda/cusparse/index.html#csr2cscEx2). However I'm not
  sure what it does when there are repeated elements.  It might be easiest to
  implement it via sorting for now.


     @param [in] src  Input tensor, see above.
     @param [in] num_cols  Number of columns in matrix to be transposed;
                  we require 0 <= src.values[i] < num_cols.
*/
Array1<int32_t> GetTransposeReordering(Ragged<int32_t> &src, int32_t num_cols);

/*
  This function is like GetCounts() that is declared in array_ops.h,
  but works on a partitioned problem (this should be faster).

   @param [in] src  A ragged array with src.NumAxes() == 2
   @param [in] ans_ragged_shape  The ragged shape of the answer; must also
                    have NumAxes() == 2 and ans_ragged_shape.Dim0() ==
                    src.Dim0().  Elements e of the i'th row of `src`
                    must satisfy:
                      row_splits[i] <= e < row_splits[i+1].
                   where row_splits is ans_ragged_shape.RowSplits(1).
   @return         Returns a ragged array with shape == ans_ragged_shape,
                   and elements corresponding to the counts in `src`.
                   Specifically, ans[i,j] is the number of times number j
                   appeared in row i of `src`.

  Implementation notes: we'll probably use cub's
  DeviceSegmentedRadixSort::SortKeys(), then call RowIdsToRowSplits() on
  the result (with num_rows == ans_ragged_shape.NumElements()), then
  for each i, let ans.values[i] = row_splits[i+1]-row_splits[i] (where
  row_splits is the output of RowIdsToRowSplits() we just called).

  This could actually be implemented using the GetCounts() of array_ops.h,
  ignoring the structure; the structure should help the speed though.
  This equivalence should be useful for testing.
*/
Ragged<int32_t> GetCountsPartitioned(Ragged<int32_t> &src,
                                     RaggedShape &ans_ragged_shape);

}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_OPS_H_
#include "k2/csrc/ragged_ops_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_OPS_H_

#endif  // K2_CSRC_RAGGED_OPS_H_
