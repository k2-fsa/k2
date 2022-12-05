/**
 * Copyright      2020-2021  Xiaomi Corporation (authors: Daniel Povey
 *                                                        Haowen Qiu
 *                                                        Fangjun Kuang)
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

#ifndef K2_CSRC_RAGGED_OPS_H_
#define K2_CSRC_RAGGED_OPS_H_

#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

namespace k2 {

/*
  Output to an array `dst` the result of reducing each sub-list along
  the last axis of `src` with a binary operator `Op`, will be called to
  implement `MaxPerSublist`, `AndPerSublist` and `OrPerSublist`

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] initial_value  Value to initialize the reduction with;
     @param [out] dst           Array to which the reduction values will be
                                written. Must satisfy
                                dst->Dim() == src.TotSize(src.NumAxes()
                                - 2).  i.e. num-rows of last axis of `src`.
*/

template <typename T, typename Op>
void SegmentedReduce(const Ragged<T> &src, T initial_value, Array1<T> *dst);

/*
  Output to an array `max_values` the maximum of each sub-list along the last
  axis of `src` i.e. the max taken over the last axis), or `initial_value`,
  whichever was larger.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] initial_value  Value to use for maximum operation as a default
                                so max is taken over this and the elements
                                of sub-lists in `src`.
     @param [out] max_values    Array to which the maximum values will be
                                written. Must satisfy
                                max_values->Dim() == src.TotSize(src.NumAxes()
                                - 2).  i.e. num-rows of last axis of `src`.
 */
template <typename T>
void MaxPerSublist(const Ragged<T> &src, T initial_value,
                   Array1<T> *max_values) {
  SegmentedReduce<T, MaxOp<T>>(src, initial_value, max_values);
}

// Same with `MaxPerSubList`, but output the `min_value` in each sub-list.
template <typename T>
void MinPerSublist(const Ragged<T> &src, T initial_value,
                   Array1<T> *min_values) {
  SegmentedReduce<T, MinOp<T>>(src, initial_value, min_values);
}

// Same with `MaxPerSubList`, but output the sum of values in each sub-list.
template <typename T>
void SumPerSublist(const Ragged<T> &src, T initial_value,
                   Array1<T> *sum_values) {
  SegmentedReduce<T, PlusOp<T>>(src, initial_value, sum_values);
}

// Same with `MaxPerSubList`, but with Op as `LogAdd`.
template <typename T>
void LogSumPerSublist(const Ragged<T> &src, T initial_value,
                      Array1<T> *dst_values) {
  K2_STATIC_ASSERT(
      (std::is_same<float, T>::value || std::is_same<double, T>::value));
  SegmentedReduce<T, LogAdd<T>>(src, initial_value, dst_values);
}

/*
  Output to an array `argmax` the arg-max within each sub-list along the
  last axis of `src` i.e. the index
  within `src.values` of the maximum element of that sub-list, or -1
  if the sub-list was empty or all values in the sub-list are less than
  `initial_value`.

     @param [in] src        Input ragged array; must have src.NumAxes() >= 2.
                            src.values is allowed to be empty.
     @param [in] initial_value  Returned index is -1 for a sub-list if its maximum
                                is less than `initial_value`.
     @param [out] argmax    Array to which the argmax indexes will be written.
                            argmax->Dim() == src.TotSize(src.NumAxes() - 2),
                            i.e. num-rows of last axis of `src`.
 */
template <typename T>
void ArgMaxPerSublist(const Ragged<T> &src, T initial_value,
                      Array1<int32_t> *argmax);

/* Normalize per sublist.

   @param [in] src  The source ragged tensor. The normalization
                    is done on the last axis.
   @param [in] use_log  Indicate which kind of normalization to use. See
                        below for detailed description.

   If use_log is true, the normalization per sublist is done as follows:

      1. Compute the log sum using LogSumPerSublist
      2. Subtract the log sum from the sublist
      3. Return the resulting sublist

   If use_log is false, the normalization per sublist is done as follows:

      1. Compute the sum using SumPerSublist
      2. Divide the sublist by the above sum
      3. Return the resulting sublist

   @return The normalized ragged tensor.
 */
template <typename T>
Ragged<T> NormalizePerSublist(const Ragged<T> &src, bool use_log);

/* Adds value, scaled by alpha, to src per sublist.

   It implements:
     src[...][i][j] = src[...][i][j] + alpha * value[i]

   @param [in] src  The source ragged tensor. The add is done on the last axis.
   @param [in] value  The value to be added to the src, whose dimension MUST
                      equal the number of sublists along src's last dimension.
   @param [in] alpha  The number used to scaled value before adding to src.

   @return The added ragged tensor.
 */
template <typename T>
Ragged<T> AddPerSublist(Ragged<T> &src, Array1<T> &value, T alpha);

/*
  Output to an array `and_values` the result of reducing each sub-list along
  the last axis of `src` with operator &, i.e. bit-wise and.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                >= 2. src.values is allowed to be empty.
     @param [in] initial_value  Value to initialize the reduction with; should
                                probably be all-ones.
     @param [out] and_values    Array to which the bitwise-and values will be
                                written. Must satisfy
                                and_values->Dim() == src.TotSize(src.NumAxes() -
                                2), i.e. num-rows of the last axis of `src`.
*/
template <typename T>
void AndPerSublist(const Ragged<T> &src, T initial_value,
                   Array1<T> *and_values) {
  SegmentedReduce<T, BitAndOp<T>>(src, initial_value, and_values);
}

// bitwise or
template <typename T>
void OrPerSublist(const Ragged<T> &src, T initial_value, Array1<T> *or_values) {
  SegmentedReduce<T, BitOrOp<T>>(src, initial_value, or_values);
}

/*
  Stack a list of RaggedShape to create a RaggedShape with one more axis.
  Similar to TF/PyTorch's Stack. The dimension of the new axis equals src_size.
  All the source RaggedShapes must have the same NumAxes().

     @param [in] axis   The new axis whose dimension will equal src_size.
                        Dimensions/shapes of all previous axes must be
                        identical.
     @param [in] src_size  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes to be stacked
     @param [out] merge_map  If not nullptr, will be set to the merge-map
                        that tells us for each 0 <= i < ans.NumElements(),
                        which element of `src` it came from (available
                        as `merge_map[i] % num_srcs`) and its element-index
                        within `src[i]` (available as `merge_map[i] / num_srcs`.


     @return  The appended result.
        Viewing the source and result as the shapes of n-dimensional arrays,
        if axis==0 we will have:
            result[i,j,k,l] = (*src[i])[j,k,l]
        and if axis==1 we will have:
            result[i,j,k,l] = (*src[j])[i,k,l]
        (although of course no such operator actually exists at the C++ level,
        and these are just the shapes of arrays).
        See also the version of Stack for class Ragged.
 */
RaggedShape Stack(int32_t axis, int32_t src_size, RaggedShape **src,
                  Array1<uint32_t> *merge_map = nullptr);


/*
  Unstack a RaggedShape to a list of RaggedShapes, all the output RaggedShapes
  have one axis less.
  This function tries to do the opposite of Stack(), i.e. to generate an array
  out such that `Equal(src, Stack(axis, out->size(), out->data()))`. But notes
  that `Stack` needs a pointer of RaggedShape pointer, Unstack produces only a
  pointer of RaggedShape, you should do some conversion before using Stack.

    @param [in] src  The shape to unstack.
    @param [in] axis  The axis to be removed, all the elements of this axis will
                      be rearranged into output RaggedShapes.
    @param [in] pad_right  Before unstack, we will (conceptually) pad the
                  sublists along axis `axis` to the same size with empty lists
                  `pad_right` tells where to put the padding empty lists, see
                  the example for more details.
                  Note, `pad_right` makes no difference when `axis == 0` or
                  `axis == src.NumAxes() - 1`.
    @param [out] out  The container where the output RaggedShapes would write
                      to. MUST NOT be a nullptr, will be reallocated.
    @param [out] split_map  If not nullptr will store the element-index within
                   `src` telling where the elements of each split RaggedShapes
                   come from. It has the same size of `out`, see notes below
                   for the dimension of it. For Array1 in each of the
                   `split_map`, It satisfies
                   `split_map[i].Dim() == out[i].NumElements()`, and
                   `0 <= split_map[i][j] < src.NumElements()`.
                   `split_map` will be reallocated by this function.

  Caution: If `src.NumAxes() == 2`, the output shapes will only have one
           dimension, to make it a RaggedShape, we will add a TrivialShape on
           each of the output tensors.

  Note: The output RaggedShape may contain empty lists on axis `axis`, you can
        remove them by RemoveEmptyLists if needed.

  Note: The number of output RaggedShape is determined by the size of sublist
        with max number of elements along axis `axis`, for `axis == 0`, it has
        only one sublist along `axis == 0`(i.e. the src itself), so the number
        of output RaggedShape will be equal to `src.Dim0()`.

  A small example of unstacking a 3 axes RaggedShape (with pad_right=true):

    src: [ [ [ x x ] [ x ] ] [ [ x ] ] ]
    unstack on axis 0:
    src.Dim0() == 2, will produce 2 RaggedShape.

    out[0] : [ [ x x ] [ x ] ]   split_map[0] : [0, 1, 2]
    out[1] : [ [ x ] ]           split_map[1] : [3]

    unstack on axis 1:
    two sublists along axis 1, the sizes are [2, 1], will produce 2 RaggedShape

    think about that we first pad src to [ [ [ x x ] [ x ] ] [ [ x ] [ ] ] ]
    then select elements along axis 1 into separate ragged shapes

    out[0] : [ [ x x ] [ x ] ]   split_map[0] : [0, 1, 3]
    out[1] : [ [ x ] [ ] ]       split_map[1] : [2]

    unstack on axis 2:
    three sublists along axis 2, the sizes are [2, 1, 1], will produce 2
    RaggedShape.

    out[0] : [ [ x x ] [ x ] ]   split_map[0] : [0, 2, 3]
    out[1] : [ [ x ] [ ] ]       split_map[1] : [1]

  for pad_right equals to false:

    src: [ [ [ x x ] [ x ] ] [ [ x ] ] ]

    unstack on axis 1:

    think about that we first pad src to [ [ [ x x ] [ x ] ] [ [ ] [ x ] ] ]
    then select elements along axis 1 into separate ragged shapes

    out[0] : [ [ x x ] [ ] ]       split_map[0] : [0, 1]
    out[1] : [ [ x ] [ x ] ]       split_map[1] : [2, 3]
 */
void Unstack(RaggedShape &src, int32_t axis, bool pad_right,
             std::vector<RaggedShape> *out,
             std::vector<Array1<int32_t>> *split_map = nullptr);

/*
 * The same as above, except that it uses `pad_right=true`.
 */
void Unstack(RaggedShape &src, int32_t axis, std::vector<RaggedShape> *out,
             std::vector<Array1<int32_t>> *split_map = nullptr);

/*
  Return a modified version of `src` in which all sub-lists on the last axis of
  the tenor have size modified by `size_delta`.  `size_delta` may have either
  sign.  If for a sub-list of size `cur_size`, `cur_size - size_delta < 0`, that
  sub-list's size will be changed to 0 but the sub-list will be kept.


     @param [in] src  Source tensor; must have NumAxes() >= 2, i.e. be valid.
                      Only the last axis, i.e. the last RowSplits/RowIds(),
                      will be affected by this.
     @param [in] size_delta  Amount by which to change the size of sub-lists.
                      May be either sign; if negative, we'll reduce the
                      sub-list size by this amount, possibly leaving empty
                      sub-lists (but it's an error if this would reduce any
                      sub-list size below zero).
     @return          Returns the modified RaggedShape.  The RowSplits()
                      and RowIds() of its last axis will not be shared
                      with `src`.

  Example: ChangeSubListSize( [ [ x x ] [ x x x ] ], 1) returns
    [ [ x x x ] [ x x x x ] ]
  (using the x as placeholders for the values since these are unknown).
 */
RaggedShape ChangeSublistSize(const RaggedShape &src, int32_t size_delta);

/*
  A version of ChangeSublistSize() with different behavior for edge
  cases.  If size_delta is positive and the original size of a sub-list
  was zero, then the sub-list is left at zero size, e.g.:

    ChangeSublistSizePinned( [[ x x ] [ ]], 1) returns [[ x x x ] []].

  If size_delta is negative then if reducing the size would take us below
  zero size we let the size be zero (unlike in ChangeSublistSize(), where this
  would be an error).   So
     ChangeSublistSizePinned( [[ x x x ] [ ]], -2) returns [[ x ] []].
 */
RaggedShape ChangeSublistSizePinned(RaggedShape &src, int32_t size_delta);

/*
  Return a sub-part of a RaggedShape containing indexes 0 through n-1 of
  its 1st axis.
      @param [in] src  Source RaggedShape
      @param [in] n    Number of (leading) indexes to keep; result will
                       satisfy ans.Dim0() == n; Must have 0 <= n <= src.Dim0().
      @return          Returns RaggedShape containing a prefix of `src`.
                       It will share memory with `src`.
 */
RaggedShape Prefix(RaggedShape &src, int32_t n);

/*
  Return a vector of RaggedShapes containing various prefixes of `src`.

        @param [in] src    Source RaggedShape
        @param [in] sizes  Lengths of desired prefixes; all elements
                           will satisfy 0 <= sizes[i] <= src.Dim0().
        @return   Returns vector of prefixes of a RaggedShape;
                  ans[i] will be equal to Prefix(src, sizes[i]).
                  We provide this interface because individual
                  calls to `Prefix` would otherwise require multiple
                  GPU->CPU memory transfers.
 */
std::vector<RaggedShape> GetPrefixes(RaggedShape &src,
                                     const std::vector<int32_t> &sizes);

/*
  This object splits a ragged shape on its axis 0, giving you efficient
  axis to the sub-parts of it for each index into its axis0.
 */
class RaggedShapeAxis0Splitter {
 public:
  explicit RaggedShapeAxis0Splitter(RaggedShape &src) { Init(src); }

  /*
     Return sub-part of `src`
        @param [in] i   Index into axis 0 of `src`, with 0 <= i < src.Dim0().
        @param [out] elem_offset  If not nullptr, will output to this
                     location the offset into the `values` array that
                     you'd need to start, to get the values of this
                     sub-part of the ragged shape.  Will satisfy
                     0 <= elem_offset <= src.NumElements().
        @return  Returns the sub-part of `src`, with one fewer axis
                  than `src`.  Is equivalent to calling src.Index(0, i,
                  elem_offset), but more efficient if you'll do this for most of
                  the `i`.
  */
  RaggedShape GetElement(int32_t i, int32_t *elem_offset = nullptr);

  /*
    This is provided in case you need to know how the indexes of the sub-pieces
    relate to the indexes of the original array; GetOffset(i, axis)
    gives the offset of an index on axis `axis - 1` of GetElement(i) versus axis
    `axis` of `src`.

    GetOffset(i, src.NumAxes() - 1) will equal the `elem_offset`
    output by `GetElement(i, &elem_offset)`.


    Below, `src` refers to the argument to the constructor.
      @param [in] i  Element-index i, with 0 <= i <= src.Dim0()
      @param [in] src_axis  Axis of `src`, with 0 = src_axis < src.NumAxes()
      @return  Returns the offset, with 0 <= ans <= src.TotSize(src_axis)

  */
  int32_t GetOffset(int32_t i, int32_t src_axis) {
    return composite_row_splits_cpu_.Accessor()(src_axis, i);
  }

  void Init(RaggedShape &src);  // called from constructor, to get around nvcc
                                // limitations.  Do not call.

 private:
  // composite_row_splits is of shape (src.NumLayers() + 1, src.Dim0() + 1).
  // Row 0 contains [ 0, 1, 2, ..  ];
  // Row 1 contains src.RowSplits(1)
  // Row 2 contains the row_splits12 = src.RowSplits(2)[src.RowSplits(1)]
  //    .. etc.
  // Note: we're making composite_row_splits_ a class member in case we need
  // it later; otherwise it could be a local in the constructor.
  Array2<int32_t> composite_row_splits_;
  Array2<int32_t> composite_row_splits_cpu_;

  // Let num_layers_out = composite_row_splits.Dim0() - 2 == src.NumLayers() -
  // 1.  The first `num_layers_out` arrays in row_splits_out and row_ids_out
  // will be populated.
  //
  // row_splits_out_[i] contains all the row_splits of layer i of the split
  // outputs, appended together, of Dim() equal to src.TotSize(i+1) + src.Dim0()
  // + 1; the extra Dim0() is needed because of the extra final element of each
  // row_splits vector, plus 1 for an extra final zero that's convenient for the
  // implementation.
  Array1<int32_t> row_splits_out_[4];
  // row_ids_out[i] contains all the row_ids of layer i of the outputs, appended
  // together, of Dim() equal to src.TotSize(i+2).
  Array1<int32_t> row_ids_out_[4];
};

template <typename T>
class RaggedAxis0Splitter : public RaggedShapeAxis0Splitter {
 public:
  explicit RaggedAxis0Splitter(Ragged<T> &src)
      : RaggedShapeAxis0Splitter(src.shape), values_(src.values) {}

  /*
     Return sub-part of `src`
        @param [in] i   Index into axis 0 of `src`, with 0 <= i < src.Dim0().
        @param [out] elem_offset  If not nullptr, will output to this
                  location the offset into the `values` array that
                  the answer's data starts from.  Will satisfy
                  0 <= elem_offset <= src.NumElements().
        @return  Returns the sub-part of `src`, with one fewer axis
                  than `src`.  Is equivalent to calling src.Index(0, i,
                  elem_offset), but more efficient if you'll do this for most of
                  the `i`.
  */
  Ragged<T> GetElement(int32_t i, int32_t *elem_offset = nullptr) {
    int32_t temp;
    if (elem_offset == nullptr) elem_offset = &temp;
    RaggedShape shape = RaggedShapeAxis0Splitter::GetElement(i, elem_offset);
    return Ragged<T>(shape, values_.Arange(*elem_offset,
                                           *elem_offset + shape.NumElements()));
  }

  // note, GetOffset() from the parent is available.
 private:
  Array1<T> values_;
};

/*
  Return a sub-range of `src` containing indexes `begin` through `end - 1`
  along axis `axis` of src.  The `axis` argument may be confusing;
  its behavior is equivalent to:

   for (int i = 0; i < axis; i++) src = src.RemoveAxis(0)
   return Arange(src, 0, begin, end, value_range)

      @param [in] src   Source RaggedShape. Must have src.NumAxes() >= 2.
      @param [in] axis  The axis we'll get ans, must have
                        0 <= axis < src.NumAxes() - 1.
      @param [in] begin The first element we'll get along axis `axis`.
                        Must have 0 <= begin <= end <= src.TotSize(axis).
      @param [in] end   The one-past-the-last element we'll get along axis
                       `axis`.
                       Must have 0 <= begin <= end <= src.TotSize(axis).
      @param [out] value_range If non-null, will be set to a pair
                        (val_begin, val_end) and users can get the values
                        of ans with src.values.Arange(val_begin, val_end).
      @return   Returns a RaggedShape with NumAxes() == src.NumAxes() - `axis`,
                it contains element of `src` along axis `axis` from `begin`
                to `end - 1`.
 */
RaggedShape Arange(RaggedShape &src, int32_t axis, int32_t begin, int32_t end,
                   std::pair<int32_t, int32_t> *value_range = nullptr);

/*
  Returns a Ragged array which is a sub-range of `src` containing indexes
  `begin` through `end - 1` along axis `axis` of src. See above version for
  RaggedShape for the requirements of input parameters.
 */
template <typename T>
Ragged<T> Arange(Ragged<T> &src, int32_t axis, int32_t begin, int32_t end) {
  std::pair<int32_t, int32_t> value_range;
  RaggedShape ans_shape = Arange(src.shape, axis, begin, end, &value_range);
  return Ragged<T>(ans_shape,
                   src.values.Arange(value_range.first, value_range.second));
}

/*
  Append a single element to each sub-array of a ragged matrix (we consider
  only its last axis).
     @param [in] src     Source ragged tensor
     @param [in] suffix  Array containing elements to append (they will
                         be appended regardless of value, for now).
                         Must have
                         `suffix.Dim() == src.TotSize(src.NumAxes() - 2)`
     @return         Returns ragged tensor with same num-axes as `src`,
                     and NumElements() equal to src.NumElements() +
                     suffix.Dim()
 */
Ragged<int32_t> AddSuffixToRagged(const Ragged<int32_t> &src,
                                  const Array1<int32_t> &suffix);

/*
  Prepend a single element to each sub-array of a ragged matrix (we consider
  only its last axis).
     @param [in] src     Source ragged tensor
     @param [in] prefix  Array containing elements to prepend (they will
                         be prepended regardless of value, for now).
                         Must have
                         `prefix.Dim() == src.TotSize(src.NumAxes() - 2)`
     @return         Returns ragged tensor with same num-axes as `src`,
                     and NumElements() equal to src.NumElements() +
                     prefix.Dim()
 */
Ragged<int32_t> AddPrefixToRagged(const Ragged<int32_t> &src,
                                  const Array1<int32_t> &prefix);
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
  things like Eval() with custom lambdas more.  Also: axis==0 is probably
  the only really useful case. See more useful notes in comments in the
  implementation.
 */
RaggedShape Unsqueeze(const RaggedShape &src, int32_t axis);

/*
  Version of Unsqueeze() above, that works for ragged tensors.
  Note: the opposite of this is not Squeeze(); it is ans.RemoveAxis(axis).
*/
template <typename T>
Ragged<T> Unsqueeze(const Ragged<T> &src, int32_t axis) {
  return Ragged<T>(Unsqueeze(src.shape, axis), src.values);
}

/*
  Parallel version of Unsqueeze() that effectively calls Unsqueeze() in parallel
  for `num_srcs` RaggedShapes.  Currently only works for axis == 0
      @param [in] num_srcs  `num_srcs >= 0` is the number of elements of the
                         array that `src` points to.
      @param [in]  srcs  The array of input RaggedShape, with elements
                         `*(src[0])`, `*(src[1])`, and so on.
      @param [in] axis  The axis to unsqueeze (see the other version of
                        Unsqueeze() for explanation).
                        CAUTION: only supports axis == 0 currently.
      @return       Returns vector of unsqueezed RaggedShape, with
                    `ans.size() == num_srcs.`
 */
std::vector<RaggedShape> UnsqueezeParallel(int32_t num_srcs, RaggedShape **src,
                                           int32_t axis);

/*
   Remove an axis; if it is not the first or last axis, this is done by appending lists
   (effectively the axis is combined with the following axis).  If it is the
   last axis it is just removed and the number of elements may be changed.
   Effectively removes element numbered `axis` from the vector tot_sizes
   `[ src.TotSize(0), src.TotSize(1), ... src.TotSize(axis - 1) ]`

          @param [in] src Ragged shape to remove axis of (`src` is conceptually
                      unchanged by this operation but non-const because row-ids
                      may need to be generated).
                      We require src.NumAxes() > 2, since the minimum number of
                      axes for a RaggedShape is 2.
          @param [in] axis  Axis to remove; must satisfy
                            0 <= axis < src.NumAxes()
          @return      Returns the modified shape with one fewer axis; will
                       satisfy ans.TotSize(axis) == src.TotSize(axis + 1)
                       if axis < src.NumAxes() - 1.
*/
RaggedShape RemoveAxis(RaggedShape &src, int32_t axis);

/*
    Return a version of `src` with one axis removed, done by appending
    lists (this axis is combined with the following axis).  Effectively removes
    element numbered `axis` from the vector of tot_sizes `[ src.TotSize(0),
    src.TotSize(1), ... src.TotSize(axis - 1) ]`

    Note *this is conceptually unchanged by this operation but non-const
    because this->shape's row-ids may need to be generated.
    This function is defined in ragged_ops_inl.h.

        @param [in] src   Source ragged tensor to modify.
        @param [in] axis  Axis to remove.  Requires 0 <= axis < NumAxes() - 1.

        @return  Returns the modified ragged tensor, which will share the same
            `values` and some of the same shape metdata as `*this`.
  */
template <typename T>
Ragged<T> RemoveAxis(Ragged<T> &src, int32_t axis) {
  return src.RemoveAxis(axis);
}

/*
  Returns a `sub-shape` of `src` consisting of one of its RaggedShapeLayer
  elements, i.e. one of the levels of its shape.  This returned shape
  will have NumAxes() == 2, but it is the minimal case of a RaggedShape.

    @param [in] src   Source RaggedShape
    @param [in] layer Layer that is desired, from 0 .. src.NumAxes() - 2.
                      View this as an index into its Layers() vector.
 */
RaggedShape GetLayer(const RaggedShape &src, int32_t layer);

/*
  This is the inverse of ComposeRaggedShapes(); it splits up a RaggedShape
  into two pieces such that `top->NumElements() == bottom->Dim0()`.

     @param [in] src   Source RaggedShape
     @param [in] axis  Axis to split at; must satisfy
                       0 < axis < src.NumLayers() - 1.  Axis `axis` of
                       the input will correspond to the last axis of
                       `top` and axis 0 of `bottom`.
     @param [out] top   Top layers of the RaggedShape, will have
                        `top->NumAxes() == axis + 1` at exit.
     @param [out] bottom Bottom layers of the RaggedShape; will satisfy
                        `top->NumElements() == bottom->Dim0()` and
                        `Equal(src, ComposeRaggedShapes(*top, *bottom))`,
                        ans `bottom.NumAxes() == src.NumAxes() - axis`.
 */
void DecomposeRaggedShape(const RaggedShape &src, int32_t axis,
                          RaggedShape *top, RaggedShape *bottom);

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
  Make a shape `src` to be transposable by appending empty rows on axis 1.
  Specifically, suppose the size of longest sub list on axis 1 is `t`
  (i.e. src.MaxSize(1) == `t`), we will append empty rows in other sub
  list to make sure that each sub list on axis 1 has size of `t`.

     @param [in] src   Shape to be transposed.  We require src.NumAxes() >= 2.
     @return           Returns the transposable shape, the sizes of sub lists on
                       axis 1 all be the same, i.e. ans.RowSplits(1) have
                       equally spaced elements.
 */
RaggedShape MakeTransposable(RaggedShape &src);

/*
  Transpose a RaggedShape: namely, axes 0 and 1.  Requires that the sizes
  of lists on axis 1 all be the same, i.e. that src.RowSplits(1) have
  equally spaced elements.

     @param [in] src   Shape to be transposed.  We require src.NumAxes() > 2.
                       (this is because the implementation would be slightly
                       different, and because if you had a ragged array
                       with 2 axes and a regular shape, you should really
                       be using an Array2 or Tensor).
     @param [out] value_indexes  If not nullptr, will be set to a vector
                       that can be used to reorder the values of a raggged
                       tensor with this shape, as in
                       `ans_values = src_values[*value_indexes].`
     @return           Returns the transposed shape, with axes 0 and 1
                       swapped.  Will satisfy
                       ans.Dim0() == src.TotSize(1) / src.Dim0()
                       and ans[i,j,k] = src[j,i,k] (Note, this is not actual C++
                       code, it represents a conceptual indexing operator).
 */
RaggedShape Transpose(RaggedShape &src,
                      Array1<int32_t> *value_indexes = nullptr);

/*
  Transpose a Ragged array: namely, axes 0 and 1.  Requires that the sizes
  of lists on axis 1 all be the same, i.e. that src.RowSplits(1) have
  equally spaced elements.

     @param [in] src   Ragged array to be transposed.  We require
                       src.NumAxes() > 2. (this is because the
                       implementation would be slightly different, and
                       because if you had a ragged array with 2 axes and
                       a regular shape, you should really be using an
                       Array2 or Tensor).
     @param [out]      value_indexes_out   If not nullptr, will be set
                       to a new Array1<int32_t> containing the indexes that
                       were used to renumber `src`, as in
                       `ans.values = src.values[*value_indexes_out]`.
     @return           Returns the transposed ragged array, with axes 0 and 1
                       swapped.  Will satisfy
                       ans.Dim0() == src.TotSize(1) / src.Dim0()
                       and ans[i,j,k] = src[j,i,k] (Note, this is not actual C++
                       code, it represents a conceptual indexing operator).
 */
template <typename T>
Ragged<T> Transpose(Ragged<T> &src,
                    Array1<int32_t> *value_indexes_out = nullptr) {
  Array1<int32_t> value_indexes;
  RaggedShape ans_shape = Transpose(src.shape, &value_indexes);
  if (value_indexes_out) *value_indexes_out = value_indexes;
  return Ragged<T>(ans_shape, src.values[value_indexes]);
}

/*
   Concatenate a list of RaggedShape to form a single RaggedShape.

      @param [in] axis     Axis to append them on.  Currently
                           we only support axis == 0 or axis == 1.
                           Previous axes must
                           have the same shape, i.e. if axis == 1
                           then `src[i]->Dim0()` must all have the
                           same value
      @param [in] num_srcs Number of source shapes to append; require
                           num_srcs > 0.
      @param [in] src      Array of sources to append
      @param [out] merge_map  If not nullptr, will be set to the merge-map
                        that tells us for each 0 <= i < ans.NumElements(),
                        which element of `src` it came from (available
                        as `merge_map[i] % num_srcs`) and its element-index
                        within `src[i]` (available as `merge_map[i] / num_srcs`.

      @return      Returns the appended RaggedShape; will have the same number
                   of axes as the sources.
*/
RaggedShape Cat(int32_t axis, int32_t num_srcs, RaggedShape **src,
                Array1<uint32_t> *merge_map = nullptr);

/*
  Extract meta-info from the shape (this will include populating any row_ids and
  row_splits that were not already populated).  This is used inside algorithms
  when we need to transfer meta-info to GPU.

     @param [in]   src   Ragged shape that we're extracting meta-info from
     @param [out] row_splits  This will be set to an array of size
                              src.NumAxes()-1, containing pointers to the
                              row_splits' Data() vectors. The array will be
                              allocated on the same device as `src`.
     @param [out] row_ids     This will be set to an array of size
                              src.NumAxes()-1, containing pointers to the
                              row_ids' Data() vectors. The array will be
                              allocated on the same device as `src`.
*/
void GetRowInfo(RaggedShape &src, Array1<int32_t *> *row_splits,
                Array1<int32_t *> *row_ids);

/*
  Get some meta-info for an array of RaggedShape, and transfer them
  to the device that `src` is located on. Just same with `GetRowInfo`
  above, but for multiple RaggedShapes.

     @param [in] num_srcs  Number of source arrays to process.
     @param [in] src      Source arrays.  All of them must have same num_axes
                          and on the same device, but we just check this in
                          debug mode.
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
*/
void GetRowInfoMulti(int32_t num_srcs, RaggedShape **src,
                     Array2<int32_t *> *row_splits, Array2<int32_t *> *row_ids);

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
  Return ragged shape with only a subset of the elements or sub-lists
  on the specified axis kept.  (This is not regular sampling, it is
  irregular subsampling with specified elements kept).

    @param [in] src  The ragged shape that we are subsampling
    @param [in] renumbering  The renumbering object that dictates
                    which elements of `src` we keep; we require
                    renumbering.NumOldElems() == src.TotSize(axis2)
                    where axis2 = (axis < 0 ? src.NumAxes() + axis : axis).
    @param [in] axis  The axis to subsample; if negative, will be
                    interpreted as an offset from src.NumAxes().
    @param [out] elems_new2old   If supplied, this function will
                    output to this location a new2old vector that
                    dictates how the elements of a ragged tensor
                    with shape `src` would be renumbered.
    @return  Returns the subsampled shape. All dimensions and tot-sizes
       preceding the axis `axis` will remain the same, which might give
       rise to empty lists on those axes; these can be removed if
       necessary with RemoveEmptyLists().

  Notice the other version of this function below.
 */
RaggedShape SubsetRaggedShape(RaggedShape &src,
                              Renumbering &renumbering,
                              int32_t axis = -1,
                              Array1<int32_t> *elems_new2old = nullptr);

/*
  Return ragged shape with only a subset of the elements on the last
  and one-before-last axes kept.

  Require renumbering_last.NumOldElems() == src.NumElements(), and
  renumbering_before_last.NumOldElems() == src.TotSize(src.NumAxes() - 2).
  Note: all dimensions and tot-sizes preceding the last two axes will remain the
  same, which might give rise to empty lists.
 */
RaggedShape SubsetRaggedShape(RaggedShape &src,
                              Renumbering &renumbering_before_last,
                              Renumbering &renumbering_last);

/*
  Removes empty lists on a particular axis (not last axis) of a RaggedShape,
  returning the modified shape with those lists removed.
     @param [in] src_shape   RaggedShape that possibly has empty lists
                          to be removed
     @param [in] axis     Axis that is not the last axis of `src_shape`,
                          i.e. with `axis + 1 < src_shape.NumAxes()`.
     @param [out] renumbering  If not nullptr, a renumbering object that maps
                         between old and new indexes on axis `axis` (e.g. if
                         `axis == 0` would map between idx0's and idx0's; if
                         `axis == 1`, would map between idx01's and idx01's).
     @return             Returns modified shape with
                         ans.NumAxes() == src_shape.NumAxes().
                         ans.TotSize(axis) may differ from
                         src_shape.TotSize(axis), but other TotSize() values,
                         and the numbering on other axes, will remain the same.
 */
RaggedShape RemoveEmptyLists(RaggedShape &src_shape, int32_t axis,
                             Renumbering *renumbering = nullptr);

/*
  Removes some subset of empty lists on a particular axis (not last axis) of
  a RaggedShape, returning the modified shape with those lists removed.

     @param [in] src_shape   RaggedShape that possibly has empty lists
                          to be removed
     @param [in] axis     Axis that is not the last axis of `src_shape`,
                          i.e. with `axis + 1 < src_shape.NumAxes()`.
     @param [in] renumbering  If not nullptr, a renumbering object that maps
                         between old and new indexes on axis `axis` (e.g. if
                         `axis == 0` would map between idx0's and idx0's; if
                         `axis == 1`, would map between idx01's and idx01's).
                         It is assumed that this renumbering preserves
                         all lists that are nonempty.
     @return             Returns modified shape with
                         ans.NumAxes() == src_shape.NumAxes().
                         ans.TotSize(axis) may differ from
                         src_shape.TotSize(axis), but other TotSize() values,
                         and the numbering on other axes, will remain the same.
 */
RaggedShape RemoveSomeEmptyLists(RaggedShape &src_shape, int32_t axis,
                                 Renumbering &renumbering);

/*
  Removes empty lists on axis 0 of a RaggedShape, returning the modified shape
  with those lists removed.  Note: a list containing empty lists is not empty.

     @param [in] src_shape   RaggedShape that possibly has empty lists on its
                         axis 0
     @param [out] renumbering  If not nullptr, a renumbering object that maps
                         between old and new indexes on axis 0, i.e. between
                         old and new idx0's.
     @return             Returns modified shape with
                         ans.NumAxes() == src_shape.NumAxes().
                         ans.Dim0() may differ from src_shape.Dim0(),
                         but for axis > 0, we have
                         `ans.TotSize(axis) == src.TotSize(axis)`.
*/
RaggedShape RemoveEmptyListsAxis0(RaggedShape &src_shape,
                                  Renumbering *renumbering = nullptr);

/*
  Removes some (but not necessarily all) empty lists on axis 0 of a RaggedShape,
  returning the modified shape with those lists removed.  Note: a list
  containing empty lists is not empty. (this is what we mean by the "Simple"
  part of the name, as it means we only have to deal with one layer).

     @param [in] src_shape   RaggedShape that possibly has empty lists on its
                         axis 0
     @param [out] renumbering  If not nullptr, a renumbering object that maps
                         between old and new indexes on axis 0, i.e. between
                         old and new idx0's.  The removed lists must be empty.

     @return             Returns modified shape with
                         ans.NumAxes() == src_shape.NumAxes().
                         ans.Dim0() may differ from src_shape.Dim0(),
                         but for axis > 0, we have
                         `ans.TotSize(axis) == src.TotSize(axis)`.
 */
RaggedShape RenumberAxis0Simple(RaggedShape &src_shape,
                                Renumbering &renumbering);



/*
  Return ragged array with only a subset of the elements or sub-lists
  on the specified axis kept.  (This is not regular sampling, it is
  irregular subsampling with specified elements kept).

    @param [in] src  The ragged shape that we are subsampling
    @param [in] renumbering  The renumbering object that dictates
                    which elements of `src` we keep; we require
                    renumbering.NumOldElems() == src.TotSize(axis2)
                    where axis2 = (axis < 0 ? src.NumAxes() - axis : axis).
    @param [in] axis  The axis to subsample; if negative, will be
                    interpreted as an offset from src.NumAxes().
    @param [out] elems_new2old   If supplied, this function will
                    output to this location a new2old array that
                    dictates how the elements of a ragged tensor
                    with shape `src` would be renumbered.
    @return  Returns the subsampled shape. All dimensions and tot-sizes
       preceding the axis `axis` will remain the same, which might give
       rise to empty lists on those axes; these can be removed if
       necessary with RemoveEmptyLists().
 */
template <typename T>
Ragged<T> SubsetRagged(Ragged<T> &src, Renumbering &renumbering,
                       int32_t axis = -1,
                       Array1<int32_t> *elems_new2old = nullptr) {
  Array1<int32_t> tmp;
  if (elems_new2old == nullptr)
    elems_new2old = &tmp;
  RaggedShape shape = SubsetRaggedShape(src.shape, renumbering,
                                           axis, elems_new2old);
  return Ragged<T>(shape, src.values[*elems_new2old]);
}

/*
  This function creates a Renumbering object that can be used to obtain subsets
  of ragged arrays via SubsetRaggedShape().  It implements beam pruning as
  used in pruned Viterbi search and similar algorithms, where there is both a
  beam and a max-active (`max_elems`) constraint.  T will probably be float or
  double, interpreted as a "positive-is-better" sense, i.e. as scores.

   @param [in] src  The ragged object to be subsampled.
   @param [in] axis  The axis to be subsampled, must satisfy
              0 <= axis < src.NumAxes().  The axis before `axis`, if axis > 0,
              will be interpreted as a "batch" axis.
   @param [in] beam  The main pruning beam.  The sub-lists of elements on axis
              `axis` will be removed if their maximum element (or the element
              itself, if axis + 1 == src.NumAxes()) is less than
              this_best_elem - beam, where this_best_elem
              is the maximum element taken over axis `axis-1` (or over the
              entire array, if axis == 0).   Think of axis `axis-1`, if
              axis > 0, as the "batch" axis, and axis `axis` as the axis that we
              actually remove elements or sub-lists on.  Empty sub-lists on axis
              `axis` will always be pruned, as their score would be treated
              as -infinity.
   @param [in] max_elems  If max_elems > 0, it is the maximum number of
              sub-lists or elements that are allowed within any sub-list
              on axis `axis-1` (or the maximum number of top-level sub-lists
              after subsampling, if axis == 0).  We keep the best ones.
              If max_elems <= 0, there is no such constraint.
    @return  Returns the renumbering object to be used to actually
             prune/subsample the specified axis.

   Example:
      PruneRagged([ [0 -1 -2 -3], [ -10, -20 ], [ ] ], 1, 5.0, 3)
    would create a Renumbering object that would prune the
    ragged tensor to [ [0 -1 -2], [ -10 ], [ ] ]

      PruneRagged([ [0 -1 -2 -3], [ -10, -20 ], [ ] ], 0, 5.0, 0)
    would create a Renumbering object that would prune the
    ragged tensor to [ [0 -1 -2 -3] ]
 */
template <typename T>
Renumbering PruneRagged(Ragged<T> &src,
                        int32_t axis,
                        T beam,
                        int32_t max_elems);

/*
  Stack a list of Ragged arrays to create a Ragged array with one more axis.
  Similar to TF/PyTorch's Stack.  The result will have Dim0 == num_srcs.  All
  the source Ragged arrays' shapes must have the same NumAxes().

     @param [in] axis   The new axis whose dimension will equal num_srcs.
                        The shapes/dimensions must be the same for all
                        preceding axes.
     @param [in] num_srcs  The number of `RaggedShape`s in `src`
     @param [in] src       The shapes to be stacked
     @param [out] merge_map  If not nullptr, will be set to the merge-map
                       that tells us for each 0 <= i < ans.NumElements(),
                       which element of `src` it came from (available
                       as `merge_map[i] % num_srcs`) and its element-index
                       within `src[i]` (available as `merge_map[i] / num_srcs`.


     @return  The appended result.
       Assuming as an example that the input had 3 axes:
       if axis==0, the result would have:
          result[i,j,k,l] = (*src[i])[j,k,l]
       and if axis==1 we would have:
          result[i,j,k,l] = (*src[j])[i,k,l]
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> **src,
                Array1<uint32_t> *merge_map = nullptr);

/*
  This version of Stack() has one fewer levels of pointer indirection,
  it is just a wrapper for the version above.
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> *src,
                Array1<uint32_t> *merge_map = nullptr);

/*
  Unstack a Ragged tensor to a list of Ragged, tensors all the output Ragged
  tensors have one less axis. Similar to TF's Unstack (or unbind in Pytorch).

    @param [in] src  The ragged tensor to be unstacked.
    @param [in] axis  The axis to be removed, all the elements of this axis will
                      be rearranged into output Raggeds.
    @param [in] pad_right  Before unstack, we will (conceptually) pad the
                  sublists along axis `axis` to the same size with empty lists
                  `pad_right` tells where to put the padding empty lists, see
                  the example for more details.
                  Note, `pad_right` makes no difference when `axis == 0` or
                  `axis == src.NumAxes() - 1`.
    @param [out] out  The container where the output ragged tensors would write
                      to. MUST NOT be a nullptr, will be reallocated.
    @param [out] split_map  If not nullptr will store the element-index within
                   the `src` telling where the elements of each split Raggeds
                   comes from. It has same size as `out`, see notes below for
                   the dimension of `out`. For Array1 in each of the
                   `split_map`, It satifies
                   `split_map[i].Dim() == out[i].values.Dim()`, and it contains
                   the element-index within `src`.
                   (i.e.`out[i].values[j] == src.values[split_map[i][j]]`)
                   `split_map` will be reallocated by this function.

  Caution: If `src.NumAxes() == 2`, the output shapes will only have one
           dimension, to make it a ragged tensor, we will add a TrivialShape on
           each of the output tensors.

  Note: The output ragged tensors may contain empty lists on axis `axis`,
        you can remove them by RemoveEmptyLists if needed.

  Note: The number of output ragged tensors is decided by the size of sublist
        with max number of elements along axis `axis`, for `axis == 0`, it has
        only one sublist along `axis == 0`(i.e. the src itself), so the number
        of output ragged will be equal to `src.Dim0()`.

  A small example of unstacking a 3 axes Ragged (with pad_right = true):

    src: [ [ [ 1 2 ] [ 3 ] ] [ [ 4 ] ] ]
    unstack on axis 0:
    src.Dim0() == 2, will produce 2 ragged tensors.

    out[0] : [ [ 1 2 ] [ 3 ] ]   split_map[0] : [0, 1, 2]
    out[1] : [ [ 4 ] ]           split_map[1] : [3]

    unstack on axis 1:
    two sublists along axis 1, the sizes are [2, 1], will produce 2 ragged tensors

    think about that we first pad src to [ [ [ 1 2 ] [ 3 ] ] [ [ 4 ] [ ] ] ]
    then select elements along axis 1 into separate raggeds

    out[0] : [ [ 1 2 ] [ 4 ] ]   split_map[0] : [0, 1, 3]
    out[1] : [ [ 3 ] [ ] ]       split_map[1] : [2]

    unstack on axis 2:
    three sublists along axis 2, the sizes are [2, 1, 1], will produce 2
    ragged tensors.

    out[0] : [ [ 1 3 ] [ 4 ] ]   split_map[0] : [0, 2, 3]
    out[1] : [ [ 2 ] [ ] ]       split_map[1] : [1]

  for pad_right equals to false:

    src: [ [ [ 1 2 ] [ 3 ] ] [ [ 4 ] ] ]

    unstack on axis 1:

    think about that we first pad src to [ [ [ 1 2 ] [ 3 ] ] [ [ ] [ 4 ] ] ]
    then select elements along axis 1 into separate raggeds

    out[0] : [ [ 1 2 ] [ ] ]       split_map[0] : [0, 1]
    out[1] : [ [ 3 ] [ 4 ] ]       split_map[1] : [2, 3]
 */
template <typename T>
void Unstack(Ragged<T> src, int32_t axis, bool pad_right,
             std::vector<Ragged<T>> *out,
             std::vector<Array1<int32_t>> *split_map = nullptr);

/*
 * The same as above, except that it uses `pad_right=true`.
 */
template <typename T>
void Unstack(Ragged<T> src, int32_t axis, std::vector<Ragged<T>> *out,
             std::vector<Array1<int32_t>> *split_map = nullptr) {
  Unstack(src, axis, true /*pad_right*/, out, split_map);
}

/*
   Concatenate a list of Ragged<T> to form a single Ragged<T>.

      @param [in] axis     Axis to append them on.
                           Previous axes must
                           have the same shape, i.e. if axis == 1
                           then `src[i]->Dim0()` must all have the
                           same value
      @param [in] num_srcs Number of source shapes to append; require
                           num_srcs > 0.
      @param [in] src      Array of sources to append
      @param [out] merge_map  If not nullptr, will be set to the merge-map
                       that tells us for each 0 <= i < ans.NumElements(),
                       which element of `src` it came from (available
                       as `merge_map[i] % num_srcs`) and its element-index
                       within `src[i]` (available as `merge_map[i] / num_srcs`.

      @return      Returns the appended RaggedShape.
*/
template <typename T>
Ragged<T> Cat(int32_t axis, int32_t num_srcs, Ragged<T> **src,
              Array1<uint32_t> *merge_map = nullptr);

/*
  This version of Cat() has one fewer levels of pointer indirection,
  it is just a wrapper for the version above.
 */
template <typename T>
Ragged<T> Cat(int32_t axis, int32_t num_srcs, Ragged<T> *src,
              Array1<uint32_t> *merge_map = nullptr);

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
     @param [in] a   RaggedShape describing the top/first layers
                     of the returned shape
     @param [in] b   RaggedShape describing the bottom layers (later
                     indexes) of the returned shape.  We require
                     a.NumElements() == b.Dim0().
     @return     Returns the combined ragged shape.  Its number of axes
                 will be equal to a.NumAxes() + b.NumAxes() - 1 (the last
                 axis of a and the first axis of b are combined).
 */
RaggedShape ComposeRaggedShapes(const RaggedShape &a, const RaggedShape &b);

/*  3-arg version of ComposeRaggedShapes,
       @param [in] a  RaggedShape describing the top/first layers
                     of the returned shape
       @param [in] b  RaggedShape describing the intermediate layers
                     of the returned shape, require b.Dim0() == a.NumElements()
       @param [in] c  RaggedShape describing the lower/last layers
                     of the returned shape, require c.Dim0() == b.NumElements()
       @return Returns the combined ragged shape; its num-layers (==num-axes -
   1) will be the total of the num-layers of the sources.  Will share memory
   with the inputs.
*/
RaggedShape ComposeRaggedShapes3(const RaggedShape &a, const RaggedShape &b,
                                 const RaggedShape &c);

/*
  Construct a RaggedShape with 3 axes.  For N=1 and 2 respectively:
  either row_splitsN or row_idsN or both must be non-NULL.
  If cached_tot_sizeN is not -1, it must equal the total size on
  that axis which will equal the last element of row_splitsN (if
  provided) and must equal the row_idsN.Dim(), if provided. See
  documentation above for RaggedShape2 for details.

  We also require that (supposing both row_splitsN and row_idsN are non-NULL):
  row_splits1[row_splits1.Dim() - 1] == row_ids1.Dim()
     == (row_splits2.Dim() - 1)
     >= (row_ids2[row_ids2.Dim() - 1] + 1)
*/
RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2);

/* See documentation of RaggedShape3, this is an obvious extension. */
RaggedShape RaggedShape4(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2,
                         Array1<int32_t> *row_splits3,
                         Array1<int32_t> *row_ids3, int32_t cached_tot_size3);

/*
  Returns a RaggedShape with 2 axes, with Dim0() == 1 and
  TotSize(1) = num_elems.
 */
RaggedShape TrivialShape(ContextPtr &c, int32_t num_elems);

/*
  Returns a RaggedShape with Dim0() == dim0 and TotSize(1) == dim0 * dim1.
  Require dim0 >= 0 and dim1 >= 0.
 */
RaggedShape RegularRaggedShape(ContextPtr &c, int32_t dim0, int32_t dim1);

/*
  Allocates an *invalid* RaggedShape given the TotSize() values for each axis.
  This allocates space for all the row_ids and row_splits, but does not write
  any data to them.  View this as an internal function, because such a
  RaggedShape would not be usable directly.

     @param [in] c          The context with which we'll allocate memory for
                            row_splits and row_ids.
     @param [in] num_axes   Number of axes of ragged shape desired; must be at
                            least 2.
     @param [in] tot_sizes  (CPU Pointer!)  Total size on each axis.
                           `tot_sizes[0]` through `tot_sizes[num_axes-1]` must
                            be set.

     @return           The returned RaggedShape satisfies
                       ans.TotSize(axis) == tot_sizes[axis], and has all
                       its row_splits and row_ids allocated but not
                       filled in, and its cached_tot_size elements
                       set.
 */
RaggedShape RaggedShapeFromTotSizes(ContextPtr c, int32_t num_axes,
                                    const int32_t *tot_sizes);

/*
  Returns an empty ragged shape with the specified number of axes.
  Require num_axes >= 2.
*/
RaggedShape EmptyRaggedShape(ContextPtr &c, int32_t num_axes);

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
     @param [out]     The indexes mapping from the sorted array to the original
                      input array. If not NULL, the caller has to pre-allocate
                      memory for it on the same device as `src`.
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

/* Return true if the objects represent the same ragged shape.
   They must be on the same device. */
bool Equal(const RaggedShape &a, const RaggedShape &b);

/* Return true if the objects represent the same ragged array with the same
   values.  They must be on the same device. */
template <typename T>
bool Equal(const Ragged<T> &a, const Ragged<T> &b) {
  return Equal(a.shape, b.shape) && Equal(a.values, b.values);
}

/*
  Indexing operation on ragged tensor's shape.

      @param [in] src      Source ragged tensor to index
      @param [in] axis     Axis to index `src` on, must satisfy
                           0 <= src < src.NumAxes().
      @param [in] indexes  Array of indexes, which will be interpreted
                           as indexes into axis `axis` of `src`,
                           i.e. with 0 <= indexes[i] < src.TotSize(axis).

                           As a special case, if axis == 0 we also support
                           -1 as an index, which will result in the
                           empty list (as if it were the index into
                           a position in `src` that had an empty list
                           at that point).

                           CAUTION: these are currently not allowed to
                           change the order on axes less than `axis`,
                           i.e. if axis > 0, we require.
                           `IsMonotonic(src.RowIds(axis)[indexes])`.

      @param [out]         If non-null, this will be set to an
                           Array1<int32_t> containing the indexes
                           into the elements of an array with shape
                           'src', that an array with shape 'ans'
                           would have (a new2old map).  As in:
                           `ans_values = src_values[*elem_indexes]`.
                           If `axis == src.NumAxes()-1`, this will
                           be aliased with `indexes`.

      @return Returns a ragged shape with
              `ans.NumAxes() == src.NumAxes()`
              and `ans.TotSize(axis) == indexes.Dim()`.

  NOET: if you are looking for something like ReorderRaggedShape(),
  RenumberRaggedShape() or the like, this may be what you want.
  (Reordering/renumbering is a special case of indexing)
*/
RaggedShape Index(RaggedShape &src, int32_t axis,
                  const Array1<int32_t> &indexes,
                  Array1<int32_t> *elem_indexes = nullptr);

/*
  Index ragged tensor with array, return ragged tensor.

      @param [in] src      Source ragged tensor to index
      @param [in] axis     Axis to index `src` on
      @param [in] indexes  Array of indexes, which will be interpreted
                           as indexes into axis `axis` of `src`,
                           i.e. with 0 <= indexes[i] < src.TotSize(axis).
                           CAUTION: these are currently not allowed to
                           change the order on axes less than `axis`,
                           i.e. if axis > 0, we require
                           `IsMonotonic(src.RowIds(axis)[indexes])`.
      @param [out]         If non-null, this will be set to a new
                           Array1<int32_t> containing the indexes
                           into src.values that ans.values has,
                           as in
                           `ans.values = src.values[value_indexes_out].`
                           May be useful for backprop.

      @return Returns a ragged tensor with
              `ans.NumAxes() == src.NumAxes()`
              and `ans.Dim0() == indexes.Dim()`.

*/
template <typename T>
Ragged<T> Index(Ragged<T> &src, int32_t axis, const Array1<int32_t> &indexes,
                Array1<int32_t> *value_indexes_out = nullptr) {
  Array1<int32_t> value_indexes;
  RaggedShape ans_shape = Index(src.shape, axis, indexes, &value_indexes);
  Ragged<T> ans(ans_shape, src.values[value_indexes]);
  if (value_indexes_out != nullptr)
    *value_indexes_out = std::move(value_indexes);
  return ans;
}

/*
   Merge a list of RaggedShape by combining their top-level lists (those
   obtained by indexing on their axis 0) with the provided merge_map
   that indicates the order to take items in.

      @param [in] num_srcs Number of source shapes to append; require
                           num_srcs > 0.
      @param [in] src      Array of sources to append; must have compatible
                           contexts and the same number of axes.
      @param [in] merge_map   Merge map (probably obtained from some previous
                         ragged operation) that dictates the order in which
                         to combine elements.  `merge_map.Dim()` must equal
                         the sum of `src[i]->Dim0()` for all 0 <= i < num_srcs.
                         If `merge_map[i] == m` then at position i on axis 0 of
                         the output we take element `m / num_srcs` on axis 0 of
                         the source numbered `m % num_srcs`.
      @param [out] merge_map_out  If not nullptr, will be set to the merge-map
                        that tells us for each 0 <= i < ans.NumElements(),
                        which element of `src` it came from (available
                        as `merge_map[i] % num_srcs`) and its element-index
                        within `src[i]` (available as `merge_map[i] / num_srcs`.

      @return       Returns the appended RaggedShape.  Will have the same
                    number of axes as the sources.
*/
RaggedShape Merge(int32_t num_srcs, RaggedShape **src,
                  const Array1<uint32_t> &merge_map,
                  Array1<uint32_t> *merge_map_out = nullptr);

/*  Version of Merge that works on Ragged objects; see documentation for Merge()
    above. */
template <typename T>
Ragged<T> Merge(int32_t num_srcs, Ragged<T> **src,
                const Array1<uint32_t> &merge_map,
                Array1<uint32_t> *merge_map_out = nullptr);

/*
  Returns a ragged tensor after removing all 'values' that were <= a provided
  cutoff.  Leaves all layers of the shape except for the last one unaffected.
  Equivalent to SubsetRaggedShape with a numbering given by (src.values[i] <=
  cutoff).
 */
template <typename T>
Ragged<T> RemoveValuesLeq(Ragged<T> &src, T cutoff);

/*
  Returns a ragged tensor after removing all 'values' that equal a provided
  target.  Leaves all layers of the shape except for the last one unaffected.
  Equivalent to SubsetRaggedShape with a numbering given by (src.values[i] ==
  target).
*/
template <typename T>
Ragged<T> RemoveValuesEq(Ragged<T> &src, T target);

/*
   Index array with ragged tensor.
       @param [in] src   Source array, to be indexed
       @param [in] indexes   Indexes into source array; the values must
                          satisfy `0 <= indexes.values[i] < src.Dim()`.
       @return   Returns ragged tensor with shape `indexes.shape`
                 and values `src[indexes.values]`.
*/
template <typename T>
Ragged<T> Index(Array1<T> &src, Ragged<int32_t> &indexes) {
  return Ragged<T>(indexes.shape, src[indexes.values]);
}

/*
 * Similar to the above `Index` but also supports -1 in indexes.
 * The given default value is used if an index is -1.
 */
template <typename T>
Ragged<T> Index(Array1<T> &src, Ragged<int32_t> &indexes, T default_value) {
  return Ragged<T>(indexes.shape,
                   Index(src, indexes.values, true, default_value));
}

/*
   Index ragged tensor with ragged tensor.
       @param [in] src   Source tensor, to be indexed
       @param [in] indexes   Indexes into source array; the values must
                          satisfy `0 <= indexes.values[i] < src.Dim0()`.
       @param [in] remove_axis  If remove_axis == true,
             then we remove the last-but-one axis, which has the effect
             of appending lists, e.g.
              `Index( [[ 10 11 ] [ 12 13 ]],  [[ 0 1 ]])` would
             give us `[[ 10 11 12 13 ]]`.  If remove_axis == false
             the answer will have at least 3 axes, e.g.
             `[[[ 10 11 ] [ 12 13 ]]]` in this case.

       @return  Returns indexed tensor.

    CAUTION: the validity of the indexes is not checked, which may
    result in segfault or undefined values.
*/
template <typename T>
Ragged<T> Index(Ragged<T> &src, Ragged<int32_t> &indexes, bool remove_axis);

/*
  Returns a vector that indexes `shape` to put its rows in decreasing order of
  length.  I.e. so that `Index(shape, GetDecreasingSizeOrder(shape))` will
  give rows of decreasing length.
 */
Array1<int32_t> GetDecreasingSizeOrder(RaggedShape &shape);

/*
  Given a list of shapes with 2 axes and the same Dim0(), return the
  smallest shape that 'covers' all of them, i.e. size the i'th sub-list of the
  answer is the maximum of the sizes of the i'th sub-list of `srcs`
    @param [in] num_srcs  Number of source shapes; must have 2 axes and
                          `Dim0()` all equal
    @param [in] srcs      Array of input shapes; inputs are `*(srcs[0])`,
                          `*(srcs[1])` ...
    @return      Returns shape with the same Dim0() as all the `srcs` and
                 sub-list sizes equal to the maximum of those of the sources.
*/
RaggedShape CoveringShape(int32_t num_srcs, RaggedShape **srcs);

/*
   Returns an arc_map that says, for each element of `covering`, the
   corresponding element of `src`, or -1 if there was no such element.
    @param [in] src   Shape that was likely an input to `CoveringShape`,
                      must have 2 axes.
    @param [in] covering  Shape with 2 axes,
                     `covering.Dim0()==src.Dim0()`, and sub-list sizes
                     not less than the corresponding sub-list sizes of
                     `src`.
    @return  Returns an array with `Dim() == covering.NumElements()`,
             containing, for each element of `covering`, either the
             corresponding element of `src` or -1 if this was not
             applicable.  E.g.  if src == [ [ x x ] [ x ] ] and
             covering == [ [ x x x ] [x] ], would return [ 0 1 -1 2 ].
*/
Array1<int32_t> CoveringShapeForwardMap(RaggedShape &src,
                                        RaggedShape &covering);

/*
  Computes a hash-function of the bottom-level lists in `src`
  (i.e. only uses the final layer of `src`).

    @param [in] src   Ragged tensor of int32_t for which
                 we want hashes of its final lists.
    @return      Returns an array of int32_t or int64_t with
            `ans.Dim() == src.TotSize(src.NumAxes() - 2)`.
            If two lists in `src` were the same, the corresponding
            hash values will be the same.

   CAUTION: T must be int32_t or int64_t, and the template argument
   cannot be deduced so must be supplied, e.g.
     ComputeHash<int64_t>(src);
  */
template <typename T>
Array1<T> ComputeHash(Ragged<int32_t> &src);

/*
  If `src` has two axes, this will return the unique sub-lists (in a possibly
  different order, but without repeats).  If `src` has 3 axes, it will
  do the above but separately for each index on axis 0; if more than 3 axes,
  the earliest axes will be ignored.

     @param [in] src  Source ragged tensor
     @param [out] num_repeats  If not NULL, it will contain the number of
                       repeats (i.e., multiplicity) of each output sequence.
                       The caller does not need to pre-allocate it. It is
                       allocated inside the function.
     @param [out] new2old_indexes
                       If not NULL, on return new2old_indexes[i] contains
                       the original input sublist for the i-th output sublist.
                       If `src` has 2 axes, this array contains `src_idx0`;
                       if `src` has 3 axes, this array contains `src_idx01`.
                       CAUTION: For repeated sublists, only one of them is kept.
                       The choice of which one to keep is **deterministic** and
                       is an implementation detail.

     @return   Returns a tensor with the same number of axes as `src` and
            possibly fewer elements due to removing repeated sequences on the
            last axis (and with the last-but-one indexes possibly in a different
            order).

  CAUTION: It does not completely guarantee that all unique sequences will
  be present in the output, as it relies on a hash and ignores collisions.
  If several sequences have the same hash, only one of them is kept, even
  if the actual content in the sequence is different.

  CAUTION: Even if there are no repeated sequences, the output may be different
  from `src`. That is, `new2old_indexes` may NOT be an identity map even if
  nothing was removed.
 */
Ragged<int32_t> UniqueSequences(Ragged<int32_t> &src,
                                Ragged<int32_t> *num_repeats = nullptr,
                                Array1<int32_t> *new2old_indexes = nullptr);

/* Compute exclusive sum per sub-list.

    @param [in] src  The input ragged tensor. The exclusive sum is computed
                     for the last axis. CAUTION: The last entry of every
                     sublist does not contribute to the final sum.
    @param [out] dst The dest array. It satisfies
                     `dst.Dim() == src.NumElements().`
                     Supports `dst == &src.values`.
 */
template <typename T>
void SegmentedExclusiveSum(Ragged<T> &src, Array1<T> *dst);

/*
  Construct a Ragged with 2 axes.
    @param [in] vecs  vecs.size() is the number of rows of the returned ans,
                      i.e. ans.Dim0() == vecs.sizes(), and vecs[i] contains
                      the elments for row i in ans.
    @return   Returns the corresponding ragged array, with a CPU context.
*/
template <typename T>
Ragged<T> CreateRagged2(const std::vector<std::vector<T>> &vecs);

/*
  Pad a ragged array to be regular.
    @param [in] src  The input ragged array.
                     CAUTION: Only support `NumAxes() == 2`.
    @param [in] mode Valid values are: "constant", "replicate".
                     When it is "constant", the given padding_value
                     is used for filling. When it is "replicate",
                     the last entry of a list is used for filling.
                     When the list is empty, the given padding_value
                     is used for filling.
    @param [in] padding_value  Value for padded elements.
                     Used only when mode is "constant" or a list
                     is empty.
    @return  Returns the corresponding regular array (Array2).
 */
template <typename T>
Array2<T> PadRagged(Ragged<T> &src, const std::string &mode, T padding_value);

}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_OPS_H_
#include "k2/csrc/ragged_ops_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_OPS_H_

#endif  // K2_CSRC_RAGGED_OPS_H_
