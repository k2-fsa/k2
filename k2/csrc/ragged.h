/**
 * @brief
 * ragged
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_H_
#define K2_CSRC_RAGGED_H_

#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/utils.h"

namespace k2 {

// Caution, RaggedShapeDim is mostly for internal use and users should not
// generally interact with it directly.
// Note: row_splits is of size num_rows + 1 and row_ids is of size
// num_elements.
struct RaggedShapeDim {
  // Search for "row_splits concept" in utils.h for explanation.  row_splits
  // is required; it must always be nonempty for a RaggedShapeDim to be valid.
  Array1<int32_t> row_splits;
  // Search for "row_ids concept" in utils.h for explanation
  Array1<int32_t> row_ids;

  // cached_tot_size can be viewed as the number of elements in a ragged
  // matrix,
  // or -1 if not known.  (Note: it can legitimately be 0, if there are no
  // elements).

  // If cached_tot_size >= 0 and row_ids is nonempty, cached_tot_size will
  // equal row_ids.Dim().
  // If cached_tot_size >= 0, it will be equal to
  // row_splits[row_splits.Dim() - 1].
  int32_t cached_tot_size;
};

class RaggedShapeIndexIterator;

class RaggedShape {
 public:
  int32_t Dim0() const {
    K2_CHECK_GT(axes_.size(), 0);
    return axes_[0].row_splits.Dim() - 1;
  }
  /* Return the  total size on this axis.  Requires 0 <= axis < NumAxes() and
     for axis=0 the returned value is the same as Dim0().
     Caution: we use const_cast inside this function as it may actually modify
     the cached_tot_size members of RaggedShapeDim if not set.
  */
  int32_t TotSize(int32_t axis) const;

  /* Append `other` to `*this` (in-place version that modifies `*this`).
     `other` must have the same number of axes as `this`.  This is efficient in
     an amortized way, i.e. should take time/work that's O(n) in the size of
     `other`, not `*this`, if you make many calls to Append().  This is due to
     the policy used in Region::Extend(), where it at least doubles the size
     each time, similar to std::vector.
  */
  void Append(const RaggedShape &other);

  // Returns the number of elements that a ragged array with this shape would
  // have.
  int32_t NumElements() const { return TotSize(NumAxes() - 1); }

  /*
    Return the row-splits for axis `axis` with `0 < axis < NumAxes()`.
    The dimension is the (total) number of rows on this axis plus one,
    and the elements are in the range [0,N] where N is the TotSize()
    on axis `axis+1`
   */
  Array1<int32_t> &RowSplits(int32_t axis) {
    K2_CHECK_GT(axis, 0);
    K2_CHECK_LT(axis, NumAxes());
    // Note row_splits is always nonempty for valid RaggedShapeDim.
    return axes_[axis - 1].row_splits;
  }

  const Array1<int32_t> &RowSplits(int32_t axis) const {
    K2_CHECK_GT(axis, 0);
    K2_CHECK_LT(axis, NumAxes());
    // Note row_splits is always nonempty for valid RaggedShapeDim.
    return axes_[axis - 1].row_splits;
  }

  /*
    Return the row-ids for axis `axis` with `0 < axis < NumAxes()`.
    The dimension is the number of elements on this axis == TotSize(axis).
  */
  Array1<int32_t> &RowIds(int32_t axis);

  int32_t NumAxes() const { return static_cast<int32_t>(axes_.size()) + 1; }

  // Gives max size of any list on the provided axis,
  // with 0 < axis < NumAxes().  Equals max difference between successive
  // row_splits on that axis.
  int32_t MaxSize(int32_t axis);

  ContextPtr &Context() const { return axes_[0].row_splits.Context(); }

  /*
    It is an error to call this if this.NumAxes() < 2.  This will return
    a RaggedShape with one fewer axis, containing only the elements of
    *this for which the value on axis `axis` is i.  CAUTION:
    currently this only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
   */
  RaggedShape Index(int32_t axis, int32_t i);

  /*
    Given a vector `indexes` of length NumAxes() which is a valid index
    for this RaggedShape, returns the integer offset for the element
    at that index (0 <= ans < NumElements()).  Note: will not work if
    this is on the GPU.
   */
  int32_t operator[](const std::vector<int32_t> &indexes);

  RaggedShapeIndexIterator Iterator();

  explicit RaggedShape(const std::vector<RaggedShapeDim> &axes,
                       bool check = true)
      : axes_(axes) {
    if (check) Check();
  }

  // A RaggedShape constructed this way will not be a valid RaggedShape.
  // The constructor is provided so you can immediately assign to it.
  RaggedShape() = default;

  // This makes sure that all of the row_splits, row_ids and cached_tot_size
  // are populated
  void Populate();

  RaggedShape(const RaggedShape &other) = default;
  RaggedShape &operator=(const RaggedShape &other) = default;

  // Axes() is intended for internal-ish use; users shouldn't really have to
  // interact with it.
  const std::vector<RaggedShapeDim> &Axes() const { return axes_; }

  // Check the RaggedShape for consistency; die on failure.
  void Check();

  // Convert to possibly different context.
  RaggedShape To(ContextPtr ctx) const;

 private:
  // TODO: could probably do away with the std::vector and have a max size and
  // a fixed length array (more efficient)

  // indexed by axis-index minus one... axis 0 is special, its dim
  // equals axes_[0].row_splits.Dim()-1.
  std::vector<RaggedShapeDim> axes_;
};

// prints a RaggedShape as e.g. [ [ 0 1 ] [ 2 ] [] ].  Note, the 'values'
// are just the positions in the array, this is for readability.
inline std::ostream &operator<<(std::ostream &stream,
                                const RaggedShape &shape) {
  // TODO: implement it
  return stream;
}

/*
  This is intended only for use in debugging.  It only works if the shape is
  on CPU.  You use it as:
    for (RaggedShapeIndexIterator iter = ragged.Iterator();
          !iter.Done(); iter.Next()) {
       const std::vector<int32_t> &vec = iter.Value();
       int32_t linear_index = ragged[vec];
    }
*/
class RaggedShapeIndexIterator {
 public:
  const std::vector<int32_t> &Value() const { return idx_; }
  void Next() {
    linear_idx_++;
    if (!Done()) UpdateVec();
  }
  bool Done() { return linear_idx_ == num_elements_; }

  explicit RaggedShapeIndexIterator(RaggedShape &shape)
      : shape_(shape),
        linear_idx_(0),
        idx_(shape.NumAxes()),
        num_elements_(shape.NumElements()) {
    K2_CHECK_EQ(shape_.Context()->GetDeviceType(), kCpu);
    for (int32_t i = 0; i + 1 < shape.NumAxes(); ++i) {
      row_splits_.push_back(shape.RowSplits(i + 1).Data());
      row_ids_.push_back(shape.RowIds(i + 1).Data());
    }
    if (!Done()) UpdateVec();
  }

 private:
  void UpdateVec() {
    K2_CHECK(!Done());
    int32_t idx = linear_idx_,
            num_axes = static_cast<int32_t>(row_splits_.size() + 1);
    for (int32_t axis = num_axes - 1; axis > 0; axis--) {
      int32_t prev_idx = row_ids_[axis - 1][idx],
              row_start = row_splits_[axis - 1][prev_idx],
              row_end = row_splits_[axis - 1][prev_idx + 1];
      K2_CHECK(idx >= row_start && idx < row_end);
      // e.g.: `idx` is an idx012, `prev_idx` is an idx01,
      //    `row_start` and `row_end` are idx01x, and
      //    this_idx is an idx2;
      int32_t this_idx = idx - row_start;
      idx_[axis] = this_idx;
      idx = prev_idx;
    }
    idx_[0] = idx;
  };
  std::vector<const int32_t *> row_splits_;
  std::vector<const int32_t *> row_ids_;
  RaggedShape &shape_;
  int32_t linear_idx_;
  std::vector<int32_t> idx_;
  const int32_t num_elements_;
};

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

template <typename T>
struct Ragged {
  RaggedShape shape;  // TODO: consider making the shape a pointer??
  Array1<T> values;

  Ragged(const RaggedShape &shape, const Array1<T> &values)
      : shape(shape), values(values) {
    K2_CHECK(IsCompatible(shape, values));
    K2_CHECK_EQ(shape.NumElements(), values.Dim());
  }

  // Default constructor will not leave this a valid Ragged object, you
  // shouldn't do anything with it.  Both members will be initialized with
  // default constructors.
  Ragged() = default;

  // Note: 'values' will be uninitialized.
  explicit Ragged(RaggedShape &shape)
      : shape(shape), values(shape.Context(), shape.NumElements()) {}

  // This will only work on the CPU, and is intended for use in testing code.
  T operator[](const std::vector<int32_t> &indexes) {
    K2_CHECK_EQ(Context()->GetDeviceType(), kCpu);
    return values[shape[indexes]];
  }

  ContextPtr &Context() const { return values.Context(); }
  int32_t NumAxes() const { return shape.NumAxes(); }

  /*
    It is an error to call this if this.shape.NumAxes() < 2.  This will return
    a Ragged<T> with one fewer axis, containing only the elements of
    *this for which the value on the provided axis is i.  CAUTION:
    currently this only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
   */
  Ragged<T> Index(int32_t axis, int32_t i) {
    // Note shape.Index(axis, i) will also check `axis` and `i` as below,
    // but we still check those requirements here in case
    // the implementation of shape.Index changes.
    // We may remove those checks finally.
    K2_CHECK_EQ(axis, 0);
    K2_CHECK_GE(i, 0);
    int32_t num_axes = shape.NumAxes();
    K2_CHECK_GE(num_axes, 2);
    const auto &axes = shape.Axes();
    K2_CHECK_LT(i + 1, axes[0].row_splits.Dim());

    // Get returned Ragged.shape
    RaggedShape sub_shape = shape.Index(axis, i);

    // Get returned Ragged.values' start and end position in this->values
    int32_t row_start = axes[0].row_splits[i];
    int32_t row_end = axes[0].row_splits[i + 1];
    for (int32_t i = 2; i < num_axes; ++i) {
      const Array1<int32_t> &row_splits = axes[i - 1].row_splits;
      row_start = row_splits[row_start];
      row_end = row_splits[row_end];
    }
    // Copy values
    ContextPtr c = Context();
    auto sub_values = Array1<T>(c, row_end - row_start);
    T *data = sub_values.Data();
    const T *src_data = values.Data();
    auto lambda_copy_values = [=] __host__ __device__(int32_t i) -> void {
      data[i] = src_data[i + row_start];
    };
    Eval(c, row_end - row_start, lambda_copy_values);
    return Ragged<T>(sub_shape, sub_values);
  }

  // Note *this is conceptually unchanged by this operation but non-const
  // because this->shape's row-ids may need to be generated.
  Ragged<T> RemoveAxis(int32_t axis) {
    K2_CHECK(axis >= 0 && axis < NumAxes());
    RaggedShape new_shape = ::k2::RemoveAxis(shape, axis);
    return Ragged<T>(new_shape, values);
  }

  Ragged<T> To(ContextPtr ctx) const {
    RaggedShape new_shape = shape.To(ctx);
    Array1<T> new_values = values.To(ctx);
    return Ragged<T>(new_shape, new_values);
  }

};

template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &r);

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
                      array to the input array. The caller
                      has to pre-allocate memory for it
                      on the same device as `src`.
 */
template <typename T, typename Op = LessThan<T>>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order);

}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_H_
#include "k2/csrc/ragged_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_H_

#endif  // K2_CSRC_RAGGED_H_
