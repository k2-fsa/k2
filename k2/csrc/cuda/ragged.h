// k2/csrc/cuda/ragged.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_RAGGED_H_
#define K2_CSRC_CUDA_RAGGED_H_

#include "k2/csrc/cuda/algorithms.h"
#include "k2/csrc/cuda/array.h"

namespace k2 {

// Caution, RaggedShapeDim is mostly for internal use and users should not
// generally interact with it directly.
// Note: row_splits is of size num_rows + 1 and row_ids is of size num_elements.
struct RaggedShapeDim {
  // Search for "row_splits concept" in utils.h for explanation
  Array1<int32_t> row_splits;
  // Search for "row_ids concept" in utils.h for explanation
  Array1<int32_t> row_ids;
  // cached_tot_size can be viewed as the number of elements in a ragged matrix,
  // or -1 if not known.
  // If cached_tot_size >= 0 and row_ids is nonempty, cached_tot_size will
  // equal row_ids.Dim().
  // If cached_tot_size >= 0 and row_splits is nonempty, cached_tot_size will
  // equal row_splits[row_splits.Dim() - 1].
  int32_t cached_tot_size;
};

class RaggedShape {
  int32_t Dim0() {
    CHECK_GT(0, axes_.size());
    return axes_[0].row_splits.Dim() - 1;
  }
  /* Return the  total size on this axis.  Requires 0 <= axis < NumAxes() and
     for axis=0 the returned value is the same as Dim0().  */
  inline int32_t TotSize(int32_t axis) {
    CHECK_LE(static_cast<size_t>(axis), axes_.size() + 1);
    if (axis == 0)
      return Dim0();
    else {
      RaggedShapeDim &rsd = axes_[axis - 1];
      if (rsd.cached_tot_size >= 0) {
        return rsd.cached_tot_size;
      } else {
        // if we had row_ids set up, we should have set cached_tot_size.
        CHECK_EQ(rsd.row_ids.Dim(), 0);
        CHECK_GT(rsd.row_splits.Dim(), 0);
        rsd.cached_tot_size = rsd.row_splits[rsd.row_splits.Dim() - 1];
        return rsd.cached_tot_size;
      }
    }
  }

  // Returns the number of elements that a ragged array with this shape would
  // have.
  inline int32_t NumElements() { return TotSize(NumAxes() - 1); }

  /*
    Return the row-splits for axis `axis` with `0 < axis < NumAxes()`.
    The dimension is the (total) number of rows on this axis plus one,
    and the elements are in the range [0,N] where N is the TotSize()
    on axis `axis+1`
   */
  Array1<int32_t> &RowSplits(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    // TODO(dan):: make sure this row_splits exists, create it if needed.
    return axes_[axis - 1].row_splits;
  }

  /*
    Return the row-ids for axis `axis` with `0 < axis < NumAxes()`.
    The dimension is the number of elements on this axis == TotSize(axis).
  */
  Array1<int32_t> &RowIds(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    // TODO(dan): make sure this row_ids exists, create it if needed.
    return axes_[axis - 1].row_ids;
  }

  /* Remove an axis by appending elements... require 0 <= axis < NumAxes() - 1
     and require NumAxes() > 2.  Effectively the axis is combined with the
     following axis, and the TotSize(axis) of `axis` in the returned shape will
     be the same as TotSize(axis + 1) in the current shape.  */
  RaggedShape RemoveAxis(int32_t axis);

  int32_t NumAxes() { return axes_.size() + 1; }

  // TODO.  Gives max size of any list on the provided axis, with 0 < axis <
  // NumAxes().  Equals max difference between successive row_splits on that
  // axis.
  int32_t MaxSize(int32_t axis);

  ContextPtr &Context() { return axes_[0].row_splits.Context(); }

  RaggedShape(const RaggedShape &other) = default;

  /*
    It is an error to call this if this.NumAxes() < 2.  This will return
    a RaggedShape with one fewer axis, containing only the elements of
    *this for which the value on axis `axis` is i.  CAUTION:
    currently this only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
   */
  RaggedShape Index(int32_t axis, int32_t value);

  RaggedShape ComposeRaggedShapes(RaggedShape &a, RaggedShape &b);

  RaggedShape(std::vector<RaggedShapeDim> &axes) : axes_(axes) {}


  // This makes sure that all of the row_splits, row_ids and cached_tot_size
  // are populated
  void Populate();

  // Axes() is intended for internal-ish use; users shouldn't really ahve to
  // interact with it.
  const std::vector<RaggedShapeDim> &Axes() const { return axes_; }
 private:
  // TODO: could probably do away with the std::vector and have a max size and a
  // fixed length array (more efficient)

  // indexed by axis-index minus one... axis 0 is special, its dim
  // equals axes_[0].row_splits.Dim()-1.
  std::vector<RaggedShapeDim> axes_;
};

/*
  Stack a list of RaggedShape to create a RaggedShape with one more axis.
  Similar to TF/PyTorch's Stack.  The result will have Dim0 == src_size.   All
  the source RaggedShapes must have the same NumAxes().

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
RaggedShape Stack(int32_t src_size, const RaggedShape **src, int32_t axis);

/*
  Insert a new axis at position `axis`, with 0 <= axis <= src.NumAxes(), for
  which the only allowed index will be 0 (which is another way of saying: all
  list sizes on that axis will be 1).

     @param [in] src   Source shape.  Row-ids and Row-splits of answer will
                      share memory with those in `src` (although it goes
                      without saying).
     @param [in] axis  Axis to insert.

  Note: you probably shouldn't be using this very often; if you are using this,
  you may be thinking in a PyTorch-y way but you should be relying on things
  like Eval() with custom lambdas more.  Read algorithms like in compose.cc to
  understand why.  Also: axis==0 is probably the only really useful case.
 */
RaggedShape Unsqueeze(const RaggedShape &src, int32_t axis);

/*
   Append a list of RaggedShape to form a single RaggedShape

      @param [in] num_srcs Number of source shapes to append
      @param [in] src      Array of sources to append
      @param [in] axis     Axis to append them on.   Previous axes must
                           have the same shape!   CAUTION: currently
                           we only support axis == 0.
      @return      Returns the appended RaggedShape.
*/
RaggedShape Append(int32_t num_srcs, RaggedShape **src, int32_t axis);

/*
    Gets an array of pointers to the row_splits of `src`, on the same
    device as `src`.
       @param [in] src  Source RaggedShape
       @return        Returns an array of size src.NumAxes() - 1 containing
                      pointers to the starts of the row_splits vetors
*/
Array<int32_t*> GetRowSpltsPtrs(RaggedShape &src);


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
RaggedShape Renumber(const RaggedShape &src, Array1<int32_t> &new2old);

template <typename T>
struct Ragged {
  RaggedShape shape;  // TODO: consider making the shape a pointer??
  Array1<T> values;

  Ragged(RaggedShape &shape, Array1<T> &values) : shape(shape), values(values) {
    CHECK_EQ(shape.TotSize(shape.NumAxes() - 1), values.Dim());
  }

  // Note: 'values' will be uninitialized.
  Ragged(RaggedShape &shape)
      : shape(shape),
        values(shape.Context(), shape.TotSize(shape.NumAxes() - 1)) {}

  ContextPtr Context() { return values.Context(); }

  /*
    It is an error to call this if this.NumAxes() < 2.  This will return
    a Ragged<T> with one fewer axis, containing only the elements of
    *this for which the value on the provided axis is i.  CAUTION:
    currently this only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
   */
  Ragged<T> Index(int32_t axis, int32_t value);
};

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
              CAUTION: only axis == 0 and axis == 1 are supported right now,
              and for the axis==1 case we have a requirement that all the
              src->Dim0() return the same value.

     @param [in] num_srcs  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes to be stacked
     @return  The appended result.

       Assuming as an example that the input had 3 axes: if axis==0, the result
       would have:
          result[i,j,k,l] = (*src[i])[j,k,l]
        and if axis==1 we would have:
          result[i,j,k,l] = (*src[j])[i,k,l]
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> **src);

/*
  Create a RaggedShape from an array of row-ids.  (which maps each element to
  its corresponding row).  The row-ids must be a nonempty vector, nonnegative
  and no-decreasing.

    @param [in]  num_rows   The number of rows (Size0()) of the object to be
                            created. If a value <= 0 is supplied, it will use
                            row_ids[-1]+1 if row_ids.size > 0, else 0.
    @param [in]  row_ids   The row-ids for axis 1; must be nonnegative
                           and non-decreasing.
 */
RaggedShape Ragged2ShapeFromRowIds(int num_rows, Array1<int32_t> &row_ids);

/*
  Construct a ragged shape with one more axis than the supplied shape, given
  row-ids for the last axis.

     @param [in] shape   The shape that will dictate the top-level axes of
                         the returned shape.
     @param [in] row_ids   A nondecreasing vector of integers
                           0 <= i < shape.TotSize(Shape.NumAxes()-1),
                           with row_ids.size() == elems.size().
 */
RaggedShape RaggedShapeFromRowIds(RaggedShape &shape, Array1<int> &row_ids);

/*
  Construct a RaggedShape with 2 axes.
     @param [in] row_splits   row_splits, or NULL (at least one of this and
  row_ids must be non-NULL).  Note: the dimension of row_splits must equal the
  number of rows plus one; row_splits[0] must be zero and the array must be
  non-decreasing; and the last element of row_splits is the total number of
  elements in the ragged matrix.
     @param [in] row_ids      row_splits, or NULL (at least one of this and
  row_ids must be non-NULL).  Note: the dimension of row_splits must equal the
  number of elements plus one; the array must be non-decreasing; and the last
  element of row_splits equals the number of rows.  If both row_ids and
  row_splits are supplied, we require row_splits[row_ids[i]] <= i <
  row_splits[row_ids[i]+1].
     @param [in] cached_tot_size   Total number of elements in the ragged
                              matrix, or -1 if the user does not wish to supply
  it now. (If >= 0, must equal `(*row_splits)[row_splits.Size()-1]` if
  row_splits is non-NULL, or row_ids->Dim()-1 if row_ids is non-NULL.
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
RaggedShape ComposeRaggedShapes(RaggedShape &a, RaggedShape &b);

/*
  Construct a RaggedShape with 3 axes.  For N=1 and 2 respectively:
  either row_splitsN or row_idsN or both must be non-NULL.
  If cached_tot_sizeN is not -1, it must equal the total size on
  that axis which will equal the last element of row_splitsN (if
  provided) and must equal the row_idsN.Dim(), if provided.
*/
RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2);

/*
  Allocates an *invalid* RaggedShape given the TotSize() values for each axis.
  This allocates space for all the row_ids and row_splits, but does not write
  any data to them.  View this as an internal function, because such a
  RaggedShape would not be usable directly.

     @input num_axes   Number of axes of ragged shape desired; must be at
                       least 2.
     @input tot_sizes  Total size on each axis
     @return           The returned RaggedShape satisfies
                       ans.TotSize(axis) == tot_sizes[axis], and has all
                       its row_splits and row_ids allocated but not
                       filled in, and its cached_tot_size elements
                       set.
 */
RaggedShape RaggedShapeFromTotSizes(int32_t num_axes, int32_t *tot_sizes);



// TODO(dan), include guard maybe.
#include "k2/csrc/cuda/ragged_inl.h"

}  // namespace k2

// TODO(dan), include guard maybe.
#include "k2/csrc/cuda/ragged_inl.h"

#endif  // K2_CSRC_CUDA_RAGGED_H_
