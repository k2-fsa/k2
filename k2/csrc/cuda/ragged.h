// k2/csrc/cuda/ragged.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_RAGGED_H_
#define K2_CSRC_CUDA_RAGGED_H_

#include "k2/csrc/cuda/algorithms.h"

namespace k2 {

class RaggedShape {
  int32_t Dim0() {
    CHECK_GT(0, axes_.size());
    return axes_[0].row_splits.Dim() - 1;
  }
  /* Return the  total size on this axis.  Requires 0 <= axis < NumAxes() and for axis=0
     the returned value is the same as Dim0().  */
  inline int32_t TotSize(int32_t axis) {
    CHECK_LE(static_cast<size_t>(axis), axes_.size() + 1);
    if (axis == 0) return Dim0();
    else {
      RaggedShapeDim &rsd = axes_[axis-1];
      if (rsd.cached_tot_size >= 0) {
        return rsd.cached_tot_size;
      } else {
        // if we had row_ids set up, we should have set cached_tot_size.
        CHECK_EQ(rsd.row_ids.Dim(), 0);
        CHECK_GT(rsd.row_splits.Dim(), 0);
        rsd.cached_tot_size = rsd.row_splits[rsd.row_splits.Dim()-1];
        return rsd.cached_tot_size;
      }
    }
  }

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
    The dimension is the number of elements on this axis PLUS ONE.
    (The last value is the number of row-splits on this axis).
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

 private:
  struct RaggedShapeDim {
    Array1<int32_t> row_splits;
    Array1<int32_t> row_ids;
    int32_t cached_tot_size;
  };

  // TODO: could probably do away with the std::vector and have a max size and a
  // fixed length array (more efficient)

  // indexed by axis-index minus one... axis 0 is special, its dim
  // equas axes_[0].row_splits.Dim()-1.
  std::vector<RaggedShapeDim> axes_;
};

/*
  Stack a list of RaggedShape to create a RaggedShape with one more axis.
  Similar to TF/PyTorch's Stack.  The result will have Dim0 == src_size.   All
  the source RaggedShapes must have the same NumAxes().


     @param [in] axis   The new axis whose dimension will equal src_size.
                        CAUTION: only axis == 0 and axis == 1 are supported
                        right now, and for the axis==1 case we have a
                        requirement that all the src->Dim0() return the
                        same value.
     @param [in] src_size  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes to be stacked

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
RaggedShape Stack(int32_t axis, int32_t src_size, const RaggedShape *src);

template <typename T>
struct Ragged {
  RaggedShape shape;  // TODO: consider making the shape a pointer??
  Array1<T> values;

  Ragged(RaggedShape &shape, Array1<T> &values) : shape(shape), values(values) {
    CHECK_EQ(shape.TotSize(shape.NumAxes()-1), values.Dim());
  }

  // Note: 'values' will be uninitialized.
  Ragged(RaggedShape &shape)
      : shape(shape),
        values(shape.Context(), shape.TotSize(shape.NumAxes() - 1)) {}

  Context *Context() { return values.Context(); }
};

/*
  Return ragged shape with only a subset of the bottom-level elements
  kept.  Require renumbering.NumOldElems() == src.TotSize(src.NumAxes()-1).
  Note: all dimensions and tot-sizes preceding that will remain the
  same, which might give rise to empty lists.
 */
RaggedShape SubsampleRaggedShape(const RaggedShape &src,
                                 Renumbering &renumbering);

/*
  Stack a list of Ragged arrays to create a Ragged array with one more axis.
  Similar to TF/PyTorch's Stack.  The result will have Dim0 == src_size.  All
  the source Ragged arrays' shapes must have the same NumAxes().

     @param [in] axis   The new axis whose dimension will equal src_size.
              CAUTION: only axis == 0 and axis == 1 are supported right now,
              and for the axis==1 case we have a requirement that all the
              src->Dim0() return the same value.

     @param [in] src_size  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes to be stacked
     @return  The appended result.

       Assuming as an example that the input had 3 axes: if axis==0, the result
       would have:
          result[i,j,k,l] = (*src[i])[j,k,l]
        and if axis==1 we would have:
          result[i,j,k,l] = (*src[j])[i,k,l]
 */
template <typename T>
Ragged<T> Stack(int32_t axis, int32_t src_size, const Ragged<T> *src);

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
RaggedShape Ragged2ShapeFromRowIds(int num_rows,
                                   const Array1<int32_t> &row_ids);

/*
  Construct a ragged shape with one more axis than the supplied shape, given
  row-ids for the last axis.

     @param [in] shape   The shape that will dictate the top-level axes of
                         the returned shape.
     @param [in] row_ids   A nondecreasing vector of integers
                           0 <= i < shape.TotSize(Shape.NumAxes()-1),
                           with row_ids.size() == elems.size().
 */
RaggedShape RaggedShapeFromRowIds(const RaggedShape &shape,
                                  const Array1<int> &row_ids);

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

}  // namespace k2

#endif  // K2_CSRC_CUDA_RAGGED_H_
