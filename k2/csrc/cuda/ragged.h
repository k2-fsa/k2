// k2/csrc/cuda/ragged.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_RAGGED_H_
#define K2_CSRC_CUDA_RAGGED_H_

#include "k2/csrc/cuda/ragged_shape.h"
#include "k2/csrc/cuda/algorithms.h"

namespace k2 {




class RaggedShape {
  int32_t Dim0() {
    CHECK_GT(0, axes_.size());
    return axes_[0].row_splits.Dim() - 1;
  }
  // total size on this axis (require 0 <= axis < NumAxes(), and for axis=0
  // is the same as Dim0()).
  int32_t TotSize(int32_t axis);

  Array1<int32_t> &RowSplits(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    return axes_[axis - 1].row_splits;
  }
  Array1<int32_t> &RowIds(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    // TODO: make sure this row_ids exists, create it if needed.
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



template <typename T> struct Ragged {
  RaggedShape shape; // TODO: consider making the shape a pointer??
  Array1<T> values;

  Ragged(RaggedShape &shape,
          Array1<T> &values): shape(shape), values(values) {
    // TODO: check values.Dim() matches shape.
  }

  Ragged(RaggedShape &shape):
      shape(shape),
      values(shape.Context(), shape.TotSize(shape.NumAxes()-1)) {
  }

  Context* Context() { return values.Context(); }
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
  Create a RaggedShape from an array of row-ids.  (which maps each element to
  its corresponding row).  The row-ids must be a nonempty vector, nonnegative
  and no-decreasing.

    @param [in]  num_rows   The number of rows (Size0()) of the object to be created.
                 If a value <= 0 is supplied, it will use row_ids[-1]+1
                 if row_ids.size > 0, else 0.
    @param [in]  row_ids   The row-ids for axis 1; must be nonnegative
                 and non-decreasing.
 */
RaggedShape Ragged2ShapeFromRowIds(int num_rows,
                                   const Array<int32_t> &row_ids);



/*
  Construct a ragged shape with one more axis than the supplied shape, given row-ids
  for the last axis.

     @param [in] shape   The shape that will dictate the top-level axes of
                       the returned shape.
     @param [in] row_ids   A nondecreasing vector of integers 0 <= i < shape.TotSize(Shape.NumAxes()-1),
                        with row_ids.size() == elems.size().
 */
RaggedShape RaggedShapeFromRowIds(const RaggedShape &shape,
                                  const Array<int> &row_ids);

/*
  Construct a RaggedShape with 3 axes.  For N=1 and 2 respectively:
  either row_splitsN or row_idsN or both must be non-NULL.
  If cached_tot_sizeN is not -1, it must equal the total size on
  that axis which will equal the last element of row_splitsN (if
  provided) and must equal the row_idsN.Dim(), if provided.
*/
RaggedShape RaggedShape3(Array<int32_t> *row_splits1,
                         Array<int32_t> *row_ids1,
                         int32_t cached_tot_size1,
                         Array<int32_t> *row_splits2,
                         Array<int32_t> *row_ids2,
                         int32_t cached_tot_size2);



}  // namespace k2

#endif  // K2_CSRC_CUDA_RAGGED_H_
