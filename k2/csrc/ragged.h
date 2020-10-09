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
  const Array1<int32_t> &RowIds(int32_t axis) const;

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
  void Check() { K2_CHECK(Validate(true)); }

  // Validate the RaggedShape; on failure will return false (may also
  // print warnings).
  bool Validate(bool print_warnings = true);

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
  const Array1<int32_t> &RowSplits(int32_t axis) const {
    return shape.RowSplits(axis);
  }
  Array1<int32_t> &RowSplits(int32_t axis) { return shape.RowSplits(axis); }
  const Array1<int32_t> &RowIds(int32_t axis) const {
    return shape.RowIds(axis);
  }
  Array1<int32_t> &RowIds(int32_t axis) { return shape.RowIds(axis); }
  int32_t TotSize(int32_t axis) const { return shape.TotSize(axis); }
  int32_t Dim0() const { return shape.Dim0(); }
  bool Validate(bool print_warnings = true);

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
  // This function is defined in ragged_ops_inl.h.
  Ragged<T> RemoveAxis(int32_t axis);

  Ragged<T> To(ContextPtr ctx) const {
    RaggedShape new_shape = shape.To(ctx);
    Array1<T> new_values = values.To(ctx);
    return Ragged<T>(new_shape, new_values);
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &r);

}  // namespace k2

#endif  // K2_CSRC_RAGGED_H_
