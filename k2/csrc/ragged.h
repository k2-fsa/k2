/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
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

#ifndef K2_CSRC_RAGGED_H_
#define K2_CSRC_RAGGED_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

// Caution, RaggedShapeLayer is mostly for internal use and users should not
// generally interact with it directly.  A layer represents the connection
// between one axis and the next; a RaggedShape with a single layer is the
// minimal RaggedShape.
//
// Note: row_splits is of size num_rows + 1 and row_ids is of size
// num_elements.
struct RaggedShapeLayer {
  RaggedShapeLayer() = default;

  RaggedShapeLayer(const RaggedShapeLayer &) = default;
  RaggedShapeLayer& operator=(const RaggedShapeLayer &) = default;

  RaggedShapeLayer(RaggedShapeLayer &&) = default;
  RaggedShapeLayer& operator=(RaggedShapeLayer &&) = default;

  // Search for "row_splits concept" in utils.h for explanation.  row_splits
  // is required; it must always be nonempty for a RaggedShapeLayer to be valid.
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
class RaggedShape;
// Will write the elements as x, e.g. "[ [ x x ] [x] ]"
std::ostream &operator<<(std::ostream &stream, const RaggedShape &shape);

// Reader from string, expects "x" (i.e. the letter x) for the elements, e.g. "[
// [ x x ] [ x x x ] ]".  The spaces are optional.  Will crash if the input was
// invalid (e.g. mismatched brackets or inconsistent depth).
std::istream &operator>>(std::istream &stream, RaggedShape &shape);

class RaggedShape {
 public:
  int32_t Dim0() const {
    K2_CHECK_GT(layers_.size(), 0);
    return layers_[0].row_splits.Dim() - 1;
  }
  /* Return the  total size on this axis.  Requires 0 <= axis < NumAxes() and
     for axis=0 the returned value is the same as Dim0().
     Caution: we use const_cast inside this function as it may actually modify
     the cached_tot_size members of RaggedShapeLayer if not set.
  */
  int32_t TotSize(int32_t axis) const;

  /* Append `other` to `*this` (in-place version that modifies `*this`).
     `other` must have the same number of axes as `this`.  This is efficient in
     an amortized way, i.e. should take time/work that's O(n) in the size of
     `other`, not `*this`, if you make many calls to Append().  This is due to
     the policy used in Region::Extend(), where it at least doubles the size
     each time, similar to std::vector.
     Be very careful with this, because many operations on Ragged tensors will
     silently share memory (there is normally an implicit assumption that
     the row_splits and row_indexes are constant).
  */
  void Append(const RaggedShape &other);

  // Returns the number of elements that a ragged array with this shape would
  // have.
  int32_t NumElements() const {
    return TotSize(NumAxes() - 1);
  }

  /*
    Return the row-splits for axis `axis` with `0 < axis < NumAxes()`.
    The dimension is the (total) number of rows on this axis plus one,
    and the elements are in the range [0,N] where N is the TotSize()
    on axis `axis+1`
   */
  Array1<int32_t> &RowSplits(int32_t axis) {
    K2_CHECK_GT(axis, 0);
    K2_CHECK_LT(axis, NumAxes());
    // Note row_splits is always nonempty for valid RaggedShapeLayer.
    return layers_[axis - 1].row_splits;
  }

  const Array1<int32_t> &RowSplits(int32_t axis) const {
    K2_CHECK_GT(axis, 0);
    K2_CHECK_LT(axis, NumAxes());
    // Note row_splits is always nonempty for valid RaggedShapeLayer.
    return layers_[axis - 1].row_splits;
  }

  /*
    Return the row-ids for axis `axis` with `0 < axis < NumAxes()`.
   The dimension is the number of elements on this axis == TotSize(axis).
  */
  Array1<int32_t> &RowIds(int32_t axis);
  const Array1<int32_t> &RowIds(int32_t axis) const {
    return const_cast<RaggedShape *>(this)->RowIds(axis);
  }

  int32_t NumAxes() const { return static_cast<int32_t>(layers_.size()) + 1; }

  int32_t NumLayers() const { return static_cast<int32_t>(layers_.size()); }

  // Gives max size of any list on the provided axis,
  // with 0 < axis < NumAxes().  Equals max difference between successive
  // row_splits on that axis.
  int32_t MaxSize(int32_t axis);

  ContextPtr &Context() const { return layers_[0].row_splits.Context(); }

  /*
    It is an error to call this if this.NumAxes() <= 2.  This will return
    a RaggedShape with one fewer axis, containing only the elements of
    *this for which the value on axis `axis` is i.  CAUTION:
    currently this only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
      @param [out] value_offset   If non-NULL, the offset into the
                         values necessary to take the needed sub-part
                         of the data will be written to here.
   */
  RaggedShape Index(int32_t axis, int32_t i,
                    int32_t *value_offset = nullptr) const;

  /*
    Given a vector `indexes` of length NumAxes() which is a valid index
    for this RaggedShape, returns the integer offset for the element
    at that index (0 <= ans < NumElements()).  Note: will not work if
    this is on the GPU.
   */
  int32_t operator[](const std::vector<int32_t> &indexes);

  RaggedShapeIndexIterator Iterator();

  // TODO(dan): will at some point make it so check = false is the default.
  explicit RaggedShape(const std::vector<RaggedShapeLayer> &layers,
                       bool check = !internal::kDisableDebug)
      : layers_(layers) {
    // the check can be disabled by setting the environment variable
    // K2_DISABLE_CHECKS.
    if (check && !internal::DisableChecks()) Check();
  }

  explicit RaggedShape(const std::string &src) {
    std::istringstream is(src);
    is >> *this >> std::ws;
    if (!is.eof() || is.fail())
      K2_LOG(FATAL) << "Failed to construct RaggedShape from string: " << src;
  }

  // Construct from context and string.  This uses delegating constructors, (a
  // c++11 feature), and an explicitly constructed RaggedShape
  // "RaggedShape(src)"
  RaggedShape(ContextPtr context, const std::string &src)
      : RaggedShape(RaggedShape(src).To(context)) {}

  // A RaggedShape constructed this way will not be a valid RaggedShape.
  // The constructor is provided so you can immediately assign to it.
  RaggedShape() = default;

  // This makes sure that all of the row_splits, row_ids and cached_tot_size
  // are populated
  void Populate();

  RaggedShape(const RaggedShape &) = default;
  RaggedShape &operator=(const RaggedShape &) = default;

  RaggedShape(RaggedShape &&) = default;
  RaggedShape &operator=(RaggedShape &&) = default;

  // Layers() is intended for internal-ish use; users shouldn't really have to
  // interact with it.
  const std::vector<RaggedShapeLayer> &Layers() const { return layers_; }
  // CAUTION: you probably shouldn't use this unless you really know what you
  // are doing.
  std::vector<RaggedShapeLayer> &Layers() { return layers_; }

  // Check the RaggedShape for consistency; die on failure.
  void Check() const;

  /*
    Copy to a possibly different device. If `copy_all == true`, will copy the
    row_ids rather than reconstructing it on the dest device; this is useful for
    debug.
   */
  RaggedShape To(ContextPtr ctx, bool copy_all = false) const;

 private:
  // TODO: could probably do away with the std::vector and have a max size and
  // a fixed length array (more efficient)

  // indexed by axis-index minus one... axis 0 is special, its dim
  // equals layers_[0].row_splits.Dim()-1.
  std::vector<RaggedShapeLayer> layers_;
};

template <typename T, int MAX_DIM>
struct SmallVec {
  T data[MAX_DIM];
  __host__ __device__ T operator()(int32_t i) const {
    return data[i];
  }
};

// call this variable `xxx_row_splits_acc`
template <int MAX_LAYERS>
struct RowSplitsAccessor {
  int32_t *ptrs[MAX_LAYERS] = {nullptr};  // these are indexed by layer, from 0.

  // row_splits_acc(0) == shape.RowSplits(1), for instance.
  __host__ __device__ int32_t *operator()(int32_t layer) const {
    return ptrs[layer];
  }

  explicit RowSplitsAccessor(RaggedShape &src);
};

// call this variable `xxx_row_ids_acc`
template <int MAX_LAYERS>
struct RowIdsAccessor {
  int32_t *ptrs[MAX_LAYERS] = {nullptr};  // these are indexed by layer, from 0.
  // row_ids_acc(0) == shape.RowIds(1), for instance.
  __host__ __device__ int32_t *operator()(int32_t layer) const {
    return ptrs[layer];
  }

  explicit RowIdsAccessor(RaggedShape &src);
};

// prints a RaggedShape, for debug purposes.  May change later how this works.
std::ostream &operator<<(std::ostream &stream, const RaggedShape &shape);

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
  bool Done() const { return linear_idx_ == num_elements_; }

  explicit RaggedShapeIndexIterator(const RaggedShape &shape)
      : linear_idx_(0),
        idx_(shape.NumAxes()),
        num_elements_(shape.NumElements()) {
    K2_CHECK_EQ(shape.Context()->GetDeviceType(), kCpu);
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

  explicit Ragged(const RaggedShape &shape, Dtype dtype = DtypeOf<T>::dtype)
      : shape(shape), values(shape.Context(), shape.NumElements(), dtype) {}

  // Defined in ragged_ops_inl.h
  // This will crash if T == Any.
  explicit Ragged(const std::string &src, bool throw_on_failure = false) {
    std::istringstream is(src);
    is >> *this >> std::ws;
    if (!is.eof() || is.fail()) {
      std::ostringstream os;
      os << "Failed to construct Ragged array from string: " << src;
      if (throw_on_failure) {
        throw std::runtime_error(os.str());
      } else {
        K2_LOG(FATAL) << os.str();
      }
    }
  }

  // Construct from context and string.  This uses delegating constructors,
  // (a c++11 feature), and an explicitly constructed Ragged<T>
  // "Ragged<T>(src)"
  Ragged(ContextPtr context, const std::string &src)
      : Ragged(Ragged<T>(src).To(context)) {}

  // Default constructor will not leave this a valid Ragged object, you
  // shouldn't do anything with it.  Both members will be initialized with
  // default constructors.
  Ragged() = default;

  Ragged &operator=(const Ragged<T> &src) = default;
  Ragged(const Ragged<T> &src) = default;
  // Move constructor
  Ragged(Ragged<T> &&src) = default;
  Ragged &operator=(Ragged<T> &&src) = default;

  // This will only work on the CPU, and is intended for use in testing code.
  // See also member-function Index().
  T operator[](const std::vector<int32_t> &indexes) {
    K2_CHECK_EQ(Context()->GetDeviceType(), kCpu);
    return values[shape[indexes]];
  }

  ContextPtr &Context() const { return values.Context(); }
  int32_t NumAxes() const { return shape.NumAxes(); }
  int32_t NumElements() const { return shape.NumElements(); }
  const Array1<int32_t> &RowSplits(int32_t axis) const {
    return shape.RowSplits(axis);
  }
  Dtype GetDtype() const { return values.GetDtype(); }
  Array1<int32_t> &RowSplits(int32_t axis) { return shape.RowSplits(axis); }
  const Array1<int32_t> &RowIds(int32_t axis) const {
    return shape.RowIds(axis);
  }
  Array1<int32_t> &RowIds(int32_t axis) { return shape.RowIds(axis); }
  int32_t TotSize(int32_t axis) const { return shape.TotSize(axis); }
  int32_t Dim0() const { return shape.Dim0(); }
  // Validates ragged shape; crashes if there is a problem.  Note: the error
  // message may appear later if we are not syncing kernel so the stack trace
  // won't be informative. You would need to set the environment variable
  // K2_SYNC_KERNELS=1 before running, to get the correct stack trace and
  // for use in a debugger.
  void Check() const;

  template <typename U>
  Ragged<U> &Specialize() {
    static_assert(std::is_same<T, Any>::value,
                  "generic arrays not supported here");
    K2_CHECK_EQ(values.GetDtype(), DtypeOf<U>::dtype);
    return *reinterpret_cast<Ragged<U>*>(this);
  }
  template <typename U>
  const Ragged<U> &Specialize() const {
    static_assert(std::is_same<T, Any>::value,
                  "generic arrays not supported here");
    K2_CHECK_EQ(values.GetDtype(), DtypeOf<U>::dtype);
    return *reinterpret_cast<const Ragged<U>*>(this);
  }

  // Return a reference to this viewed as `Any` type (for when we
  // want a generic array without type information).
  // Reverse this with Specialize().
  Ragged<Any> &Generic() {
    return *reinterpret_cast<Ragged<Any>*>(this);
  }
  const Ragged<Any> &Generic() const {
    return *reinterpret_cast<const Ragged<Any>*>(this);
  }


  /*
    It is an error to call this if this.shape.NumAxes() <= 2.  This will return
    a Ragged<T> with one fewer axis, containing only the elements of
    *this for which the value on the provided axis is i; it will share
    the underlying data with `*this` where possible. CAUTION: currently this
    only works for `axis == 0`.

      @param [in]  axis   Axis to index on.  CAUTION: currently only 0
                         is supported.
      @param [in]  i     Index to select
   */
  Ragged<T> Index(int32_t axis, int32_t i) const {
    // Get returned Ragged.shape
    int32_t values_offset;
    RaggedShape sub_shape = shape.Index(axis, i, &values_offset);
    return Ragged<T>(sub_shape,
                     values.Range(values_offset, sub_shape.NumElements()));
  }

  /*
    Return a version of `*this` with one axis removed, done by appending
    lists (this axis is combined with the following axis).  Effectively removes
    element numbered `axis` from the vector of tot_sizes `[ src.TotSize(0),
    src.TotSize(1), ... src.TotSize(axis - 1) ]`

    Note *this is conceptually unchanged by this operation but non-const
    because this->shape's row-ids may need to be generated.
    This function is defined in ragged_ops_inl.h.

        @param [in] axis  Axis to remove.  Requires 0 <= axis < NumAxes() - 1.
        @return  Returns the modified ragged tensor, which will share the same
            `values` and some of the same shape metdata as `*this`.
  */
  Ragged<T> RemoveAxis(int32_t axis);

  /*
    Copy to a possibly different device. If `copy_all == true`, will copy
    the cached_tot_size and row_ids rather than reconstructing them on the dest
    device; this is useful for debug.
   */
  Ragged<T> To(ContextPtr ctx, bool copy_all = false) const {
    RaggedShape new_shape = shape.To(ctx, copy_all);
    Array1<T> new_values = values.To(ctx);
    return Ragged<T>(new_shape, new_values);
  }

  // The ToType() macro will be expanded to
  //
  //  Ragged<int32_t> ToInt() const;
  //  Ragged<float> ToFloat() const;
  //  Ragged<double> ToDouble() const;
  //  Ragged<int64_t> ToLong() const;
  //
  // which is roughly equivalent to the following template
  //
  //  template<typename U>
  //  Ragged<U> To() const;
  //
  // The purpose is to convert Ragged<T> to Ragged<U>, e.g.,
  // convert Ragged<int32_t> to Ragged<float>.
  //
  // If T == U, then the Ragged itself is returned; otherwise,
  // a new Ragged is returned.
  //
#define ToType(type, name)                                  \
  Ragged<type> To##name() const {                           \
    Array1<type> new_values = values.To##name();            \
    return Ragged<type>(shape, new_values);                 \
  }
ToType(float, Float)
ToType(double, Double)
ToType(int32_t, Int)
ToType(int64_t, Long)
#undef ToType

  // There is no need to clone the shape because it's a kind of convention
  // that Array1's that are the row_ids or row_splits of a Ragged object are
  // not mutable so they can be re-used.
  Ragged<T> Clone() const { return Ragged<T>(shape, values.Clone()); }

  // Convert a ragged tensor with 2 axes into a vector of vector.
  //
  // CAUTION: this->NumAxes() must be 2.
  std::vector<std::vector<T>> ToVecVec() const {
    K2_CHECK_EQ(NumAxes(), 2);
    if (Context()->GetDeviceType() == kCuda) {
      return this->To(GetCpuContext()).ToVecVec();
    }
    int32_t dim0 = this->Dim0();
    std::vector<std::vector<T>> ans(dim0);
    const int32_t *row_splits_data = RowSplits(1).Data();
    const T *values_data = values.Data();
    for (int32_t i = 0; i != dim0; ++i) {
      int32_t len = row_splits_data[i + 1] - row_splits_data[i];
      ans[i].resize(len);
      std::copy(values_data + row_splits_data[i],
                values_data + row_splits_data[i + 1], ans[i].begin());
    }
    return ans;
  }
};

// e.g. will produce something like "[ [ 3 4 ] [ 1 ] ]".
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &r);

// caution: when reading "[ ]" it will assume 2 axes.
// This is defined in ragged_ops_inl.h.
template <typename T>
std::istream &operator>>(std::istream &stream, Ragged<T> &r);

}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_H_
#include "k2/csrc/ragged_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_H_

#endif  // K2_CSRC_RAGGED_H_
