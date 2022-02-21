/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_TENSOR_H_
#define K2_CSRC_TENSOR_H_

#include <memory>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/eval.h"
#include "k2/csrc/log.h"

namespace k2 {
class Shape {
 public:
  int32_t NumAxes() const { return num_axes_; }

  int32_t Dim(int32_t i) const {
    K2_CHECK_GE(i, 0);
    K2_CHECK_LT(i, num_axes_);
    return dims_[i];
  }

  // Stride for axis i, in elements, not bytes.
  int32_t Stride(int32_t i) const {
    K2_CHECK_GE(i, 0);
    K2_CHECK_LT(i, num_axes_);
    return strides_[i];
  }

  int32_t NumElements() const { return num_elements_; }
  // storage size in elements

  std::vector<int32_t> Dims() const {
    return std::vector<int32_t>(dims_, dims_ + num_axes_);
  }

  // Strides, in elements, not bytes.
  std::vector<int32_t> Strides() const {
    return std::vector<int32_t>(strides_, strides_ + num_axes_);
  }

  /*
    IsContiguous() has essentially the same meaning as in PyTorch, that
    strides[i] equals the product of dims[i] for j > i; however, we
    we allow strides[i] to have any value if dims[i] <= 1 (in this
    case strides[i] is a don't care.
   */
  bool IsContiguous() const { return is_contiguous_; }

  // Returns true if the two shapes have the same dims (but not necessarily
  // strides).
  bool SameDims(const Shape &other) const {
    if (num_axes_ != other.NumAxes()) return false;
    const int32_t *other_dims = other.dims_;
    for (int32_t i = 0; i != num_axes_; ++i) {
      if (dims_[i] != other_dims[i]) return false;
    }
    return true;
  }

  Shape(): num_axes_(0), num_elements_(1), is_contiguous_(true) { }

  explicit Shape(const std::vector<int32_t> &dims);

  explicit Shape(const std::vector<int32_t> &dims,
                 const std::vector<int32_t> strides);

  Shape(const Shape &other) = default;

  // Set stride on axis `axis`, with 0 <= axis < num_axes_.
  void SetStride(int32_t axis, int32_t stride);

  /*
    This function outputs the beginning and end of the range of elements
    reachable by this tensor.  (Note: this tensor type allows negative stride,
    so you cannot assume that begin == 0).

     @param [out] begin
                   At exit, *begin will be set to the most negative element
                   index reachable from this shape (or zero if this shape
                   contains no elements)
     @param [out] end   At exit, *end will be set to one plus the highest index
                  reachable from this shape (or to zero if this shape contains no
                  elements).
   */
  void GetReachableElems(int64_t *begin, int64_t *end) const;

  /*
    Be cautious with this function.  It returns the (end - begin) from
    calling GetReachableElems(), which is the smallest possible
    size of storage required to store this data, but because we support
    negative strides you cannot assume that the data starts at the
    beginning of the region.  Also, it is not cached.
   */
  int64_t StorageSize() const;

 private:
  static const int32_t kMaxDim = 4;  // Will increase this as needed

  int32_t num_axes_;  // Must be >= 0

  // num_elements_ is the number of distinct tuples of indexes; since strides
  // may be zero, we do not guarantee that all these elements occupy distinct
  // memory locations.  See NumElements()
  int64_t num_elements_;
  // see documentation for IsContiguous() for its meaning.  This is "derived
  // data"; it is computed by IsContiguous().
  bool is_contiguous_;

  // elements of dims_ and strides_ >= num_axes_ are currently not set;
  // in future we may change this.
  int32_t dims_[kMaxDim];
  int32_t strides_[kMaxDim];  // Strides in elements.  May be negative or zero.

  // compute the number of elements
  int64_t ComputeNumElements() const;
  bool ComputeIsContiguous() const;
};

std::ostream &operator<<(std::ostream &os, const Shape &shape);

struct TensorImpl : public std::enable_shared_from_this<TensorImpl> {
  // This struct is not visible to the user and should be accessed via the
  // public
  // interface of Tensor.
  Shape shape;
  Dtype dtype;
  size_t byte_offset;
  // note: unlike Array1 and Array2, there is no support for data_ == nullptr,
  // i.e.  we will require that data_ is always allocated.  (This is because
  // we plan to generally hold Tensors as pointers, so there isn't much
  // need for an empty constructor).
  RegionPtr data;
  TensorImpl() = default;
  TensorImpl(const Shape &shape, Dtype dtype, size_t byte_offset,
             RegionPtr data)
      : shape(shape), dtype(dtype), byte_offset(byte_offset), data(data) {}
};

using TensorImplPtr = std::shared_ptr<TensorImpl>;

/*
  Tensor is similar to PyTorch or TF Tensor.  Note, we don't use this that
  often: more often, we use templated types that make stronger assumptions about
  the dtype and layout, such as Array1, Array2 and Ragged.

  Note, it's allowable for some but not all of the dimensions to be zero,
  e.g. shapes like (0,4) are allowed.
 */
// TODO(haowen): we now only support positive strides
class Tensor {
 public:
  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const Shape &shape);

  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims);

  // Create Tensor backed by existing memory.
  Tensor(Dtype type, const Shape &shape, RegionPtr region, size_t byte_offset);

  Tensor(const Tensor &other) = default;
  Tensor &operator=(const Tensor &other) = default;

  // Returns pointer to elem with index all-zeros... will check that the type
  // matches the correct one.
  template <typename T>
  T *Data() {
    K2_CHECK_EQ(impl_->dtype, DtypeOf<T>::dtype);
    return reinterpret_cast<T *>(reinterpret_cast<char *>(impl_->data->data) +
                                 impl_->byte_offset);
  }

  template <typename T>
  const T *Data() const {
    K2_CHECK_EQ(impl_->dtype, DtypeOf<T>::dtype);
    return reinterpret_cast<const T *>(
        reinterpret_cast<char *>(impl_->data->data) + impl_->byte_offset);
  }

  void *Data() const {
    return reinterpret_cast<char *>(impl_->data->data) + impl_->byte_offset;
  }

  // Return the result of indexing one of the axes, which will result in a
  // Tensor with one fewer axis.
  Tensor Index(int32_t axis, int32_t index) const;

  Dtype GetDtype() const { return impl_->dtype; }
  const Shape &GetShape() const { return impl_->shape; }
  size_t ByteOffset() const { return impl_->byte_offset; }
  RegionPtr &GetRegion() const { return impl_->data; }

  // Forward some functions from the shape.  Will forward more later.
  inline bool SameDims(const Tensor &other) const {
    return impl_->shape.SameDims(other.GetShape());
  }
  inline int32_t NumAxes() const { return impl_->shape.NumAxes(); }
  inline int32_t Dim(int32_t i) const { return impl_->shape.Dim(i); }
  inline std::vector<int32_t> Dims() const { return impl_->shape.Dims(); }
  // Stride for axis i, in elements, not bytes.
  inline int32_t Stride(int32_t i) const { return impl_->shape.Stride(i); }
  // Strides, in elements, not bytes.
  inline std::vector<int32_t> Strides() const { return impl_->shape.Strides(); }
  inline int32_t NumElements() const { return impl_->shape.NumElements(); }
  inline bool IsContiguous() const { return impl_->shape.IsContiguous(); }
  inline int32_t ElementSize() const { return TraitsOf(GetDtype()).NumBytes(); }

  /*
    Convert to possibly-different context, may require CPU/GPU transfer.
    The returned value may share the same underlying `data` memory as *this.
    This should work even for tensors with empty data.


    Note: the answer will always be contiguous, i.e. there is a possibility that
    it will have a different memory layout than the input.  [Internally it will
    call `Contiguous()`.
  */
  Tensor To(ContextPtr ctx) const;

  // Return a contiguous tensor that does not share memory with this tensor.
  Tensor Clone() const;

  ContextPtr &Context() const { return impl_->data->context; }

  TensorImplPtr Impl() const { return impl_; }
  // This is for use by implementation code; be careful with it.
  explicit Tensor(TensorImplPtr impl);

 private:
  // For use when `shape` and `dtype` are already set up; sets data and
  // byte_offset.
  void Init(ContextPtr c);
  TensorImplPtr impl_;  // Must always be non-NULL.
};

// the primary declaration is in tensor_ops.h; included here to avoid
// compilation problems in array.h
Tensor ToContiguous(const Tensor &src);

}  // namespace k2
#endif  // K2_CSRC_TENSOR_H_
