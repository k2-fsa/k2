/**
 * @brief
 * tensor
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_TENSOR_H_
#define K2_CSRC_TENSOR_H_

#include <vector>
#include <memory>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"

namespace k2 {
class Shape {
 public:
  int32_t NumAxes() const { return num_axes_; }

  const int32_t *Dims() const { return dims_; }

  const int32_t *Strides() const { return strides_; }

  int32_t Dim(int32_t i) const {
    CHECK_LT(static_cast<uint32_t>(i), static_cast<uint32_t>(num_axes_));
    return dims_[i];
  }

  int32_t Stride(int32_t i) const {
    CHECK_LT(static_cast<uint32_t>(i), static_cast<uint32_t>(num_axes_));
    return strides_[i];
  }

  int32_t Nelement() const { return num_element_; }
  // storage size in elements
  int32_t StorageSize() const {
    return storage_size_;
  };
  bool IsContiguous() const { return is_contiguous_; }

  // Returns true if the two shapes have the same dims (but not necessarily
  // strides).
  bool SameDims(const Shape &other) const {
    if (num_axes_ != other.NumAxes()) return false;
    const int32_t *other_dims = other.Dims();
    for (int32_t i = 0; i != num_axes_; ++i) {
      if (dims_[i] != other_dims[i]) return false;
    }
    return true;
  }

  Shape() : num_axes_(0), num_element_(0), is_contiguous_(true) {}

  explicit Shape(const std::vector<int32_t> &dims);

  explicit Shape(const std::vector<int32_t> &dims,
                 const std::vector<int32_t> strides);

  Shape(const Shape &other) = default;

 private:
  static const int32_t kMaxDim = 4;  // Will increase this as needed

  int32_t num_axes_;  // Must be >= 0
  int32_t num_element_;
  int32_t storage_size_;
  bool is_contiguous_;

  // elements of dims_ and strides_ >= num_axes_ are currently not set;
  // in future we may change this.
  int32_t dims_[kMaxDim];
  int32_t strides_[kMaxDim];  // Strides in elements

  // compute the number of elements
  int32_t ComputeNumElement();
  int32_t ComputeStorageSize();
  bool CheckContiguous();
};

struct TensorImpl : public std::enable_shared_from_this<TensorImpl> {
  // This struct is not visible to the user and should be accessed via the
  // public
  // interface of Tensor.
  Shape shape;
  Dtype dtype;
  int32_t bytes_offset;
  // note: unlike Array1 and Array2, there is no support for data_ == nullptr,
  // i.e.  we will require that data_ is always allocated.  (This is because
  // we plan to generally hold Tensors as pointers, so there isn't much
  // need for an empty constructor).
  std::shared_ptr<Region> data;
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
  Tensor(Dtype type, const Shape &shape, RegionPtr region,
         int32_t bytes_offset);

  Tensor(Tensor &other) : impl_(other.impl_) {}

  // Returns pointer to elem with index all-zeros... will check that the type
  // matches the correct one.
  template <typename T>
  T *Data() {
    K2_CHECK(impl_->dtype == DtypeOf<T>::dtype);
    return reinterpret_cast<T *>(reinterpret_cast<char *>(impl_->data->data) +
                                 impl_->bytes_offset);
  }

  template <typename T>
  const T *Data() const {
    assert(impl_->dtype == DtypeOf<T>::dtype);
    return reinterpret_cast<const T *>(
        reinterpret_cast<char *>(impl_->data->data) + impl_->bytes_offset);
  }

  // Return the result of indexing one of the axes, which will result in a
  // Tensor with one fewer axis.
  Tensor Index(int32_t axis, int32_t index) const;

  // Assignment is shallow.
  Tensor &operator=(const Tensor &other) { impl_ = other.impl_; }

  Dtype GetDtype() const { return impl_->dtype; }
  const Shape &GetShape() const { return impl_->shape; }
  int32_t ByteOffset() const { return impl_->bytes_offset; }
  std::shared_ptr<Region> &GetRegion() { return impl_->data; }

  // Forward some funtions from the shape.  Will forward more later.
  inline bool SameDim(const Tensor &other) const {
    return impl_->shape.SameDims(other.GetShape());
  }
  inline bool NumAxes() const { return impl_->shape.NumAxes(); }
  inline int32_t Dim(int32_t i) { return impl_->shape.Dim(i); }
  inline int32_t Stride(int32_t i) { return impl_->shape.Stride(i); }
  inline int32_t Nelement(int32_t i) { return impl_->shape.Nelement(); }
  inline bool IsContiguous(const Tensor &other) {
    return impl_->shape.IsContiguous(other.impl_->shape);
  }

  /*
    Convert to possibly-different context, may require CPU/GPU transfer.
    The returned value may share the same underlying `data` memory as *this.
    This should work even for tensors with empty data.

    If dim_ == 0 and region_ is NULL, this will return a direct copy of *this
    (i.e.  with region_ also NULL)

    If dim == 0 and region_ is non-NULL, it will return a copy of *this with an
    empty region with the supplied context (if different from current region's
    context).

    Note: the answer will always be contiguous, i.e. there is a possibility that
    it will have a different memory layout than the input.  [Internally it will
    call `Contiguous()`.
  */
  Tensor To(ContextPtr ctx);

  ContextPtr GetContext() { return impl_->data->context; }

 private:
  TensorImplPtr impl_;  // Must always be non-NULL.

  void Init(ContextPtr c);
};

}  // namespace k2
#endif  // K2_CSRC_TENSOR_H_
