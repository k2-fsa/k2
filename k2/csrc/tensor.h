/**
 * @brief
 * tensor
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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

  int32_t Nelement() const { return num_element_; }
  // storage size in elements

  std::vector<int32_t> Dims() const {
    return std::vector<int32_t>(dims_, dims_ + num_axes_);
  }

  // Strides, in elements, not bytes.
  std::vector<int32_t> Strides() const {
    return std::vector<int32_t>(strides_, strides_ + num_axes_);
  }

  int64_t StorageSize() const { return storage_size_; }

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

  Shape() = default;

  explicit Shape(const std::vector<int32_t> &dims);

  explicit Shape(const std::vector<int32_t> &dims,
                 const std::vector<int32_t> strides);

  Shape(const Shape &other) = default;

 private:
  static const int32_t kMaxDim = 4;  // Will increase this as needed

  int32_t num_axes_ = 0;  // Must be >= 0
  int64_t num_element_ = 0;
  int64_t storage_size_ = 0;
  bool is_contiguous_ = true;

  // elements of dims_ and strides_ >= num_axes_ are currently not set;
  // in future we may change this.
  int32_t dims_[kMaxDim];
  int32_t strides_[kMaxDim];  // Strides in elements

  // compute the number of elements
  int64_t ComputeNumElement() const;
  // compute the size of storage needed to hold this tensor, in elements.
  // (different than ComputeNumElements(), because of strides).
  int64_t ComputeStorageSize() const;
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
  Tensor(Dtype type, const Shape &shape, RegionPtr region, int32_t byte_offset);

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
  inline bool SameDim(const Tensor &other) const {
    return impl_->shape.SameDims(other.GetShape());
  }
  inline int32_t NumAxes() const { return impl_->shape.NumAxes(); }
  inline int32_t Dim(int32_t i) const { return impl_->shape.Dim(i); }
  inline std::vector<int32_t> Dims() const { return impl_->shape.Dims(); }
  // Stride for axis i, in elements, not bytes.
  inline int32_t Stride(int32_t i) const { return impl_->shape.Stride(i); }
  // Strides, in elements, not bytes.
  inline std::vector<int32_t> Strides() const { return impl_->shape.Strides(); }
  inline int32_t Nelement() const { return impl_->shape.Nelement(); }
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

 private:
  void Init(ContextPtr c);
  explicit Tensor(TensorImplPtr impl);
  TensorImplPtr impl_;  // Must always be non-NULL.
};

// the primary declaration is in tensor_ops.h; included here to avoid
// compilation problems in array.h
Tensor ToContiguous(const Tensor &src);

}  // namespace k2
#endif  // K2_CSRC_TENSOR_H_
