/**
 * @brief
 * tensor
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <memory>
#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {

Shape::Shape(const std::vector<int32_t> &dims)
    : num_axes_(static_cast<int32_t>(dims.size())) {
  K2_CHECK_LT(num_axes_, kMaxDim);

  std::copy(dims.begin(), dims.end(), dims_);

  // compute strides_
  if (num_axes_ > 0) strides_[num_axes_ - 1] = 1;

  for (int32_t i = num_axes_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * dims_[i + 1];
  }

  num_element_ = ComputeNumElement();
  is_contiguous_ = true;  // always be true here as we compute strides from dims
  storage_size_ = ComputeStorageSize();
}

Shape::Shape(const std::vector<int32_t> &dims,
             const std::vector<int32_t> strides)
    : num_axes_(static_cast<int32_t>(dims.size())) {
  K2_CHECK_LT(num_axes_, kMaxDim);
  K2_CHECK_EQ(static_cast<int32_t>(strides.size()), num_axes_);
  std::copy(dims.begin(), dims.end(), dims_);
  std::copy(strides.begin(), strides.end(), strides_);
  num_element_ = ComputeNumElement();
  is_contiguous_ = ComputeIsContiguous();
  storage_size_ = ComputeStorageSize();
}

int32_t Shape::ComputeNumElement() const {
  if (num_axes_ == 0) return 0;

  int32_t elements = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    elements *= dims_[i];
  }
  return elements;
}

int32_t Shape::ComputeStorageSize() const {
  if (num_axes_ == 0) return 0;

  int32_t size = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    size += (dims_[i] - 1) * strides_[i];
  }
  return size;
}

bool Shape::ComputeIsContiguous() const {
  int32_t z = 1;
  for (int32_t i = num_axes_ - 1; i >= 0; --i) {
    K2_CHECK_GE(strides_[i], z);
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape)
    : impl_(std::make_shared<TensorImpl>()) {
  impl_->dtype = type;
  impl_->shape = shape;
  Init(c);
}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims)
    : impl_(std::make_shared<TensorImpl>()) {
  impl_->dtype = type;
  impl_->shape = Shape(dims);
  Init(c);
}

Tensor::Tensor(Dtype type, const Shape &shape, RegionPtr region,
               int32_t byte_offset)
    : impl_(std::make_shared<TensorImpl>()) {
  int32_t storage_size = shape.StorageSize();
  int32_t element_size = TraitsOf(type).NumBytes();
  impl_->dtype = type;
  impl_->shape = shape;
  impl_->data = region;
  impl_->byte_offset = byte_offset;
  K2_CHECK_GE(impl_->data->num_bytes - impl_->byte_offset,
              storage_size * element_size);
}

Tensor Tensor::Index(int32_t axis, int32_t index) const {
  const auto &this_shape = impl_->shape;
  K2_CHECK_LT(axis, this_shape.NumAxes());
  K2_CHECK_LT(index, this_shape.Dim(axis));
  std::vector<int32_t> dims = this_shape.Dims();
  std::vector<int32_t> strides = this_shape.Strides();
  dims.erase(dims.begin() + axis);
  strides.erase(strides.begin() + axis);
  Shape shape(dims, strides);
  int32_t byte_offset =
      impl_->byte_offset +
      index * this_shape.Stride(axis) * TraitsOf(impl_->dtype).NumBytes();
  return Tensor(impl_->dtype, shape, impl_->data, byte_offset);
}

void Tensor::Init(ContextPtr c) {
  int32_t storage_size = impl_->shape.StorageSize();
  int32_t element_size = TraitsOf(impl_->dtype).NumBytes();
  impl_->data = NewRegion(c, static_cast<size_t>(storage_size * element_size));
  impl_->byte_offset = 0;
}

}  // namespace k2
