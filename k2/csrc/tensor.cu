/**
 * @brief
 * tensor
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <glog/logging.h>
#include <memory>

#include "k2/csrc/dtype.h"
#include "k2/csrc/tensor.h"

namespace k2 {

Shape::Shape(const std::vector<int32_t> &dims) : num_axes_(static_cast<int32_t>(dims.size())) {
  CHECK_LT(num_axes_, kMaxDim);

  std::copy(dims.begin(), dims.end(), dims_);

  // compute strides_
  if (num_axes_ > 0) {
    strides_[num_axes_ - 1] = 1;
  }
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
  CHECK_LT(num_axes_, kMaxDim);
  CHECK_EQ(strides.size(), num_axes_);
  std::copy(dims.begin(), dims.end(), dims_);
  std::copy(strides.begin(), strides.end(), strides_);
  num_element_ = ComputeNumElement();
  is_contiguous_ = CheckContiguous();
  storage_size_ = ComputeStorageSize();
}

int32_t Shape::ComputeNumElement() {
  if (num_axes_ == 0) {
    return 0;
  }
  int32_t elements = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    elements *= dims_[i];
  }
  return elements;
}

int32_t Shape::ComputeStorageSize() {
  if (num_axes_ == 0) {
    return 0;
  }
  int32_t size = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    size += (dims_[i] - 1) * strides_[i];
  }
  return size;
}

bool Shape::CheckContiguous() {
  int32_t z = 1;
  for (int32_t i = num_axes_ - 1; i >= 0; --i) {
    CHECK_GE(strides_[i], z);
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape) {
  impl_ = std::make_shared<TensorImpl>();
  impl_->dtype = type;
  impl_->shape = shape;
  Init(c);
}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims) {
  impl_ = std::make_shared<TensorImpl>();
  impl_->dtype = type;
  impl_->shape = Shape(dims);
  Init(c);
}

Tensor::Tensor(Dtype type, const Shape &shape, RegionPtr region,
               int32_t bytes_offset) {
  int32_t storage_size = shape.StorageSize();
  int32_t element_size = TraitsOf(type).NumBytes();
  impl_ = std::make_shared<TensorImpl>();
  impl_->dtype = type;
  impl_->shape = shape;
  impl_->data = region;
  impl_->bytes_offset = bytes_offset;
  CHECK_GE(impl_->data->num_bytes - impl_->bytes_offset,
           storage_size * element_size);
}

Tensor Tensor::Index(int32_t axis, int32_t index) const {
  const auto &this_shape = impl_->shape;
  CHECK_LT(axis, this_shape.NumAxes());
  CHECK_LT(index, this_shape.Dim(axis));
  std::vector<int32_t> dims(this_shape.Dims(),
                            this_shape.Dims() + this_shape.NumAxes());
  std::vector<int32_t> strides(this_shape.Strides(),
                               this_shape.Strides() + this_shape.NumAxes());
  dims.erase(dims.begin() + axis);
  strides.erase(strides.begin() + axis);
  Shape shape(dims, strides);
  int32_t bytes_offset =
      impl_->bytes_offset +
      index * this_shape.Stride(axis) * TraitsOf(impl_->dtype).NumBytes();
  return Tensor(impl_->dtype, shape, impl_->data, bytes_offset);
}

void Tensor::Init(ContextPtr c) {
  int32_t storage_size = impl_->shape.StorageSize();
  int32_t element_size = TraitsOf(impl_->dtype).NumBytes();
  impl_->data = NewRegion(c, static_cast<size_t>(storage_size * element_size));
  impl_->bytes_offset = 0;
}
Tensor Tensor::To(ContextPtr ctx) {
  return Tensor(std::shared_ptr(), kInt64Dtype, k2::Shape());
}

}  // namespace k2
