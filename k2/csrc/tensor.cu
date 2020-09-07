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
#include <memory>
#include <vector>

#include "k2/csrc/dtype.h"
#include <glog/logging.h>
#include "k2/csrc/tensor.h"

namespace k2 {

Shape::Shape(const std::vector<int32_t> &dims) : ndim_(dims.size()) {
  CHECK_LT(ndim_, kMaxDim);

  std::copy(dims.begin(), dims.end(), dims_);

  // compute strides_
  if (ndim_ > 0) {
    strides_[ndim_ - 1] = 1;
  }
  for (int32_t i = ndim_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * dims_[i + 1];
  }

  num_element_ = ComputeNumElement();
  is_contiguous_ = true;  // always be true here as we compute strides from dims
  storage_size_ = ComputeStorageSize();
}

Shape::Shape(const std::vector<int32_t> &dims,
             const std::vector<int32_t> strides)
    : ndim_(dims.size()) {
  CHECK_LT(ndim_, kMaxDim);
  CHECK_EQ(strides.size(), ndim_);
  std::copy(dims.begin(), dims.end(), dims_);
  std::copy(strides.begin(), strides.end(), strides_);
  num_element_ = ComputeNumElement();
  is_contiguous_ = CheckContiguous();
  storage_size_ = ComputeStorageSize();
}

int32_t Shape::ComputeNumElement() {
  if (ndim_ == 0) {
    return 0;
  }
  int32_t elements = 1;
  for (int32_t i = 0; i < ndim_; ++i) {
    elements *= dims_[i];
  }
  return elements;
}

int32_t Shape::ComputeStorageSize() {
  if (ndim_ == 0) {
    return 0;
  }
  int32_t size = 1;
  for (int32_t i = 0; i < ndim_; ++i) {
    size += (dims_[i] - 1) * strides_[i];
  }
  return size;
}

bool Shape::CheckContiguous() {
  int32_t z = 1;
  for (int32_t i = ndim_ - 1; i >= 0; --i) {
    CHECK_GE(strides_[i], z);
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape)
    : dtype_(type), shape_(shape) {
  Init(c);
}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims)
    : dtype_(type), shape_(dims) {
  Init(c);
}

Tensor::Tensor(Dtype type, const Shape &shape, RegionPtr region,
               int32_t bytes_offset)
    : dtype_(type), shape_(shape), data_(region), bytes_offset_(bytes_offset) {
  int32_t storage_size = shape_.StorageSize();
  int32_t element_size = TraitsOf(dtype_).NumBytes();
  CHECK_GE(data_->num_bytes - bytes_offset_, storage_size * element_size);
}

TensorPtr Tensor::Index(int32_t axis, int32_t index) const {
  CHECK_LT(axis, shape_.Ndim());
  CHECK_LT(index, shape_.Dim(axis));
  std::vector<int32_t> dims(shape_.Dims(), shape_.Dims() + shape_.Ndim());
  std::vector<int32_t> strides(shape_.Strides(),
                               shape_.Strides() + shape_.Ndim());
  dims.erase(dims.begin() + axis);
  strides.erase(strides.begin() + axis);
  Shape shape(dims, strides);
  int32_t bytes_offset =
      bytes_offset_ + index * shape_.Stride(axis) * TraitsOf(dtype_).NumBytes();
  return std::make_shared<Tensor>(dtype_, shape, data_, bytes_offset);
}

void Tensor::Init(ContextPtr c) {
  int32_t storage_size = shape_.StorageSize();
  int32_t element_size = TraitsOf(dtype_).NumBytes();
  data_ = NewRegion(c, storage_size * element_size);
  bytes_offset_ = 0;
}

}  // namespace k2
