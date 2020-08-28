// k2/csrc/cuda/tensor.cu

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "glog/logging.h"
#include "k2/csrc/cuda/tensor.h"

namespace k2 {

Shape::Shape(const std::vector<int32_t> &dims)
    : dims_(dims), ndim_(dims.size()) {
  CHECK_LT(ndim_, kMaxDim);

  // compute strides_
  strides_.resize(ndim_);
  if (ndim_ > 0) {
    strides_[ndim_ - 1] = 1;
  }
  for (int32_t i = ndim_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * dims_[i + 1];
  }

  num_element_ = ComputeNumElement();
  is_contiguous_ = true;  // always be true as we :wqa
}

Shape::Shape(const std::vector<int32_t> &dims,
             const std::vector<int32_t> strides)
    : dims_(dims), strides_(strides), ndim_(dims.size()) {
  CHECK_LT(ndim_, kMaxDim);
  CHECK_EQ(strides_.size(), ndim_);
  num_element_ = ComputeNumElement();
  is_contiguous_ = CheckContiguous();
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

bool Shape::CheckContiguous() {
  int32_t z = 1;
  for (int32_t i = ndim_ - 1; i >= 0; --i) {
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape) {}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims) {}

Tensor::Tensor(const Shape &shape, Dtype dtype, RegionPtr region,
               size_t bytes_offset) {}

// TensorPtr Tensor::Index(int32_t axis, int32_t index) {}

}  // namespace k2
