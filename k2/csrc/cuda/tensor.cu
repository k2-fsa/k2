// k2/csrc/cuda/tensor.cu

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/cuda/dtype.h"
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
  is_contiguous_ = true;  // always be true here as we compute strides from dims
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

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape)
    : dtype_(type), shape_(shape) {
  Init(c);
}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims)
    : dtype_(type), shape_(dims) {
  Init(c);
}

Tensor::Tensor(Dtype type, const Shape &shape, RegionPtr region,
               size_t bytes_offset)
    : dtype_(type), shape_(shape), data_(region), bytes_offset_(bytes_offset) {}

TensorPtr Tensor::Index(int32_t axis, int32_t index) const {
  CHECK_LT(axis, shape_.Ndim());
  CHECK_LT(index, shape_.Dim(axis));
  auto dims = shape_.Dims();
  auto strides = shape_.Strides();
  dims.erase(dims.begin() + axis);
  strides.erase(strides.begin() + axis);
  Shape shape(dims, strides);
  int32_t bytes_offset =
      index * shape_.Stride(axis) * TraitsOf(dtype_).NumBytes();
  return std::make_shared<Tensor>(dtype_, shape, data_, bytes_offset);
}

void Tensor::Init(ContextPtr c) {
  int32_t nelement = shape_.Nelement();
  int32_t element_size = TraitsOf(dtype_).NumBytes();
  data_ = NewRegion(c, nelement * element_size);
  bytes_offset_ = 0;
}

}  // namespace k2
