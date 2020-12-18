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
#include <string>
#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {

Shape::Shape(const std::vector<int32_t> &dims)
    : num_axes_(static_cast<int32_t>(dims.size())) {
  NVTX_RANGE(K2_FUNC);
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
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_LT(num_axes_, kMaxDim);
  K2_CHECK_EQ(static_cast<int32_t>(strides.size()), num_axes_);
  std::copy(dims.begin(), dims.end(), dims_);
  std::copy(strides.begin(), strides.end(), strides_);
  num_element_ = ComputeNumElement();
  is_contiguous_ = ComputeIsContiguous();
  storage_size_ = ComputeStorageSize();
}

int64_t Shape::ComputeNumElement() const {
  NVTX_RANGE(K2_FUNC);
  if (num_axes_ == 0) return 0;

  int64_t elements = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    elements *= dims_[i];
  }
  return elements;
}

int64_t Shape::ComputeStorageSize() const {
  NVTX_RANGE(K2_FUNC);
  if (num_axes_ == 0) return 0;

  int64_t size = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    size += (dims_[i] - 1) * (int64_t)strides_[i];
  }
  K2_CHECK_GE(size, 0);
  return size;
}

bool Shape::ComputeIsContiguous() const {
  NVTX_RANGE(K2_FUNC);

  // It may happen that all strides are zero,
  // i.e., the tensor contains only one element.
  // In this case, the tensor is contiguous.
  int32_t s = 0;
  for (int32_t i = num_axes_ - 1; i >= 0; --i) {
    K2_CHECK_GE(strides_[i], 0);
    s += strides_[i];
  }
  if (s == 0) return true;

  int64_t z = 1;
  for (int32_t i = num_axes_ - 1; i >= 0; --i) {
    K2_CHECK_GE(strides_[i], z);
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

std::ostream &operator<<(std::ostream &os, const Shape &shape) {
  os << "num_axes: " << shape.NumAxes() << "\n";
  os << "dims: ";
  std::string sep;
  for (int32_t i = 0; i != shape.NumAxes(); ++i) {
    os << sep << shape.Dim(i);
    sep = " ";
  };
  os << "\n";
  os << "strides: ";
  sep = "";
  for (int32_t i = 0; i != shape.NumAxes(); ++i) {
    os << sep << shape.Stride(i);
    sep = " ";
  }
  os << "\n";
  return os;
}

Tensor::Tensor(ContextPtr c, Dtype type, const Shape &shape)
    : impl_(std::make_shared<TensorImpl>()) {
  NVTX_RANGE(K2_FUNC);
  impl_->dtype = type;
  impl_->shape = shape;
  Init(c);
}

Tensor::Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims)
    : impl_(std::make_shared<TensorImpl>()) {
  NVTX_RANGE(K2_FUNC);
  impl_->dtype = type;
  impl_->shape = Shape(dims);
  Init(c);
}

Tensor::Tensor(Dtype type, const Shape &shape, RegionPtr region,
               int32_t byte_offset)
    : impl_(std::make_shared<TensorImpl>()) {
  NVTX_RANGE(K2_FUNC);
  size_t storage_size = shape.StorageSize();
  size_t element_size = TraitsOf(type).NumBytes();
  impl_->dtype = type;
  impl_->shape = shape;
  impl_->data = region;
  impl_->byte_offset = byte_offset;
  K2_CHECK_GE(impl_->data->num_bytes - impl_->byte_offset,
              storage_size * element_size);
}

Tensor Tensor::Index(int32_t axis, int32_t index) const {
  NVTX_RANGE(K2_FUNC);
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
  NVTX_RANGE(K2_FUNC);
  int32_t storage_size = impl_->shape.StorageSize();
  int32_t element_size = TraitsOf(impl_->dtype).NumBytes();
  impl_->data = NewRegion(c, static_cast<size_t>(storage_size * element_size));
  impl_->byte_offset = 0;
}

Tensor::Tensor(TensorImplPtr impl) : impl_(impl) {}

Tensor Tensor::To(ContextPtr ctx) const {
  NVTX_RANGE(K2_FUNC);
  if (!IsContiguous()) return ToContiguous(*this).To(ctx);

  if (ctx->IsCompatible(*Context())) return *this;

  RegionPtr region = NewRegion(ctx, GetRegion()->num_bytes);

  int8_t *dst = region->GetData<int8_t>();
  const int8_t *src = GetRegion()->GetData<int8_t>();
  Context()->CopyDataTo(region->num_bytes, src, ctx, dst);
  TensorImplPtr impl =
      std::make_shared<TensorImpl>(GetShape(), GetDtype(), size_t(0), region);
  return Tensor(impl);
}

Tensor Tensor::Clone() const {
  NVTX_RANGE(K2_FUNC);
  if (!IsContiguous()) return ToContiguous(*this);

  ContextPtr &context = Context();
  RegionPtr region = NewRegion(context, GetRegion()->num_bytes);

  int8_t *dst = region->GetData<int8_t>();
  const int8_t *src = GetRegion()->GetData<int8_t>();
  context->CopyDataTo(region->num_bytes, src, context, dst);
  TensorImplPtr impl =
      std::make_shared<TensorImpl>(GetShape(), GetDtype(), size_t(0), region);
  return Tensor(impl);
}

}  // namespace k2
