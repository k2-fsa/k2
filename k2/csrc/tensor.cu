/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
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
  K2_CHECK_LT(num_axes_, kMaxDim);

  std::copy(dims.begin(), dims.end(), dims_);

  // compute strides_
  if (num_axes_ > 0) strides_[num_axes_ - 1] = 1;

  for (int32_t i = num_axes_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * dims_[i + 1];
  }

  num_elements_ = ComputeNumElements();
  is_contiguous_ = true;  // always be true here as we compute strides from dims
}

Shape::Shape(const std::vector<int32_t> &dims,
             const std::vector<int32_t> strides)
    : num_axes_(static_cast<int32_t>(dims.size())) {
  K2_CHECK_LT(num_axes_, kMaxDim);
  K2_CHECK_EQ(static_cast<int32_t>(strides.size()), num_axes_);
  std::copy(dims.begin(), dims.end(), dims_);
  std::copy(strides.begin(), strides.end(), strides_);
  num_elements_ = ComputeNumElements();
  is_contiguous_ = ComputeIsContiguous();
}

int64_t Shape::ComputeNumElements() const {
  if (num_axes_ == 0) return 1;  // scalar

  int64_t elements = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    elements *= dims_[i];
  }
  return elements;
}

void Shape::GetReachableElems(int64_t *begin_out, int64_t *end_out) const {
  int64_t begin = 0,
      end = 1;
  for (int32_t i = 0; i < num_axes_; ++i) {
    if (dims_[i] == 0) {
      goto empty_output;
    } else if (strides_[i] > 0) {
      end += (dims_[i] - 1) * (int64_t)strides_[i];
    } else {
      begin += (dims_[i] - 1) * (int64_t)strides_[i];
    }
  }
  *begin_out = begin;
  *end_out = end;
  return;
empty_output:
  *begin_out = 0;
  *end_out = 0;
}

int64_t Shape::StorageSize() const {
  int64_t begin, end;
  GetReachableElems(&begin, &end);
  return end - begin;
}


bool Shape::ComputeIsContiguous() const {
  int64_t z = 1;
  for (int32_t i = num_axes_ - 1; i >= 0; --i) {
    if (dims_[i] != 1) {
      if (strides_[i] != z) return false;
      z *= dims_[i];
    }
  }
  return true;
}

void Shape::SetStride(int32_t axis, int32_t stride) {
  K2_CHECK_LT(static_cast<uint32_t>(axis), static_cast<uint32_t>(num_axes_));
  strides_[axis] = stride;
  ComputeIsContiguous();
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
               size_t byte_offset)
    : impl_(std::make_shared<TensorImpl>()) {
  int64_t begin_elem, end_elem;
  shape.GetReachableElems(&begin_elem, &end_elem);
  int64_t element_size = TraitsOf(type).NumBytes();
  impl_->dtype = type;
  impl_->shape = shape;
  impl_->data = region;
  impl_->byte_offset = byte_offset;
  K2_CHECK_GE(int64_t(impl_->byte_offset) + begin_elem * element_size, 0)
      << "impl_->byte_offset: " << int64_t(impl_->byte_offset) << ", "
      << "begin_elem: " << begin_elem << ", "
      << "element_size: " << element_size;

  K2_CHECK_LE(int64_t(impl_->byte_offset) + end_elem * element_size,
              int64_t(impl_->data->num_bytes))
      << "impl_->byte_offset: " << int64_t(impl_->byte_offset) << ", "
      << "end_elem: " << end_elem << ", "
      << "element_size: " << element_size << ", "
      << "impl_->data->num_bytes: " << int64_t(impl_->data->num_bytes);
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
  int64_t begin_elem, end_elem;
  impl_->shape.GetReachableElems(&begin_elem, &end_elem);
  int64_t element_size = TraitsOf(impl_->dtype).NumBytes();
  int64_t byte_offset = -begin_elem * element_size,
      storage_size_bytes = byte_offset + end_elem * element_size;
  impl_->data = NewRegion(c, static_cast<size_t>(storage_size_bytes));
  impl_->byte_offset = byte_offset;
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
