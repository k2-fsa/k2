/**
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <vector>

#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

torch::DeviceType ToTorchDeviceType(DeviceType type) {
  switch (type) {
    case kCuda:
      return torch::kCUDA;
    case kCpu:
      return torch::kCPU;
    case kUnk:  // fall-through
    default:
      K2_LOG(FATAL) << "kUnk is not supported!";
      return torch::kCPU;  // unreachable code
  }
}

DeviceType FromTorchDeviceType(const torch::DeviceType &type) {
  switch (type) {
    case torch::kCUDA:
      return kCuda;
    case torch::kCPU:
      return kCpu;
    default:
      K2_LOG(FATAL) << "Unsupported device type: " << type
                    << ". Only torch::kCUDA and torch::kCPU are supported";
      return kUnk;  // unreachable code
  }
}

Dtype ScalarTypeToDtype(torch::ScalarType scalar_type) {
  switch (scalar_type) {
    case torch::kFloat:
      return kFloatDtype;
    case torch::kDouble:
      return kDoubleDtype;
    case torch::kInt:
      return kInt32Dtype;
    default:
      // TODO(fangjun): add other type when needed
      K2_LOG(FATAL) << "Unsupported scalar_type: " << scalar_type;
      return kInt32Dtype;  // unreachable code
  }
}

torch::ScalarType ScalarTypeFromDtype(Dtype dtype) {
  switch (dtype) {
    case kFloatDtype:
      return torch::kFloat;
    case kDoubleDtype:
      return torch::kDouble;
    case kInt32Dtype:
      return torch::kInt;
    default:
      // TODO(fangjun): add other type when needed
      K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(dtype).Name();
      return torch::ScalarType::Undefined;  // unreachable code
  }
}

template <>
torch::Tensor ToTensor(Array1<Arc> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<int32_t>::value;
  // an Arc has 4 members
  K2_STATIC_ASSERT(sizeof(Arc) == 4 * sizeof(int32_t));
  std::vector<int64_t> sizes = {array.Dim(), 4};  // [num_rows, num_cols]
  std::vector<int64_t> strides = {4, 1};          // in number of elements
  auto options = torch::device(device).dtype(scalar_type);
  if (array.Dim() == 0) return torch::empty({0, 4}, options);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that the returned tensor outlives the input array.
  return torch::from_blob(
      array.Data(), sizes, strides,
      [saved_region = array.GetRegion()](void *) {}, options);
}

template <>
Array1<Arc> FromTensor<Arc>(torch::Tensor &tensor) {
  K2_CHECK_EQ(tensor.dim(), 2) << "Expected dim: 2. Given: " << tensor.dim();
  K2_CHECK_EQ(tensor.scalar_type(), ToScalarType<int32_t>::value)
      << "Expected scalar type: " << ToScalarType<int32_t>::value
      << ". Given: " << tensor.scalar_type();

  K2_CHECK_EQ(tensor.strides()[0], 4) << "Expected stride: 4. "
                                      << "Given: " << tensor.strides()[0];

  K2_CHECK_EQ(tensor.strides()[1], 1) << "Expected stride: 1. "
                                      << "Given: " << tensor.strides()[1];

  K2_CHECK_EQ(tensor.numel() % 4, 0);

  auto region = NewRegion(tensor);
  Array1<Arc> ans(tensor.numel() / 4, region, 0);
  return ans;
}

Tensor FromTensor(torch::Tensor &tensor, TensorTag) {
  Dtype dtype = ScalarTypeToDtype(tensor.scalar_type());
  torch::IntArrayRef sizes = tensor.sizes();
  torch::IntArrayRef strides = tensor.strides();
  Shape shape({sizes.begin(), sizes.end()}, {strides.begin(), strides.end()});

  auto region = NewRegion(tensor);
  return Tensor(dtype, shape, region, 0);
}
torch::Tensor ToTensor(Tensor &tensor) {
  auto device_type = ToTorchDeviceType(tensor.Context()->GetDeviceType());
  int32_t device_id = tensor.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ScalarTypeFromDtype(tensor.GetDtype());
  auto options = torch::device(device).dtype(scalar_type);

  auto dims_int32 = tensor.Dims();
  auto strides_int32 = tensor.Strides();
  std::vector<int64_t> sizes(dims_int32.begin(), dims_int32.end());
  std::vector<int64_t> strides(strides_int32.begin(), strides_int32.end());

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  // This prevent the memory managed by k2::Tensor from being freed
  // as long as torch::Tensor is alive.
  return torch::from_blob(
      tensor.Data(), sizes, strides,
      [saved_region = tensor.GetRegion()](void *) {}, options);
}

ContextPtr GetContext(torch::Tensor tensor) {
  if (tensor.device().type() == torch::kCPU) return GetCpuContext();

  K2_CHECK(tensor.is_cuda());
  return GetCudaContext(tensor.device().index());
}

}  // namespace k2
