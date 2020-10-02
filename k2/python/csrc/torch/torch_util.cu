/**
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

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

template <>
torch::Tensor ToTensor(Array1<Arc> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<int32_t>::value;
  // an Arc has 4 members
  torch::IntArrayRef strides = {4, 1};  // in number of elements
  auto options = torch::device(device).dtype(scalar_type);

  // NOTE: we keep a copy of `array` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  return torch::from_blob(
      array.Data(), {array.Dim(), 4}, strides, [array](void *) {}, options);
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

}  // namespace k2
