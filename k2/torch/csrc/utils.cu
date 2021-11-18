/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "k2/csrc/array.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/utils.h"

#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 9)
// for torch::jit::readArchiveAndTensors
#include "torch/csrc/jit/serialization/import_read.h"
#endif

namespace k2 {

torch::DeviceType ConvertDeviceType(DeviceType device_type) {
  switch (device_type) {
    case kCpu:
      return torch::kCPU;
    case kCuda:
      return torch::kCUDA;
    default:
      K2_LOG(FATAL) << "Unknown device type: " << device_type;
  }
  // Unreachable code
  return torch::kCPU;
}

DeviceType ConvertDeviceType(torch::DeviceType device_type) {
  switch (device_type) {
    case torch::kCPU:
      return kCpu;
    case torch::kCUDA:
      return kCuda;
    default:
      K2_LOG(FATAL) << "Unknown device type: " << device_type;
  }
  // Unreachable code
  return kCpu;
}

Dtype ConvertDtype(torch::ScalarType scalar_type) {
  switch (scalar_type) {
    case torch::kFloat:
      return kFloatDtype;
    case torch::kDouble:
      return kDoubleDtype;
    case torch::kInt:
      return kInt32Dtype;
    case torch::kLong:
      return kInt64Dtype;
    default:
      // TODO(fangjun): add other types when needed
      K2_LOG(FATAL) << "Unsupported scalar_type: " << scalar_type;
      return kInt32Dtype;  // unreachable code
  }
}

torch::ScalarType ConvertDtype(Dtype dtype) {
  switch (dtype) {
    case kFloatDtype:
      return torch::kFloat;
    case kDoubleDtype:
      return torch::kDouble;
    case kInt32Dtype:
      return torch::kInt;
    case kInt64Dtype:
      return torch::kLong;
    default:
      // TODO(fangjun): add other types when needed
      K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(dtype).Name();
      return torch::ScalarType::Undefined;  // unreachable code
  }
}

torch::Device DeviceFromContext(ContextPtr context) {
  auto device_type = ConvertDeviceType(context->GetDeviceType());
  int32_t device_id = context->GetDeviceId();
  return torch::Device(device_type, device_id);
}

ContextPtr ContextFromDevice(torch::Device device) {
  torch::DeviceType device_type = device.type();

  if (device_type == torch::kCPU) return GetCpuContext();

  K2_CHECK_EQ(device.type(), torch::kCUDA);
  return GetCudaContext(device.index());
}

template <>
Array1<Arc> Array1FromTorch<Arc>(torch::Tensor tensor) {
  K2_CHECK_EQ(tensor.dim(), 2) << "Expected dim: 2. Given: " << tensor.dim();
  K2_CHECK(tensor.dtype().Match<int32_t>())
      << "Expected dtype type: " << caffe2::TypeMeta::Make<int32_t>()
      << ". Given: " << tensor.scalar_type();

  K2_CHECK_EQ(tensor.stride(0), 4) << "Expected stride: 4. "
                                   << "Given: " << tensor.stride(0);

  K2_CHECK_EQ(tensor.stride(1), 1) << "Expected stride: 1. "
                                   << "Given: " << tensor.stride(1);

  K2_CHECK_EQ(tensor.numel() % 4, 0);

  auto region = NewRegion(tensor);
  Array1<Arc> ans(tensor.numel() / 4, region, 0);
  return ans;
}

Tensor TensorFromTorch(torch::Tensor tensor) {
  Dtype dtype = ConvertDtype(tensor.scalar_type());
  torch::IntArrayRef sizes = tensor.sizes();
  torch::IntArrayRef strides = tensor.strides();
  Shape shape({sizes.begin(), sizes.end()}, {strides.begin(), strides.end()});

  auto region = NewRegion(tensor);
  return Tensor(dtype, shape, region, 0);
}

torch::Tensor TensorToTorch(Tensor &tensor) {
  auto device = DeviceFromContext(tensor.Context());
  auto scalar_type = ConvertDtype(tensor.GetDtype());
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

}  // namespace k2
