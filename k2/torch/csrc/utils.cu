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

#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "k2/csrc/array.h"
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

torch::IValue PickleLoad(const std::string &filename) {
  auto rai = std::make_unique<caffe2::serialize::FileAdapter>(filename);
  auto reader = torch::make_unique<caffe2::serialize::PyTorchStreamReader>(
      std::move(rai));

#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 9)
  return torch::jit::readArchiveAndTensors("data", "", "",
                                           /*class_resolver=*/torch::nullopt,
                                           /*obj_loader=*/torch::nullopt,
                                           /*device=*/c10::nullopt, *reader);

#else
  return torch::jit::readArchiveAndTensors("data",
                                           /*class_resolver=*/torch::nullopt,
                                           /*obj_loader=*/torch::nullopt,
                                           /*device=*/c10::nullopt, *reader);
#endif
}

}  // namespace k2
