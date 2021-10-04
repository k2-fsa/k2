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

#include "k2/torch/csrc/utils.h"

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

}  // namespace k2
