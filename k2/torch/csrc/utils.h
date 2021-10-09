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

#ifndef K2_TORCH_CSRC_UTILS_H_
#define K2_TORCH_CSRC_UTILS_H_

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/pytorch_context.h"
#include "torch/script.h"

namespace k2 {

/** Convert a device type in k2 to torch.

    @param device_type A k2 device type.
    @return Return a torch device type.
 */
torch::DeviceType ConvertDeviceType(DeviceType device_type);

/** Convert a device type in torch to k2.

    @param device_type A torch device type.
    @return Return a k2 device type.
 */
DeviceType ConvertDeviceType(torch::DeviceType device_type);

/** Construct a k2 context from a torch device.

    @param device   A torch device. It can be either a CPU device or
                    a CUDA device.
    @return Return a k2 context.
 */
ContextPtr ContextFromDevice(torch::Device device);

inline ContextPtr ContextFromTensor(torch::Tensor tensor) {
  return ContextFromDevice(tensor.device());
}

template <typename T>
Array1<T> Array1FromTorch(torch::Tensor tensor) {
  K2_CHECK_EQ(tensor.dim(), 1) << "Expected dim: 1. Given: " << tensor.dim();
  K2_CHECK(tensor.dtype().Match<T>())
      << "Expected dtype type: " << caffe2::TypeMeta::Make<T>()
      << ". Given: " << tensor.scalar_type();
  // Some empty tensor may have stride not equal to 1, e.g., tensor returned by
  // clone() method, it is valid here, so we won't check its strides.
  if (tensor.numel() > 0)
    K2_CHECK_EQ(tensor.stride(0), 1)
        << "Expected stride: 1. Given: " << tensor.stride(0);

  auto region = NewRegion(tensor);
  Array1<T> ans(tensor.numel(), region, 0);
  return ans;
}

template <>
Array1<Arc> Array1FromTorch<Arc>(torch::Tensor tensor);

template <typename T>
Array2<T> Array2FromTorch(torch::Tensor tensor) {
  K2_CHECK_EQ(tensor.dim(), 2) << "Expected dim: 2. Given: " << tensor.dim();
  K2_CHECK(tensor.dtype().Match<T>())
      << "Expected dtype type: " << caffe2::TypeMeta::Make<T>()
      << ". Given: " << tensor.scalar_type();

  K2_CHECK_EQ(tensor.stride(1), 1)
      << "Expected stride: 1. Given: " << tensor.stride(1);

  auto region = NewRegion(tensor);
  Array2<T> ans(tensor.size(0),    // dim0
                tensor.size(1),    // dim1
                tensor.stride(0),  // elem_stride0
                0,                 // byte_offset
                region);           // region
  return ans;
}

}  // namespace k2

#endif  // K2_TORCH_CSRC_UTILS_H_
