/**
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
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

#ifndef K2_CSRC_TORCH_UTIL_H_
#define K2_CSRC_TORCH_UTIL_H_

#include <string>

#include "k2/csrc/array.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/log.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

/* Convert k2::DeviceType to torch::DeviceType.
   Abort on failure.

   @param [in] type  We support only kCpu and kCuda at present.

   @return torch::kCUDA or torch.kCPU.
 */
torch::DeviceType ToTorchDeviceType(DeviceType type);

/* Convert torch::DeviceType to k2::DeviceType.
   Abort on failure.

   @param [in] type  We support only torch::kCPU and torch::kCUDA currently.

   @return  kCpu or kCuda.
 */
DeviceType FromTorchDeviceType(const torch::DeviceType &type);

// Some versions of PyTorch do not have `c10::CppTypeToScalarType`,
// so we implement our own here.
template <typename T>
struct ToScalarType;

#define TO_SCALAR_TYPE(cpp_type, scalar_type) \
  template <>                                 \
  struct ToScalarType<cpp_type>               \
      : std::integral_constant<torch::ScalarType, scalar_type> {};

// TODO(fangjun): add other types if needed
TO_SCALAR_TYPE(float, torch::kFloat);
TO_SCALAR_TYPE(double, torch::kDouble);
TO_SCALAR_TYPE(int32_t, torch::kInt);
TO_SCALAR_TYPE(int64_t, torch::kLong);
TO_SCALAR_TYPE(bool, torch::kBool);

#undef TO_SCALAR_TYPE

Dtype ScalarTypeToDtype(torch::ScalarType scalar_type);
torch::ScalarType ScalarTypeFromDtype(Dtype dtype);

/* Convert an Array1<T> to torch::Tensor.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in]  array The input array.

   @return a 1-D torch::Tensor which shares the underlying memory
           with the input array.
 */
template <typename T>
torch::Tensor ToTorch(Array1<T> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  auto options = torch::device(device).dtype(scalar_type);
  // We will call torch::from_blob below. However, if we
  // call it with an empty Array1, we'll get error:
  // RuntimeError: CUDA error: invalid argument Exception raised from
  // getDeviceFromPtr at /pytorch/aten/src/ATen/cuda/CUDADevice.h
  // Definitely we need look into this, but let's just return an empty tensor
  // when the input Array1 is empty for now.
  if (array.Dim() == 0) return torch::empty(0, options);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  return torch::from_blob(
      array.Data(), array.Dim(), [saved_region = array.GetRegion()](void *) {},
      options);
}

/* Convert a 1-D torch::Tensor to an Array1<T>.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in] tensor
                     The input torch tensor.
   @return an Array1<T> sharing the underlying memory with the
           input tensor.
 */
template <typename T>
Array1<T> FromTorch(torch::Tensor tensor) {
  K2_CHECK_EQ(tensor.dim(), 1) << "Expected dim: 1. Given: " << tensor.dim();
  K2_CHECK_EQ(tensor.scalar_type(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << tensor.scalar_type();
  // Some empty tensor may have stride not equal to 1, e.g., tensor returned by
  // clone() method, it is valid here, so we won't check its strieds.
  if (tensor.numel())
    K2_CHECK_EQ(tensor.strides()[0], 1)
        << "Expected stride: 1. Given: " << tensor.strides()[0];

  auto region = NewRegion(tensor);
  Array1<T> ans(tensor.numel(), region, 0);
  return ans;
}

/* Convert an Array1<Arc> to a torch::Tensor.

   CAUTION: the returned tensor has dtype == torch.int32, but
   its last column contains scores of type `float`. That is,
   the float binary pattern is reinterpreted as int.

   @param [in]  array Then input array.

   @return a 2-D torch::Tensor, whose
           dtype == torch.int32,
           num_rows == array.Dim(), and
           num_cols == 4
 */
template <>
torch::Tensor ToTorch(Array1<Arc> &array);

/* Convert a tensor to an Array1<Arc>.

  CAUTION: the given tensor's dtype is torch.int32, but its
  last column contains scores of type `float`. That is,
  the int binary pattern is reinterpreted as float.

  @param [in]  tensor  a 2-D type with dtype == torch.int32 and
                       num_cols == 4

  @return an Array1<Arc> sharing the underlying memory with
          the input tensor.
 */
template <>
Array1<Arc> FromTorch<Arc>(torch::Tensor tensor);

struct Array2Tag {};

template <typename T>
Array2<T> FromTorch(torch::Tensor tensor, Array2Tag) {
  K2_CHECK_EQ(tensor.dim(), 2) << "Expected dim: 2. Given: " << tensor.dim();
  K2_CHECK_EQ(tensor.scalar_type(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << tensor.scalar_type();

  K2_CHECK_EQ(tensor.strides()[1], 1)
      << "Expected stride: 1. Given: " << tensor.strides()[1];

  auto region = NewRegion(tensor);
  Array2<T> ans(tensor.sizes()[0],    // dim0
                tensor.sizes()[1],    // dim1
                tensor.strides()[0],  // elem_stride0
                0,                    // byte_offset
                region);              // region
  return ans;
}

template <typename T>
torch::Tensor ToTorch(Array2<T> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  auto options = torch::device(device).dtype(scalar_type);

  // If array is empty, the `array.Data()` will be a nullptr, which will
  // cause crash when calling `torch::from_blob`. Just return an empty tensor
  // here.
  if (array.Dim0() == 0 || array.Dim1() == 0)
    return torch::empty({array.Dim0(), array.Dim1()}, options);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  auto tensor = torch::from_blob(
      array.Data(), {array.Dim0(), array.Dim1()}, {array.ElemStride0(), 1},
      [saved_region = array.GetRegion()](void *) {}, options);
  return tensor;
}

struct TensorTag {};

Tensor FromTorch(torch::Tensor tensor, TensorTag);
torch::Tensor ToTorch(Tensor &tensor);

/** Create a k2 context from a torch device.

   @param [in] device   It must be either a CPU or a GPU.

   @return Return either a CPU context or a CUDA context
           depending on the given device.
 */
ContextPtr GetContext(torch::Device device);

inline ContextPtr GetContext(torch::Tensor tensor) {
  return GetContext(tensor.device());
}

}  // namespace k2

#endif  // K2_CSRC_TORCH_UTIL_H_
