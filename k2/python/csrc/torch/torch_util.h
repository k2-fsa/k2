/**
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_TORCH_UTIL_H_
#define K2_PYTHON_CSRC_TORCH_TORCH_UTIL_H_

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/log.h"
#include "k2/csrc/pytorch_context.h"
#include "torch/extension.h"

namespace k2 {

/* Convert k2::DeviceType to torch::DeviceType.
   Abort on failure.
 */
torch::DeviceType ToTorchDeviceType(DeviceType type);

/* Convert torch::DeviceType to k2::DeviceType.
   Abort on failure.
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
TO_SCALAR_TYPE(int32_t, torch::kInt);

#undef TO_SCALAR_TYPE

template <typename T>
torch::Tensor ToTensor(Array1<T> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  auto options = torch::device(device).dtype(scalar_type);

  // NOTE: we keep a copy of `array` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  return torch::from_blob(
      array.Data(), array.Dim(), [array](void *p) {}, options);
}

template <typename T>
Array1<T> FromTensor(torch::Tensor &tensor) {
  K2_CHECK_EQ(tensor.dim(), 1) << "Expected dim: 1. Given: " << tensor.dim();
  K2_CHECK_EQ(tensor.scalar_type(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << tensor.scalar_type();
  K2_CHECK_EQ(tensor.strides()[0], 1)
      << "Expected stride: 1. Given: " << tensor.strides()[0];

  auto region = NewRegion(tensor);
  Array1<T> ans(tensor.numel(), region, 0);
  return ans;
}

/* Return a 2-d tensor.

   The number of rows of the returned tensor is array.Dim();
   the number of columns of the returned tensor is 4.

   CAUTION: the returned tensor has dtype == torch.int32, but
   its last column contains scores of type `float`. That is,
   the float binary pattern is reinterpreted as int.
 */
template <>
torch::Tensor ToTensor(Array1<Arc> &array);

/* Convert a tensor to an Array1<Arc>.

  CAUTION: the given tensor's dtype is torch.int32, but its
  last columns contains scores of type `float`. That is,
  the int binary pattern is reinterpreted as float.
 */
template <>
Array1<Arc> FromTensor<Arc>(torch::Tensor &tensor);

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_TORCH_UTIL_H_
