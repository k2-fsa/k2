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
TO_SCALAR_TYPE(int32_t, torch::kInt);

#undef TO_SCALAR_TYPE

/* Convert an Array1<T> to torch::Tensor.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in]  array The input array.

   @return a 1-D torch::Tensor which shares the underlying memory
           with the input array.
 */
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
      array.Data(), array.Dim(), [array](void *) {}, options);
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
torch::Tensor ToTensor(Array1<Arc> &array);

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
Array1<Arc> FromTensor<Arc>(torch::Tensor &tensor);

struct Array2Tag {};

template <typename T>
Array2<T> FromTensor(torch::Tensor &tensor, Array2Tag) {
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
torch::Tensor ToTensor(Array2<T> &array) {
  auto device_type = ToTorchDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  auto options = torch::device(device).dtype(scalar_type);

  // NOTE: we keep a copy of `array` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  auto tensor = torch::from_blob(
      array.Data(), {array.Dim0(), array.Dim1()}, {array.ElemStride0(), 1},
      [array](void *) {}, options);
  return tensor;
}

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_TORCH_UTIL_H_
