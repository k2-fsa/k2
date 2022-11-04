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

#include <string>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/pytorch_context.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/tensor_ops.h"
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

/** Create a torch device from a k2 context.
   @param [in] context   It must be a CPU or a CUDA context.
   @return Return a CPU or a GPU device depending on the given context.
 */
torch::Device DeviceFromContext(ContextPtr context);

/** Convert torch ScalarType to k2 Dtype

    @param scalar_type  A torch ScalarType.
    @return  Return a k2 Dtype.
 */
Dtype ConvertDtype(torch::ScalarType scalar_type);

/** Conver k2 Dtype to torch ScalarType.

    @param dtype  A k2 Dtype.
    @return Return a torch ScalarType
 */
torch::ScalarType ConvertDtype(Dtype dtype);

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

template <typename T>
torch::Tensor Array2ToTorch(Array2<T> &array) {
  auto device = DeviceFromContext(array.Context());
  auto scalar_type = caffe2::TypeMeta::Make<T>();
  auto options = torch::device(device).dtype(scalar_type);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  auto tensor = torch::from_blob(
      array.Data(), {array.Dim0(), array.Dim1()}, {array.ElemStride0(), 1},
      [saved_region = array.GetRegion()](void *) {}, options);
  return tensor;
}

/* Convert an Array1<T> to torch::Tensor.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in]  array The input array.

   @return a 1-D torch::Tensor which shares the underlying memory
           with the input array.
 */
template <typename T>
torch::Tensor Array1ToTorch(Array1<T> &array) {
  auto device = DeviceFromContext(array.Context());
  auto scalar_type = caffe2::TypeMeta::Make<T>();
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

/** Convert torch Tensor to k2 Tensor

    @param tensor  A torch Tensor.
    @return Return a k2 Tensor.
 */
Tensor TensorFromTorch(torch::Tensor tensor);

/** Convert k2 Tensor to torch Tensor

    @param tensor  A k2 Tensor.
    @return Return a torch Tensor.
 */
torch::Tensor TensorToTorch(Tensor &tensor);

/* Returns a 1-D tensor which indexes the src tensor using entries
   from `index`.

   @param  [in]  src    A 1-D tensor.
   @param  [in]  index  A 1-D tensor with dtype torch.int32.
                        It has to satisfy:
                            -1 <= index[i] < src.numel()
                            for i in [0, index.numel())
                        CAUTION: We require that index.is_contiguous() is true.
   @param [in] default_value  The value for ans[i] when index[i] is -1.
   @return
      Returns a 1-D contiguous tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = default_value if index[i] is -1
 */
template <typename T>
torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index,
                          T default_value) {
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK(src.dtype().Match<T>())
      << "Expected dtype type: " << caffe2::TypeMeta::Make<T>()
      << ". Given: " << src.scalar_type();
  K2_CHECK_EQ(index.dim(), 1)
      << "Expected index dim: 1. Given : " << index.dim();
  K2_CHECK(index.dtype().Match<int32_t>())
      << "Expected dtype type: " << caffe2::TypeMeta::Make<int32_t>()
      << ". Given: " << index.scalar_type();
  K2_CHECK(index.is_contiguous()) << "Expected contiguous";
  K2_CHECK_EQ(src.device(), index.device())
      << "Expected in the same device"
      << " Given : " << src.device() << ", " << index.device();

  bool allow_minus_one = true;
  Array1<int32_t> index_array = Array1FromTorch<int32_t>(index);
  // If index_array.Dim() equals to zero, the `Index` below would produce an
  // ans with `ans.Data()` be a nullptr, which will cause crash when calling
  // `torch::from_blob`. Just return an empty tensor here.
  // If src is an empty tensor, we should return an empty torch.
  if (index_array.Dim() == 0 || src.numel() == 0)
    return torch::empty({0}, src.options());
  if (src.is_contiguous()) {
    Array1<T> src_array = Array1FromTorch<T>(src);
    Array1<T> ans_array =
        Index(src_array, index_array, allow_minus_one, default_value);
    return Array1ToTorch(ans_array);
  }
  Tensor tensor = TensorFromTorch(src);
  Tensor ans = Index(tensor, index_array, allow_minus_one, default_value);
  return TensorToTorch(ans);
}

}  // namespace k2

#endif  // K2_TORCH_CSRC_UTILS_H_
