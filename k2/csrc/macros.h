/**
 * Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang
 *                                                   Haowen Qiu)
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

#ifndef K2_CSRC_MACROS_H_
#define K2_CSRC_MACROS_H_

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define K2_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define K2_FUNC __func__
#endif

/*
`K2_EVAL` simplifies the task of writing lambdas
for CUDA as well as for CPU.

Assume you have a lambda to increment the elements of an array:

@code
  Array1<int32_t> array = ...; // initialize it
  int32_t *array_data = array.Data();
  ContextPtr &context = array.Context();
  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != array.Dim(); ++i) array_data[i] += 1;
  } else {
    auto lambda_inc = [=] __device__ (int32_t i) {
      array_data[i] += 1;
    };
    EvalDevice(context, array.Dim(); lambda_inc);
  }
@endcode

You can replace the above code with `K2_EVAL` by the following code:

@code
  Array1<int32_t> array = ...; // initialize it
  int32_t *array_data = array.Data();
  K2_EVAL(
      c, array.Dim(), inc, (int32_t i)->void { array_data[i] += 1; });
@endcode
 */

#define K2_EVAL(context, dim, lambda_name, ...)                        \
  do {                                                                 \
    if (context->GetDeviceType() == kCpu) {                            \
      auto lambda_name = [=] __VA_ARGS__;                              \
      int32_t lambda_name##_dim = dim;                                 \
      for (int32_t i = 0; i != lambda_name##_dim; ++i) lambda_name(i); \
    } else {                                                           \
      auto lambda_name = [=] __device__ __VA_ARGS__;                   \
      EvalDevice(context, dim, lambda_name);                           \
    }                                                                  \
  } while (0)

/*
`K2_EVAL2` simplifies the task of writing lambdas
for CUDA as well as for CPU.

Assume you have a lambda to increment the elements of an array2:

@code
  Array2<int32_t> array = ...; // initialize it
  int32_t *array_data = array.Data();
  int32_t elem_stride0 = array.ElemStride0();
  ContextPtr &context = array.Context();
  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != array.Dim0(); ++i)
      for (int32_t j = 0; j != array.Dim1(); ++j)
        array_data[i * elem_stride0 + j] += 1;
  } else {
    auto lambda_inc = [=] __device__ (int32_t i, j) {
      array_data[i * elem_stride0 + j] += 1;
    };
    Eval2Device(context, array.Dim0(), array.Dim1(), lambda_inc);
  }
@endcode

You can replace the above code with `K2_EVAL2` by the following code:

@code
  Array2<int32_t> array = ...; // initialize it
  int32_t *array_data = array.Data();
  int32_t elem_stride0 = array.ElemStride0();
  ContextPtr &context = array.Context();
  K2_EVAL2(
      context, array.Dim0(), array.Dim1(), lambda_inc,
      (int32_t i, int32_t j)->void {
        array_data[i * elem_stride0 + j] += 1;
      });
@endcode
 */
#define K2_EVAL2(context, m, n, lambda_name, ...)                         \
  do {                                                                    \
    if (context->GetDeviceType() == kCpu) {                               \
      auto lambda_name = [=] __VA_ARGS__;                                 \
      int32_t lambda_name##_m = m;                                        \
      int32_t lambda_name##_n = n;                                        \
      for (int32_t i = 0; i != lambda_name##_m; ++i)                      \
        for (int32_t j = 0; j != lambda_name##_n; ++j) lambda_name(i, j); \
    } else {                                                              \
      auto lambda_name = [=] __device__ __VA_ARGS__;                      \
      Eval2Device(context, m, n, lambda_name);                            \
    }                                                                     \
  } while (0)

/*
`K2_TRANS_EXCSUM`, calls a lambda function (int32_t i) -> int32_t for i in range
[0, dim), then does an exclusive sum on the returned values and writes them to
`output`, i.e. it does an operation `transform and then exclusive sum`.
Noted `output` must have size `dim + 1`. It works for CUDA as well as for CPU.

Here is an example:

@code
  ContextPtr c = GetCudaContext();
  int32_t dim = 3;
  Array1<int32_t> ans(c, dim + 1);
  int32_t *ans_data = ans.Data();
  K2_TRANS_EXCSUM(
      c, dim, ans_data, lambda_multiple2, (int32_t i)->int32_t{ return i*2; });
@endcode

`ans` will be {0, 0, 2, 6}
 */
#define K2_TRANS_EXCSUM(context, dim, ans_data, lambda_name, ...)              \
  do {                                                                         \
    if (context->GetDeviceType() == kCpu) {                                    \
      auto lambda_name = [=] __VA_ARGS__;                                      \
      int32_t lambda_name##_dim = dim;                                         \
      ans_data[0] = 0;                                                         \
      for (int32_t i = 0; i != lambda_name##_dim; ++i) {                       \
        int32_t value = lambda_name(i);                                        \
        ans_data[i + 1] = ans_data[i] + value;                                 \
      }                                                                        \
    } else {                                                                   \
      K2_CHECK_EQ(context->GetDeviceType(), kCuda);                            \
      auto lambda_name = [=] __device__ __VA_ARGS__;                           \
      mgpu::context_t *mgpu_context = GetModernGpuAllocator(context);          \
      K2_CUDA_SAFE_CALL(mgpu::transform_scan<int32_t>(                         \
          lambda_name, dim, ans_data, mgpu::plus_t<int32_t>(), ans_data + dim, \
          *mgpu_context));                                                     \
    }                                                                          \
  } while (0)

#endif  // K2_CSRC_MACROS_H_
