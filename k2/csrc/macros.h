/**
 * @brief
 * macros
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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





#endif  // K2_CSRC_MACROS_H_
