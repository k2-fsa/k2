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

#define K2_EVAL(context, dim, lambda_name, ...)           \
  do {                                                    \
    if (context->GetDeviceType() == kCpu) {               \
      auto lambda_name = [=] __VA_ARGS__;                 \
      int32_t _dim = dim;                                 \
      for (int32_t i = 0; i != _dim; ++i) lambda_name(i); \
    } else {                                              \
      auto lambda_name = [=] __device__ __VA_ARGS__;      \
      EvalDevice(context, dim, lambda_name);              \
    }                                                     \
  } while (0)

#endif  // K2_CSRC_MACROS_H_
