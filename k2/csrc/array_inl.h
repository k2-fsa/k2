/**
 * @brief
 * array_inl
 *
 * @note
 * Don't include this file directly; it is included by array.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_INL_H_
#define K2_CSRC_ARRAY_INL_H_

#ifndef IS_IN_K2_CSRC_ARRAY_H_
#error "this file is supposed to be included only by array.h"
#endif

#include <algorithm>
#include <cassert>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "cub/cub.cuh"
#include "k2/csrc/utils.h"

namespace k2 {

template <typename T>
Array1<T> Array1<T>::Clone() const {
  Array1<T> ans(Context(), Dim());
  ans.CopyFrom(*this);
  return ans;
}

template <typename T>
void Array1<T>::CopyFrom(const Array1<T> &src) {
  K2_CHECK_EQ(dim_, src.dim_);
  if (dim_ == 0) return;
  auto kind = GetMemoryCopyKind(*src.Context(), *Context());
  const T *src_data = src.Data();
  T *dst_data = this->Data();
  MemoryCopy(static_cast<void *>(dst_data), static_cast<const void *>(src_data),
             Dim() * ElementSize(), kind, Context().get());
}
template <typename T>
template <typename S>
Array1<S> Array1<T>::AsType() {
  if (std::is_same<T, S>::value) return *reinterpret_cast<Array1<S> *>(this);
  Array1<S> ans(Context(), Dim());
  S *ans_data = ans.Data();
  const T *this_data = Data();
  auto lambda_set_values = [=] __host__ __device__(int32_t i) {
    ans_data[i] = static_cast<S>(this_data[i]);
  };
  Eval(Context(), Dim(), lambda_set_values);
  return ans;
}
}  // namespace k2

#endif  // K2_CSRC_ARRAY_INL_H_
