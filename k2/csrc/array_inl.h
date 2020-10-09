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
#error "this file is supposed to be included only by array_ops.h"
#endif

#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "k2/csrc/utils.h"

namespace k2 {

template <typename T>
Array1<T> Array1<T>::Clone() {
  Array1<T> ans(Context(), Dim());
  ans.CopyFrom(*this);
  return ans;
}

template <typename T>
void Array1<T>::CopyFrom(const Array1<T> &src) {
  K2_CHECK_EQ(dim_, src.dim_);
  if (dim_ == 0)
    return;
  auto kind = GetMemoryCopyKind(*src.Context(), *Context());
  const T *src_data = src.Data();
  T *dst_data = this->Data();
  MemoryCopy(static_cast<void *>(dst_data), static_cast<const void *>(src_data),
             Dim() * ElementSize(), kind, Context().get());
}
  
  
  
}  // namespace k2

#endif  // K2_CSRC_ARRAY_INL_H_
