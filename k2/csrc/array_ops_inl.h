/**
 * @brief
 * array_ops_inl
 *
 * @note
 * Don't include this file directly; it is included by ops.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Fangjun Kuang (csukuangfj@gmail.com)
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_OPS_INL_H_
#define K2_CSRC_ARRAY_OPS_INL_H_

#ifndef IS_IN_K2_CSRC_ARRAY_OPS_H_
#error "this file is supposed to be included only by array_ops.h"
#endif

// No header guard for this file since it will only be included
// in ops.h

namespace k2 {

template <typename T>
void ExclusiveSum(ContextPtr &c,
                  Array2<T> &src, Array2<T> *dest, int32_t axis) {
  // TODO
}

template <typename T>
void ExclusiveSum(Array2<T> &src, Array2<T> *dest) {
  // TODO
}

template <typename T>
void ExclusiveSum(Array1<T> &src) {
  Array1<T> ans(src.Context(), src.Dim());
  ExclusiveSum(src, &ans);
  return ans;
}

template <typename T>
Array1<T> RandUniformArray1(ContextPtr &c, int32_t dim, T min_value,
                            T max_value) {
  Array1<T> temp(GetCpuContext(), dim);
  T *data = temp.Data();
  K2_CHECK(max_value >= min_value);
  if (max_value == min_value) {
    for (int32_t i = 0; i < dim; i++) data[i] = 0;
  } else if (std::is_floating_point<T>::value ||
             std::abs(min_value) > RAND_MAX || std::abs(max_value) > RAND_MAX) {
    for (int32_t i = 0; i < dim; i++)
      data[i] = min_value + (rand() * (max_value - min_value) / RAND_MAX);
  } else {
    for (int32_t i = 0; i < dim; i++)
      data[i] = min_value + (rand() % (max_value + 1 - min_value));
  }
  return temp.To(c);
}

}  // namespace k2

#endif
