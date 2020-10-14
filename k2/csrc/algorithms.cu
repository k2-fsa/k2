/**
 * @brief
 * algorithms
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"

namespace k2 {
Array1<int32_t> Renumbering::New2Old() {
  ContextPtr c = keep_.Context();
  // `+1` as we need prefix sum including the last element of `keep_`.
  Array1<int32_t> sum(c, num_old_elems_ + 1);
  // Note we have allocated an extra element in `keep_`, so it's safe
  // to call ExclusiveSum with dest->Dim() == src->Dim() + 1.
  ExclusiveSum(keep_, &sum);
  num_new_elems_ = sum.Back();
  Array1<int32_t> new2old(c, num_new_elems_);

  const int32_t *sum_data = sum.Data();
  int32_t *new2old_data = new2old.Data();
  auto lambda_set_indexes = [=] __host__ __device__(int32_t old_idx) {
    if (sum_data[old_idx + 1] > sum_data[old_idx]) {
      int32_t new_idx = sum_data[old_idx];
      new2old_data[new_idx] = old_idx;
    }
  };
  Eval(c, num_old_elems_, lambda_set_indexes);
  return new2old;
}

Array1<int32_t> Renumbering::Old2New() {
  ContextPtr c = keep_.Context();
  // `+1` as we need prefix sum including the last element of `keep_`.
  Array1<int32_t> sum(c, num_old_elems_ + 1);
  // Note we have allocated an extra element in `keep_`, so it's safe
  // to call ExclusiveSum with dest->Dim() == src->Dim() + 1.
  ExclusiveSum(keep_, &sum);
  num_new_elems_ = sum.Back();
  Array1<int32_t> old2new(c, num_old_elems_);

  const int32_t *sum_data = sum.Data();
  int32_t *old2new_data = old2new.Data();
  auto lambda_set_indexes = [=] __host__ __device__(int32_t old_idx) {
    if (sum_data[old_idx + 1] > sum_data[old_idx])
      old2new_data[old_idx] = sum_data[old_idx];
    else
      old2new_data[old_idx] = -1;
  };
  Eval(c, num_old_elems_, lambda_set_indexes);
  return old2new;
}
}  // namespace k2
