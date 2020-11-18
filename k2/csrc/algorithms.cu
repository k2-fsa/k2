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
void Renumbering::ComputeOld2New() {
  old2new_ = Array1<int32_t>(keep_.Context(), keep_.Dim() + 1);
  ExclusiveSum(keep_, &old2new_);
  num_new_elems_ = old2new_.Back();
  K2_CHECK_GE(num_new_elems_, 0);
  K2_CHECK_LE(num_new_elems_, keep_.Dim());
  old2new_ = old2new_.Range(0, keep_.Dim());
}

namespace {
// This small piece of code had to be put in a separate function due to
// CUDA limitations about lambdas in classes with private members.
inline void ComputeNew2OldHelper(ContextPtr &c, const int32_t *old2new_data,
                                 int32_t *new2old_data, int32_t old_dim) {
  // Note: the following accesses data one past the end of (current)
  // old2new_, but it does actually exist.
  if (c->GetDeviceType() == kCpu) {
    for (int32_t old_idx = 0; old_idx < old_dim; old_idx++)
      if (old2new_data[old_idx + 1] > old2new_data[old_idx])
        new2old_data[old2new_data[old_idx]] = old_idx;
    new2old_data[old2new_data[old_dim]] = old_dim;
  } else {
    auto lambda_set_new2old = [=] __device__(int32_t old_idx) {
      if (old_idx == old_dim ||
          old2new_data[old_idx + 1] > old2new_data[old_idx])
        new2old_data[old2new_data[old_idx]] = old_idx;
    };
    EvalDevice(c, old_dim + 1, lambda_set_new2old);
  }
}

}  // namespace

void Renumbering::ComputeNew2Old() {
  if (!old2new_.IsValid()) ComputeOld2New();
  new2old_ = Array1<int32_t>(keep_.Context(), num_new_elems_ + 1);
  const int32_t *old2new_data = old2new_.Data();
  int32_t *new2old_data = new2old_.Data();
  ComputeNew2OldHelper(keep_.Context(), old2new_data, new2old_data,
                       keep_.Dim());
  new2old_ = new2old_.Range(0, num_new_elems_);
}

}  // namespace k2
