/**
 * @brief
 * test_utils
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_TEST_UTILS_H_
#define K2_CSRC_TEST_UTILS_H_

#include <gtest/gtest.h>

#include <vector>

#include "k2/csrc/array.h"

namespace k2 {

template <typename T>
void CheckArrayData(const Array1<T> &array, const std::vector<T> &target) {
  ASSERT_EQ(array.Dim(), target.size());
  const T *array_data = array.Data();
  // copy data from CPU/GPU to CPU
  auto kind = GetMemoryCopyKind(*array.Context(), *GetCpuContext());
  std::vector<T> cpu_data(array.Dim());
  MemoryCopy(static_cast<void *>(cpu_data.data()),
             static_cast<const void *>(array_data),
             array.Dim() * array.ElementSize(), kind, nullptr);
  EXPECT_EQ(cpu_data, target);
}

// check if `array` and `target` have the same values
template <typename T>
void CheckArrayData(const Array1<T> &array, const Array1<T> &target) {
  ASSERT_EQ(array.Dim(), target.Dim());
  int32_t dim = array.Dim();
  ContextPtr cpu = GetCpuContext();
  Array1<T> cpu_array = array.To(cpu);
  Array1<T> cpu_target = target.To(cpu);
  std::vector<T> array_data(cpu_array.Data(), cpu_array.Data() + dim);
  std::vector<T> target_data(cpu_target.Data(), cpu_target.Data() + dim);
  EXPECT_EQ(array_data, target_data);
}

void CheckRowSplits(RaggedShape &shape,
                    const std::vector<std::vector<int32_t>> &target) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    Array1<int32_t> curr_row_splits = shape.RowSplits(i);
    CheckArrayData<int32_t>(curr_row_splits, target[i - 1]);
  }
}

}  // namespace k2

#endif  //  K2_CSRC_TEST_UTILS_H_
