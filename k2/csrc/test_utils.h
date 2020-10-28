/**
 * @brief
 * test_utils
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_TEST_UTILS_H_
#define K2_CSRC_TEST_UTILS_H_

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"

namespace k2 {

// clang-format off
bool operator==(const Arc &a, const Arc &b) {
  return a.src_state == b.src_state && \
         a.dest_state == b.dest_state && \
         a.label == b.label && \
         fabs(a.score - b.score) < 1e-6;
}
// clang-format on

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

#define EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, abs_error)           \
  ASSERT_EQ((expected).size(), (actual).size()) << "Different Array Size."; \
  for (std::size_t i = 0; i != (expected).size(); ++i) {                    \
    EXPECT_TRUE(ApproxEqual((expected)[i], (actual)[i], abs_error))         \
        << "expected value at index " << i << " is " << (expected)[i]       \
        << ", but actual value is " << (actual)[i];                         \
  }

// TODO(haowen): remove the double version in host later?
template <typename FloatType>
bool ApproxEqual(FloatType a, FloatType b, double delta = 0.001) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  // a==b handles infinities.
  if (a == b) return true;
  FloatType diff = std::abs(a - b);
  if (diff == std::numeric_limits<FloatType>::infinity() || diff != diff)
    return false;  // diff is +inf or nan.
  return diff <= delta;
}

template <typename T>
void ExpectEqual(const std::vector<T> &expected, const std::vector<T> &actual) {
  EXPECT_EQ(expected, actual);
}

template <>
inline void ExpectEqual<float>(const std::vector<float> &expected,
                               const std::vector<float> &actual) {
  EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, 0.001);
}

template <>
inline void ExpectEqual<double>(const std::vector<double> &expected,
                                const std::vector<double> &actual) {
  EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, 0.001);
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
  ExpectEqual(target_data, array_data);
}

void CheckRowSplits(RaggedShape &shape,
                    const std::vector<std::vector<int32_t>> &target) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    Array1<int32_t> curr_row_splits = shape.RowSplits(i);
    CheckArrayData<int32_t>(curr_row_splits, target[i - 1]);
  }
}

// Return a random acyclic FSA that is NOT topo sorted
Fsa GetRandFsa();

}  // namespace k2

#endif  //  K2_CSRC_TEST_UTILS_H_
