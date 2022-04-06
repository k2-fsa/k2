/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_TEST_UTILS_H_
#define K2_CSRC_TEST_UTILS_H_


#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/log.h"

namespace k2 {

template <typename T>
void CheckArrayData(const Array1<T> &array, const std::vector<T> &target) {
  ASSERT_EQ(array.Dim(), target.size());
  const T *array_data = array.Data();
  // copy data from CPU/GPU to CPU
  std::vector<T> cpu_data(array.Dim());
  array.Context()->CopyDataTo(array.Dim() * array.ElementSize(), array_data,
                              GetCpuContext(), cpu_data.data());
  EXPECT_EQ(cpu_data, target);
}

#define EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, abs_error)           \
  ASSERT_EQ((expected).size(), (actual).size()) << "Different Array Size."; \
  for (std::size_t i = 0; i != (expected).size(); ++i) {                    \
    EXPECT_TRUE(ApproxEqual((expected)[i], (actual)[i], abs_error))         \
        << "expected value at index " << i << " is " << (expected)[i]       \
        << ", but actual value is " << (actual)[i];                         \
  }

#define EXPECT_FLOAT_ARRAY2_APPROX_EQ(expected, actual, abs_error)           \
  ASSERT_EQ((expected).Dim0(), (actual).Dim0()) << "Different Array2 Size."; \
  ASSERT_EQ((expected).Dim1(), (actual).Dim1()) << "Different Array2 Size."; \
  for (int32_t i = 0; i != (expected).Dim0(); ++i) {                         \
    for (int32_t j = 0; j != (expected).Dim1(); ++j) {                       \
      EXPECT_TRUE(                                                           \
          ApproxEqual((expected).Row(i)[j], (actual).Row(i)[j], abs_error))  \
          << "expected value at index (" << i << "," << j << ") is "         \
          << (expected).Row(i)[j] << ", but actual value is "                \
          << (actual).Row(i)[j];                                             \
    }                                                                        \
  }

// TODO(haowen): remove the double version in host later?
template <typename FloatType>
bool ApproxEqual(FloatType a, FloatType b, FloatType delta = (FloatType)0.001) {
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
void ExpectEqual(const std::vector<T> &expected, const std::vector<T> &actual,
                 T abs_error = (T)0.001) {
  // noted abs_error is not used here, but will be used in below instantiation
  // for double and float, so that when we call `ExpectEqual` in below
  // `CheckArrayData` we don't need a if-else branch based on `T`.
  EXPECT_EQ(expected, actual);
}

template <>
inline void ExpectEqual<float>(const std::vector<float> &expected,
                               const std::vector<float> &actual,
                               float abs_error) {
  EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, abs_error);
}

template <>
inline void ExpectEqual<double>(const std::vector<double> &expected,
                                const std::vector<double> &actual,
                                double abs_error) {
  EXPECT_FLOAT_ARRAY_APPROX_EQ(expected, actual, abs_error);
}

// check if `array` and `target` have the same values
template <typename T>
void CheckArrayData(const Array1<T> &array, const Array1<T> &target,
                    T abs_error = T(0.001)) {
  if (array.Dim() != target.Dim()) {
    K2_LOG(FATAL) << "Dims mismatch " << array.Dim() << " vs. " << target.Dim();
  }
  int32_t dim = array.Dim();
  ContextPtr cpu = GetCpuContext();
  Array1<T> cpu_array = array.To(cpu);
  Array1<T> cpu_target = target.To(cpu);
  std::vector<T> array_data(cpu_array.Data(), cpu_array.Data() + dim);
  std::vector<T> target_data(cpu_target.Data(), cpu_target.Data() + dim);
  ExpectEqual(target_data, array_data, abs_error);
}

inline void CheckRowSplits(RaggedShape &shape,
                           const std::vector<std::vector<int32_t>> &target) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    Array1<int32_t> curr_row_splits = shape.RowSplits(i);
    CheckArrayData<int32_t>(curr_row_splits, target[i - 1]);
  }
}

// Return a random acyclic FSA that is NOT topologically sorted
Fsa GetRandFsaNotTopSorted();

/* Return 1-D array filled with random values.

   @param [in] context  The device context specifying where the returned
                        array resides.
   @param [in] allow_minus_one
                        If true, the returned array will contain values
                        in the range [-1, max_value]; [0, max_value] otherwise.
   @param [in] dim        It specifies the length of the returned array.
   @param [in] max_value  It specifies the maximum value the returned array can
                          contain.

   @return  Return a 1-D array with the given `dim`.
 */
Array1<int32_t> GenerateRandomIndexes(ContextPtr context, bool allow_minus_one,
                                      int32_t dim, int32_t max_value);

}  // namespace k2

#endif  //  K2_CSRC_TEST_UTILS_H_
