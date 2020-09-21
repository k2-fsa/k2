/**
 * @brief
 * utils_test
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/utils.h"

namespace k2 {
template <typename T, DeviceType d>
void TestExclusiveSum() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    std::vector<T> data(5);
    std::iota(data.begin(), data.end(), 0);
    EXPECT_THAT(data, ::testing::ElementsAre(0, 1, 2, 3, 4));
    Array1<T> src(context, data);
    Array1<T> dst(context, src.Dim());
    T *dst_data = dst.Data();
    ExclusiveSum(context, src.Dim(), src.Data(), dst.Data());
    // copy data from CPU/GPU to CPU
    Array1<T> cpu_array = dst.To(cpu);
    std::vector<T> cpu_data(cpu_array.Data(),
                            cpu_array.Data() + cpu_array.Dim());
    EXPECT_THAT(cpu_data, ::testing::ElementsAre(0, 0, 1, 3, 6));
  }
}

TEST(UtilsTest, ExclusiveSum) {
  TestExclusiveSum<int32_t, kCpu>();
  TestExclusiveSum<int32_t, kCuda>();
  TestExclusiveSum<double, kCpu>();
  TestExclusiveSum<double, kCuda>();

  // TODO(haowen): add tests where output type differs from input type?
}

template <typename T, DeviceType d>
void TestMaxValue() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // empty input array
    std::vector<T> data;
    Array1<T> src(context, data);
    T max_value = MaxValue(context, src.Dim(), src.Data());
    EXPECT_EQ(max_value, 0);
  }

  {
    // no empty input array
    std::vector<T> data = {0, 1, 5, 7, 8, 1};
    Array1<T> src(context, data);
    T max_value = MaxValue(context, src.Dim(), src.Data());
    EXPECT_EQ(max_value, 8);
  }

  // TODO(haowen): tests with larger random size
}

TEST(UtilsTest, MaxValue) {
  TestMaxValue<int32_t, kCpu>();
  TestMaxValue<int32_t, kCuda>();
  TestMaxValue<double, kCpu>();
  TestMaxValue<double, kCuda>();
}

template <DeviceType d>
void TestRowSplitsToRowIds() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // test empty case
    const std::vector<int32_t> row_splits_vec = {0};
    const std::vector<int32_t> row_ids_vec;
    Array1<int32_t> row_splits(context, row_splits_vec);
    int32_t num_rows = row_splits.Dim() - 1;
    int32_t num_elements = row_splits[num_rows];
    Array1<int32_t> row_ids(context, num_elements);
    int32_t *row_ids_data = row_ids.Data();
    EXPECT_EQ(row_ids.Dim(), num_elements);
    // just run to check if there is any error
    RowSplitsToRowIds(context, num_rows, row_splits.Data(), num_elements,
                      row_ids_data);
  }

  {
    const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                 12, 13, 15, 15, 16};
    const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                              4, 5, 5, 5, 6, 7, 7, 9};
    Array1<int32_t> row_splits(context, row_splits_vec);
    int32_t num_rows = row_splits.Dim() - 1;
    int32_t num_elements = row_splits[num_rows];
    Array1<int32_t> row_ids(context, num_elements);
    int32_t *row_ids_data = row_ids.Data();
    EXPECT_EQ(row_ids.Dim(), num_elements);
    RowSplitsToRowIds(context, num_rows, row_splits.Data(), num_elements,
                      row_ids_data);
    // copy data from CPU/GPU to CPU
    Array1<int32_t> cpu_array = row_ids.To(cpu);
    std::vector<int32_t> cpu_data(cpu_array.Data(),
                                  cpu_array.Data() + cpu_array.Dim());
    EXPECT_EQ(cpu_data, row_ids_vec);
  }
}

TEST(UtilsTest, RowSplitsToRowIds) {
  TestRowSplitsToRowIds<kCpu>();
  TestRowSplitsToRowIds<kCuda>();
}

template <DeviceType d>
void TestRowIdsToRowSplits() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // test empty case
    const std::vector<int32_t> row_ids_vec;
    const std::vector<int32_t> row_splits_vec;
    Array1<int32_t> row_ids(context, row_ids_vec);
    int32_t num_rows = 0;
    int32_t num_elements = row_ids.Dim();
    Array1<int32_t> row_splits(context, num_rows + 1);
    int32_t *row_splits_data = row_splits.Data();
    RowIdsToRowSplits(context, num_elements, row_ids.Data(), true, num_rows,
                      row_splits_data);
    EXPECT_EQ(row_splits[0], 0);
  }

  {
    // no empty rows
    const std::vector<int32_t> row_ids_vec = {0, 0, 1, 1, 1, 2};
    const std::vector<int32_t> row_splits_vec = {0, 2, 5, 6};
    Array1<int32_t> row_ids(context, row_ids_vec);
    int32_t num_elements = row_ids.Dim();
    int32_t num_rows = row_ids[num_elements - 1] + 1;
    Array1<int32_t> row_splits(context, num_rows + 1);
    EXPECT_EQ(row_splits.Dim(), num_rows + 1);
    int32_t *row_splits_data = row_splits.Data();
    RowIdsToRowSplits(context, num_elements, row_ids.Data(), true, num_rows,
                      row_splits_data);
    // copy data from CPU/GPU to CPU
    Array1<int32_t> cpu_array = row_splits.To(cpu);
    std::vector<int32_t> cpu_data(cpu_array.Data(),
                                  cpu_array.Data() + cpu_array.Dim());
    EXPECT_EQ(cpu_data, row_splits_vec);
  }

  {
    // has empty rows
    const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                 12, 13, 15, 15, 16};
    const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                              4, 5, 5, 5, 6, 7, 7, 9};
    Array1<int32_t> row_ids(context, row_ids_vec);
    int32_t num_elements = row_ids.Dim();
    int32_t num_rows = row_ids[num_elements - 1] + 1;
    Array1<int32_t> row_splits(context, num_rows + 1);
    EXPECT_EQ(row_splits.Dim(), num_rows + 1);
    int32_t *row_splits_data = row_splits.Data();
    RowIdsToRowSplits(context, num_elements, row_ids.Data(), false, num_rows,
                      row_splits_data);
    // copy data from CPU/GPU to CPU
    Array1<int32_t> cpu_array = row_splits.To(cpu);
    std::vector<int32_t> cpu_data(cpu_array.Data(),
                                  cpu_array.Data() + cpu_array.Dim());
    EXPECT_EQ(cpu_data, row_splits_vec);
  }
}

TEST(UtilsTest, RowIdsToRowSplits) {
  TestRowIdsToRowSplits<kCpu>();
  TestRowIdsToRowSplits<kCuda>();
}
}  // namespace k2
