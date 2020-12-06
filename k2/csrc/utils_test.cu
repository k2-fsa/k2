/**
 * @brief
 * utils_test
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/utils.h"

namespace k2 {
template <typename T>
void TestExclusiveSum() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
}

TEST(UtilsTest, ExclusiveSum) {
  TestExclusiveSum<int32_t>();
  TestExclusiveSum<double>();

  // TODO(haowen): add tests where output type differs from input type?
}

template <typename T>
void TestMaxValue() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    {
      // tests with larger random size
      int32_t max_elem = 10000;
      int32_t num_elems = RandInt(2000, max_elem);
      std::vector<int32_t> data(num_elems);
      std::iota(data.begin(), data.end(), 0);
      // shuffle data
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(data.begin(), data.end(), g);
      // assign max_elem to a random position
      int32_t pos = RandInt(0, num_elems - 1);
      data[pos] = max_elem;

      Array1<int32_t> src(context, data);
      int32_t max_value = MaxValue(context, src.Dim(), src.Data());
      EXPECT_EQ(max_value, max_elem);
    }
  }
}

TEST(UtilsTest, MaxValue) {
  TestMaxValue<int32_t>();
  TestMaxValue<double>();
}

TEST(UtilsTest, RowSplitsToRowIds) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    {
      // test with random large size
      const int32_t min_num_elements = 2000;
      RaggedShape shape =
          RandomRaggedShape(true, 2, 2, min_num_elements, 10000);
      ASSERT_EQ(shape.NumAxes(), 2);
      const auto &axes = shape.Layers();
      ASSERT_EQ(axes.size(), 1);
      // note `src_row_splits` is on CPU as it is created with RandomRaggedShape
      const Array1<int32_t> &src_row_splits = axes[0].row_splits;
      Array1<int32_t> row_splits = src_row_splits.To(context);
      // don't call `shape.RowIds(axis)` here, it will make the test meaningless
      // as `shape.RowId(axis)` calls `RowSplitsToRowIds()` internally.
      // note `row_ids` is on CPU as it is created with RandomRaggedShape
      const Array1<int32_t> &expected_row_ids = axes[0].row_ids;
      int32_t num_rows = row_splits.Dim() - 1;
      int32_t num_elements = row_splits[num_rows];
      ASSERT_GE(num_elements, min_num_elements);
      ASSERT_EQ(expected_row_ids.Dim(), num_elements);
      Array1<int32_t> row_ids(context, num_elements);
      int32_t *row_ids_data = row_ids.Data();
      EXPECT_EQ(row_ids.Dim(), num_elements);
      RowSplitsToRowIds(context, num_rows, row_splits.Data(), num_elements,
                        row_ids_data);
      // copy data from CPU/GPU to CPU
      Array1<int32_t> cpu_array = row_ids.To(cpu);
      std::vector<int32_t> cpu_data(cpu_array.Data(),
                                    cpu_array.Data() + cpu_array.Dim());
      std::vector<int32_t> expected_row_ids_data(
          expected_row_ids.Data(),
          expected_row_ids.Data() + expected_row_ids.Dim());
      EXPECT_EQ(cpu_data, expected_row_ids_data);
    }
  }
}

TEST(UtilsTest, RowIdsToRowSplits) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    {
      // test with random large size
      const int32_t min_num_elements = 2000;
      RaggedShape shape =
          RandomRaggedShape(true, 2, 2, min_num_elements, 10000);
      ASSERT_EQ(shape.NumAxes(), 2);
      const auto &axes = shape.Layers();
      ASSERT_EQ(axes.size(), 1);
      // note `src_row_ids` is on CPU as it is created with RandomRaggedShape
      const Array1<int32_t> &src_row_ids = axes[0].row_ids;
      Array1<int32_t> row_ids = src_row_ids.To(context);
      // note `row_splits` is on CPU as it is created with RandomRaggedShape
      const Array1<int32_t> &expected_row_splits = axes[0].row_splits;
      int32_t num_elements = row_ids.Dim();
      ASSERT_GE(num_elements, min_num_elements);
      int32_t num_rows = expected_row_splits.Dim() - 1;
      Array1<int32_t> row_splits(context, num_rows + 1);
      EXPECT_EQ(row_splits.Dim(), num_rows + 1);
      int32_t *row_splits_data = row_splits.Data();
      RowIdsToRowSplits(context, num_elements, row_ids.Data(), false, num_rows,
                        row_splits_data);
      // copy data from CPU/GPU to CPU
      Array1<int32_t> cpu_array = row_splits.To(cpu);
      std::vector<int32_t> cpu_data(cpu_array.Data(),
                                    cpu_array.Data() + cpu_array.Dim());
      std::vector<int32_t> expected_row_splits_data(
          expected_row_splits.Data(),
          expected_row_splits.Data() + expected_row_splits.Dim());
      EXPECT_EQ(cpu_data, expected_row_splits_data);
    }
  }
}
}  // namespace k2
