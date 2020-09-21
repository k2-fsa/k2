/**
 * @brief
 * utils_test
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
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

TEST(UtilsTest, CpuExclusiveSum) {
  void *deleter_context;
  ContextPtr c = GetCpuContext();
  int32_t n = 5;
  // [0, 1, 2, 3, 4]
  // the exclusive prefix sum is [0, 0, 1, 3, 6]
  auto *src = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  std::iota(src, src + n, 0);

  auto *dst = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  ExclusiveSum(c, n, src, dst);

  EXPECT_THAT(std::vector<int32_t>(dst, dst + n),
              ::testing::ElementsAre(0, 0, 1, 3, 6));

  c->Deallocate(dst, deleter_context);
  c->Deallocate(src, deleter_context);
}

TEST(UtilsTest, CudaExclusiveSum) {
  void *deleter_context;
  ContextPtr c = GetCudaContext();
  int32_t n = 5;
  auto *src = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));

  std::vector<int32_t> h(n);
  std::iota(h.begin(), h.end(), 0);
  cudaMemcpy(src, h.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice);

  auto *dst = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  ExclusiveSum(c, n, src, dst);

  cudaMemcpy(h.data(), dst, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);

  EXPECT_THAT(h, ::testing::ElementsAre(0, 0, 1, 3, 6));

  c->Deallocate(dst, deleter_context);
  c->Deallocate(src, deleter_context);
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
