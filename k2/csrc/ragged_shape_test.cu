/**
 * @brief
 * ragged_shape_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/tensor.h"

namespace {
static void CheckRowSplitsOrIds(k2::RaggedShape &shape,
                                const std::vector<std::vector<int32_t>> &target,
                                bool row_splits) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    k2::Array1<int32_t> curr_row_splits =
        row_splits ? shape.RowSplits(i) : shape.RowIds(i);
    // copy data from CPU/GPU to CPU
    auto kind =
        GetMemoryCopyKind(*curr_row_splits.Context(), *k2::GetCpuContext());
    std::vector<int32_t> cpu_data(curr_row_splits.Dim());
    k2::MemoryCopy(static_cast<void *>(cpu_data.data()),
                   static_cast<const void *>(curr_row_splits.Data()),
                   curr_row_splits.Dim() * curr_row_splits.ElementSize(), kind);
    EXPECT_EQ(cpu_data, target[i - 1]);
  }
}
}  // namespace

namespace k2 {
template <DeviceType d>
void TestShape() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // constructed with row_splits and row_ids
    // RaggedTensor4 t = [
    //  [ [[ 1, 2], [4]],  [[3, 0]] ],
    //  [ [[7, 8, 9]], [[6], [3, 5, 7]], [[2]] ],
    //  [ [[3, 4], [], [8]] ]
    // ]
    const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
    const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
    const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
    const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
    const std::vector<int32_t> row_splits3 = {0,  2,  3,  5,  8, 9,
                                              12, 13, 15, 15, 16};
    const std::vector<int32_t> row_ids3 = {0, 0, 1, 2, 2, 3, 3, 3,
                                           4, 5, 5, 5, 6, 7, 7, 9};
    std::vector<RaggedShapeDim> axes;
    axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits1),
                                     Array1<int32_t>(context, row_ids1),
                                     static_cast<int32_t>(row_ids1.size())});
    axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits2),
                                     Array1<int32_t>(context, row_ids2),
                                     static_cast<int32_t>(row_ids2.size())});
    axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits3),
                                     Array1<int32_t>(context, row_ids3),
                                     static_cast<int32_t>(row_ids3.size())});

    RaggedShape shape(axes, true);

    // test NumAxes()
    EXPECT_EQ(shape.NumAxes(), 4);
    // test Dim0()
    EXPECT_EQ(shape.Dim0(), 3);
    // test TotSize()
    EXPECT_EQ(shape.TotSize(0), 3);
    EXPECT_EQ(shape.TotSize(1), row_ids1.size());
    EXPECT_EQ(shape.TotSize(2), row_ids2.size());
    EXPECT_EQ(shape.TotSize(3), row_ids3.size());
    // test NumElements()
    EXPECT_EQ(shape.NumElements(), row_ids3.size());

    // test RowSplits()
    const std::vector<std::vector<int32_t>> row_splits_vec = {
        row_splits1, row_splits2, row_splits3};
    CheckRowSplitsOrIds(shape, row_splits_vec, true);

    // test RowIds()
    const std::vector<std::vector<int32_t>> row_ids_vec = {row_ids1, row_ids2,
                                                           row_ids3};
    CheckRowSplitsOrIds(shape, row_ids_vec, false);

    // test MaxSize()
    EXPECT_EQ(shape.MaxSize(1), 3);
    EXPECT_EQ(shape.MaxSize(2), 3);
    EXPECT_EQ(shape.MaxSize(3), 3);

    // test Index(axis, i)
    {
      // values: [[[ 1, 2], [4]], [[3, 0]]]
      RaggedShape sub_shape = shape.Index(0, 0);
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 2, 3}, {0, 2, 3, 5}};
      CheckRowSplitsOrIds(sub_shape, sub_row_splits_vec, true);
    }
    {
      // values: [[[7, 8, 9]], [[6], [3, 5, 7]], [[2]]]
      RaggedShape sub_shape = shape.Index(0, 1);
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 1, 3, 4}, {0, 3, 4, 7, 8}};
      CheckRowSplitsOrIds(sub_shape, sub_row_splits_vec, true);
    }

    {
      // values: [[[3, 4], [], [8]]]
      RaggedShape sub_shape = shape.Index(0, 2);
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 3}, {0, 2, 2, 3}};
      CheckRowSplitsOrIds(sub_shape, sub_row_splits_vec, true);
    }

    // test operator[](indexes)
    if (d == kCpu) {
      {
        std::vector<int32_t> indexes = {0, 0, 0, 0};
        EXPECT_EQ(shape[indexes], 0);
      }
      {
        std::vector<int32_t> indexes = {0, 1, 0, 0};
        EXPECT_EQ(shape[indexes], 3);
      }
      {
        std::vector<int32_t> indexes = {1, 0, 0, 1};
        EXPECT_EQ(shape[indexes], 6);
      }
      {
        std::vector<int32_t> indexes = {1, 1, 1, 0};
        EXPECT_EQ(shape[indexes], 9);
      }
      {
        std::vector<int32_t> indexes = {2, 0, 0, 1};
        EXPECT_EQ(shape[indexes], 14);
      }
      {
        std::vector<int32_t> indexes = {2, 0, 2, 0};
        EXPECT_EQ(shape[indexes], 15);
      }
    }

    // test To(ctx)
    {
      // to GPU
      RaggedShape other = shape.To(GetCudaContext());
      CheckRowSplitsOrIds(other, row_splits_vec, true);
    }
    {
      // to CPU
      RaggedShape other = shape.To(GetCpuContext());
      CheckRowSplitsOrIds(other, row_splits_vec, true);
    }
  }

  // TODO(haowen): created with only row_splits, need to test
  // `RowIdsFromRowSplits` first test RowIds() and Populate()
}
TEST(RaggedShapeTest, RaggedShape) {
  TestShape<kCuda>();
  TestShape<kCpu>();
}

}  // namespace k2
