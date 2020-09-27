/**
 * @brief
 * ragged_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/tensor.h"

namespace {
// TODO(haowen): may move below functions to some file like `test_utils.h`,
// in case other Tests may use it?
template <typename T>
static void CheckArrayData(const k2::Array1<T> &array,
                           const std::vector<T> &target) {
  ASSERT_EQ(array.Dim(), target.size());
  const T *array_data = array.Data();
  // copy data from CPU/GPU to CPU
  auto kind = k2::GetMemoryCopyKind(*array.Context(), *k2::GetCpuContext());
  std::vector<T> cpu_data(array.Dim());
  k2::MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 array.Dim() * array.ElementSize(), kind);
  EXPECT_EQ(cpu_data, target);
}

static void CheckRowSplits(k2::RaggedShape &shape,
                           const std::vector<std::vector<int32_t>> &target) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    k2::Array1<int32_t> curr_row_splits = shape.RowSplits(i);
    CheckArrayData<int32_t>(curr_row_splits, target[i - 1]);
  }
}
}  // namespace

namespace k2 {
// returns a random ragged shape where the dims on axis 1 are all the same
// (so: can be transposed).
RaggedShape RandomRaggedShapeToTranspose(ContextPtr c) {
  ContextPtr c_cpu = GetCpuContext();

  RaggedShape random = RandomRaggedShape().To(c);

  int32_t input_dim0 = random.Dim0(), divisor = 1;
  for (int32_t i = 1; i * i <= input_dim0; i++) {
    if (input_dim0 % i == 0 && i > divisor) divisor = i;
  }

  int32_t output_dim0 = divisor, output_dim1 = input_dim0 / divisor;

  Array1<int32_t> row_splits =
      Range<int32_t>(c, output_dim0 + 1, 0, output_dim1);
  int32_t cached_tot_size = input_dim0;

  RaggedShape top_level_shape =
      RaggedShape2(&row_splits, nullptr, cached_tot_size);
  return ComposeRaggedShapes(top_level_shape, random);
}

template <DeviceType d>
void TestTranspose() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  RaggedShape to_transpose = RandomRaggedShapeToTranspose(context);
  RaggedShape transposed = Transpose(to_transpose);

  if (d != kCpu) {
    ContextPtr c = GetCpuContext();
    to_transpose = to_transpose.To(c);
    transposed = transposed.To(c);
  }

  for (auto iter = transposed.Iterator(); !iter.Done(); iter.Next()) {
    std::vector<int32_t> index = iter.Value();
    int32_t i = transposed[index];  // Just make sure this doesn't crash, dont
                                    // need the value.
    std::swap(index[0], index[1]);
    i = to_transpose[index];  // don't need the value, just need to make
                              // sure it's an allowable index.
  }
  for (auto iter = to_transpose.Iterator(); !iter.Done(); iter.Next()) {
    std::vector<int32_t> index = iter.Value();
    std::swap(index[0], index[1]);
    int32_t i = transposed[index];  // don't need the value, just need to make
                                    // sure it's an allowable index.
  }
}
TEST(RaggedTest, TestTranspose) {
  // TODO(haowen): make it be a real test: add EXPECT_EQ, test related
  // algorithms, etc.
  // TestTranspose<kCpu>();
  // TestTranspose<kCuda>();
}

template <typename T, DeviceType d>
void TestRagged() {
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
    const std::vector<T> values_vec = {1, 2, 4, 3, 0, 7, 8, 9,
                                       6, 3, 5, 7, 2, 3, 4, 8};
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
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    // test Index(axis, i)
    {
      // values: [[[ 1, 2], [4]], [[3, 0]]]
      Ragged<T> sub_raggged = ragged.Index(0, 0);
      RaggedShape &sub_shape = sub_raggged.shape;
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 2, 3}, {0, 2, 3, 5}};
      CheckRowSplits(sub_shape, sub_row_splits_vec);
      const Array1<T> &sub_values = sub_raggged.values;
      const std::vector<T> sub_values_vec = {1, 2, 4, 3, 0};
      CheckArrayData<T>(sub_values, sub_values_vec);
    }
    {
      // values: [[[7, 8, 9]], [[6], [3, 5, 7]], [[2]]]
      Ragged<T> sub_raggged = ragged.Index(0, 1);
      RaggedShape &sub_shape = sub_raggged.shape;
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 1, 3, 4}, {0, 3, 4, 7, 8}};
      CheckRowSplits(sub_shape, sub_row_splits_vec);
      const Array1<T> &sub_values = sub_raggged.values;
      const std::vector<T> sub_values_vec = {7, 8, 9, 6, 3, 5, 7, 2};
      CheckArrayData<T>(sub_values, sub_values_vec);
    }
    {
      // values: [[[3, 4], [], [8]]]
      Ragged<T> sub_raggged = ragged.Index(0, 2);
      RaggedShape &sub_shape = sub_raggged.shape;
      EXPECT_EQ(sub_shape.NumAxes(), 3);
      const std::vector<std::vector<int32_t>> sub_row_splits_vec = {
          {0, 3}, {0, 2, 2, 3}};
      CheckRowSplits(sub_shape, sub_row_splits_vec);
      const Array1<T> &sub_values = sub_raggged.values;
      const std::vector<T> sub_values_vec = {3, 4, 8};
      CheckArrayData<T>(sub_values, sub_values_vec);
    }

    // test operator[](const std::vector<int32_t> &indexes)
    if (d == kCpu) {
      {
        std::vector<int32_t> indexes = {0, 0, 0, 0};
        EXPECT_EQ(ragged.shape[indexes], 0);
        EXPECT_EQ(ragged[indexes], 1);
      }
      {
        std::vector<int32_t> indexes = {0, 1, 0, 0};
        EXPECT_EQ(ragged.shape[indexes], 3);
        EXPECT_EQ(ragged[indexes], 3);
      }
      {
        std::vector<int32_t> indexes = {1, 0, 0, 1};
        EXPECT_EQ(ragged.shape[indexes], 6);
        EXPECT_EQ(ragged[indexes], 8);
      }
      {
        std::vector<int32_t> indexes = {1, 1, 1, 0};
        EXPECT_EQ(ragged.shape[indexes], 9);
        EXPECT_EQ(ragged[indexes], 3);
      }
      {
        std::vector<int32_t> indexes = {2, 0, 0, 1};
        EXPECT_EQ(ragged.shape[indexes], 14);
        EXPECT_EQ(ragged[indexes], 4);
      }
      {
        std::vector<int32_t> indexes = {2, 0, 2, 0};
        EXPECT_EQ(ragged.shape[indexes], 15);
        EXPECT_EQ(ragged[indexes], 8);
      }
    }

    const std::vector<std::vector<int32_t>> row_splits_vec = {
        row_splits1, row_splits2, row_splits3};
    // test To(ctx)
    {
      // to GPU
      Ragged<T> other = ragged.To(GetCudaContext());
      CheckRowSplits(other.shape, row_splits_vec);
      CheckArrayData<T>(other.values, values_vec);
    }
    {
      // to CPU
      Ragged<T> other = ragged.To(GetCpuContext());
      CheckRowSplits(other.shape, row_splits_vec);
      CheckArrayData<T>(other.values, values_vec);
    }
  }
}

template <typename T, typename OP = LessThan<T>>
static void CpuSortSublists(const Array1<int32_t> &row_splits, Array1<T> *src) {
  K2_CHECK(src->Context()->GetDeviceType() == kCpu);
  T *p = src->Data();
  OP comp = OP();
  for (int32_t i = 0; i < row_splits.Dim() - 1; ++i) {
    int32_t cur = row_splits[i];
    int32_t next = row_splits[i + 1];
    std::sort(p + cur, p + next, comp);
  }
}

template <typename T, typename OP = LessThan<T>>
static void TestSortSublists() {
  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();

  RaggedShape shape = RandomRaggedShape(false,  // set_row_ids
                                        2,      // min_num_axes
                                        4,      // max_num_axes
                                        1,      // min_num_elements
                                        2000);  // max_num_elements

  Array1<T> values =
      RandUniformArray1<T>(shape.Context(), shape.NumElements(), -2000, 2000);
  Ragged<T> ragged(shape, values);
  ragged = ragged.To(cuda_context);
  values = values.To(cpu_context);  // to be sorted by cpu

  // TODO(fangjun): add a `Clone` method to Array1<T>
  Array1<T> unsorted = values.To(cuda_context).To(cpu_context);

  Array1<int32_t> order(ragged.Context(), ragged.values.Dim());
  SortSublists<T, OP>(&ragged, &order);

  Array1<int32_t> &segment = ragged.shape.RowSplits(ragged.NumAxes() - 1);
  CpuSortSublists<T, OP>(segment, &values);

  int32_t n = order.Dim();
  for (int i = 0; i != n; ++i) {
    EXPECT_EQ(values[i], ragged.values[i]);
    EXPECT_EQ(ragged.values[i], unsorted[order[i]]);
  }
}

TEST(RaggedTest, Ragged) {
  TestRagged<int32_t, kCuda>();
  TestRagged<int32_t, kCpu>();
  TestRagged<double, kCuda>();
  TestRagged<double, kCpu>();

  TestSortSublists<int32_t>();
  TestSortSublists<double>();
}

// TODO(Haowen): add more tests for other algorithms

}  // namespace k2
