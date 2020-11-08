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
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

class RaggedShapeOpsSuiteTest : public ::testing::Test {
 protected:
  RaggedShapeOpsSuiteTest() {
    ContextPtr context = GetCpuContext();
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

    simple_shape_ = RaggedShape(axes, true);

    // random_shape_ is on CPU
    random_shape_ = RandomRaggedShape(true,   // set_row_ids
                                      3,      // min_num_axes
                                      4,      // max_num_axes
                                      0,      // min_num_elements
                                      1000);  // max_num_elements
  }

  RaggedShape simple_shape_;
  RaggedShape random_shape_;
};


TEST(RaggedShapeTest, TestConstructFromString) {
  RaggedShape rs(" [ [ x x ] [x] ]");
  Array1<int32_t> row_splits1(GetCpuContext(), std::vector<int32_t>{0, 2, 3});
  K2_LOG(INFO) << rs.RowSplits(1);
  K2_CHECK(Equal(rs.RowSplits(1), row_splits1));

  RaggedShape rs2(" [ [ [ x x ] ] [[x]] ]");
  K2_LOG(INFO) << "rs2 = " << rs2;

  K2_CHECK_EQ(RaggedShape("[ ]").Dim0(), 0);

  ASSERT_DEATH(RaggedShape(" [ [ x x ] [x] "), "");
  ASSERT_DEATH(RaggedShape(" [ [ x x ] [[x]]] "), "");
  ASSERT_DEATH(RaggedShape(" [ [ x [] x ] "), "");
  ASSERT_DEATH(RaggedShape(" [ x ] "), "");
  ASSERT_DEATH(RaggedShape(" [ x ] [ x ] "), "");
  ASSERT_DEATH(RaggedShape(" [ x | x ] "), "");

  for (int i = 0; i < 5; i++) {
    RaggedShape rs = RandomRaggedShape(true,
                                       2,      // min_num_axes
                                       4,      // max_num_axes
                                       0,      // min_num_elements
                                       1000);  // max_num_elements
    std::ostringstream os;
    os << rs;
    RaggedShape rs2;
    std::istringstream is(os.str());
    K2_LOG(INFO) << "Shape is: " << os.str();
    is >> rs2;
    K2_CHECK(is.good());
    // the reason for the || below is that in "[ ]", the number of
    // axes is ambiguous; we assume 2.
    K2_CHECK(Equal(rs, rs2) || rs.NumElements() == 0);
  }
}



TEST(RaggedTest, TestRaggedFromString) {
  Ragged<int32_t> rs(" [ [ 1 2 ] [3] ]");
  Array1<int32_t> row_splits1(GetCpuContext(), std::vector<int32_t>{0, 2, 3});
  K2_LOG(INFO) << rs.RowSplits(1);
  K2_CHECK(Equal(rs.RowSplits(1), row_splits1));
  K2_CHECK_EQ(rs.values.Back(), 3);
  K2_CHECK_EQ(rs.values[0], 1);

  Ragged<int32_t>  rs2(" [ [ [ 0 5 ] ] [[10]] ]");
  K2_LOG(INFO) << "rs2 = " << rs2;

  ASSERT_DEATH(RaggedShape(" [ [ 0 0 ] [0] "), "");
  ASSERT_DEATH(RaggedShape(" [ [ 0 0 ] [[0]]] "), "");
  ASSERT_DEATH(RaggedShape(" [ [ 0 [] 0 ] "), "");
  ASSERT_DEATH(RaggedShape(" [ 0 ] "), "");
  ASSERT_DEATH(RaggedShape(" [ 0 ] [ 0 ] "), "");
  ASSERT_DEATH(RaggedShape(" [ 0 | 0 ] "), "");

  for (int32_t i = 0; i < 5; i++) {
    Ragged<int32_t> r = RandomRagged<int32_t>();
    std::ostringstream os;
    os << r;
    Ragged<int32_t> r2(os.str());
    // the reason for the || below is that in "[ ]", the number of
    // axes is ambiguous; we assume 2.
    K2_CHECK(Equal(r, r2) || r.values.Dim() == 0);
  }
}





template <typename T, DeviceType d>
void TestMaxPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // empty case
    const std::vector<int32_t> row_splits = {0};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = 0;
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    Array1<T> values(context, 0);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    ASSERT_EQ(num_rows, 0);
    Array1<T> max_values(context, num_rows);
    // just run to check if there's any error
    MaxPerSublist(ragged, 1, &max_values);
    EXPECT_EQ(max_values.Dim(), 0);
  }

  {
    const std::vector<int32_t> row_splits = {0, 2, 2, 5, 6};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = row_splits.back();
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    const std::vector<T> values_vec = {1, 3, 2, 8, 0, -1};
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    Array1<T> max_values(context, num_rows);
    T default_value = 2;
    MaxPerSublist(ragged, default_value, &max_values);
    // copy memory from GPU/CPU to CPU
    std::vector<T> cpu_data(max_values.Dim());
    auto kind = GetMemoryCopyKind(*max_values.Context(), *cpu);
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<const void *>(max_values.Data()),
               max_values.Dim() * max_values.ElementSize(), kind, nullptr);
    std::vector<T> expected_data = {3, default_value, 8, default_value};
    EXPECT_EQ(cpu_data, expected_data);
  }

  {
    // test with random large size
    const int32_t min_num_elements = 2000;
    // not random shape is on CPU
    RaggedShape shape = RandomRaggedShape(false, 2, 2, min_num_elements, 5000);
    ASSERT_EQ(shape.NumAxes(), 2);
    RaggedShape gpu_shape;
    if (d == kCuda) {
      // copy shape to GPU
      const Array1<T> &row_splits = shape.RowSplits(1);
      RaggedShapeDim shape_dim;
      shape_dim.row_splits = row_splits.To(GetCudaContext());
      shape_dim.cached_tot_size = shape.NumElements();
      std::vector<RaggedShapeDim> axes = {shape_dim};
      gpu_shape = RaggedShape(axes, true);
    }

    int32_t num_elems = shape.NumElements();
    std::vector<T> data(num_elems);
    for (int32_t i = 0; i != 10; ++i) {
      std::iota(data.begin(), data.end(), 0);
      // randomly set data[pos] = num_elems which is
      // greater than any element in data
      int32_t pos = RandInt(0, num_elems - 1);
      data[pos] = num_elems;
      // find the corresponding row
      int32_t num_rows = shape.Dim0();
      const int32_t *row_splits_data = shape.RowSplits(1).Data();
      int32_t row = 0;
      for (int32_t i = 0; i < num_rows; ++i) {
        if (pos >= row_splits_data[i] && pos < row_splits_data[i + 1]) {
          row = i;
          break;
        }
      }

      Array1<T> values(context, data);
      Ragged<T> ragged(d == kCuda ? gpu_shape : shape, values);
      Array1<T> max_values(context, num_rows);
      T default_value = 0;
      MaxPerSublist(ragged, default_value, &max_values);
      EXPECT_EQ(max_values[row], num_elems);
    }
  }
}

TEST(RaggedShapeOpsTest, MaxPerSubListTest) {
  TestMaxPerSubListTest<int32_t, kCpu>();
  TestMaxPerSubListTest<int32_t, kCuda>();
}

template <typename T, DeviceType d>
void TestMinPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // empty case
    std::vector<int32_t> row_splits_vec = {0};
    Array1<T> row_splits(context, row_splits_vec);
    RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
    Array1<T> values(context, 0);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    ASSERT_EQ(num_rows, 0);
    Array1<T> min_values(context, num_rows);
    // just run to check if there's any error
    MinPerSublist(ragged, 1, &min_values);
    EXPECT_EQ(min_values.Dim(), 0);
  }

  {
    std::vector<int32_t> row_splits_vec = {0, 2, 2, 5, 6};
    Array1<T> row_splits(context, row_splits_vec);
    RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
    const std::vector<T> values_vec = {1, 3, 3, 8, 4, -1};
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    Array1<T> min_values(context, num_rows);
    T default_value = 2;
    MinPerSublist(ragged, default_value, &min_values);
    // copy memory from GPU/CPU to CPU
    min_values = min_values.To(cpu);
    std::vector<T> cpu_data(min_values.Data(),
                            min_values.Data() + min_values.Dim());
    std::vector<T> expected_data = {1, default_value, default_value, -1};
    EXPECT_EQ(cpu_data, expected_data);
  }

  // May add tests for random large size? (but maybe it's fine to not add as we
  // have tested large cases in MaxPerSubList)
}

TEST(RaggedShapeOpsTest, MinPerSubListTest) {
  TestMinPerSubListTest<int32_t, kCpu>();
  TestMinPerSubListTest<int32_t, kCuda>();
}

template <typename T, DeviceType d>
void TestAndOrPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // And
    const std::vector<int32_t> row_splits = {0, 2, 2, 5, 6};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = row_splits.back();
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    const std::vector<T> values_vec = {1, 3, 3, 6, 11, 0};
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    Array1<T> dst(context, num_rows);
    T default_value = -1;
    AndPerSublist(ragged, default_value, &dst);
    // copy memory from GPU/CPU to CPU
    dst = dst.To(cpu);
    std::vector<T> cpu_data(dst.Data(), dst.Data() + dst.Dim());
    std::vector<T> expected_data = {1, -1, 2, 0};
    EXPECT_EQ(cpu_data, expected_data);
  }

  {
    // Or
    const std::vector<int32_t> row_splits = {0, 2, 2, 5, 6};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = row_splits.back();
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    const std::vector<T> values_vec = {1, 3, 3, 4, 6, 0};
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    Array1<T> dst(context, num_rows);
    T default_value = 0;
    OrPerSublist(ragged, default_value, &dst);
    // copy memory from GPU/CPU to CPU
    dst = dst.To(cpu);
    std::vector<T> cpu_data(dst.Data(), dst.Data() + dst.Dim());
    std::vector<T> expected_data = {3, 0, 7, 0};
    EXPECT_EQ(cpu_data, expected_data);
  }
}

TEST(RagedShapeOpsTest, AndOrPerSubListTest) {
  TestAndOrPerSubListTest<int32_t, kCpu>();
  TestAndOrPerSubListTest<int32_t, kCuda>();
}

void TestUnsqueeze(ContextPtr context, const RaggedShape &input_shape) {
  RaggedShape src_shape = input_shape.To(context);
  src_shape.Populate();  // set row_ids
  {
    // axis = 0.
    RaggedShape shape = Unsqueeze(src_shape, 0);
    int32_t dim0 = src_shape.Dim0();
    const std::vector<RaggedShapeDim> &src_axes = src_shape.Axes();
    const std::vector<RaggedShapeDim> &dest_axes = shape.Axes();

    {
      const Array1<int32_t> &row_splits0 = dest_axes[0].row_splits;
      std::vector<int32_t> data = {0, dim0};
      CheckArrayData(row_splits0, data);
    }

    {
      const Array1<int32_t> &row_ids0 = dest_axes[0].row_ids;
      std::vector<int32_t> data(dim0, 0);
      CheckArrayData(row_ids0, data);
    }

    {
      for (auto i = 0; i != src_axes.size(); ++i) {
        CheckArrayData(src_axes[i].row_splits, dest_axes[i + 1].row_splits);
        CheckArrayData(src_axes[i].row_ids, dest_axes[i + 1].row_ids);
      }
    }
  }

  {
    // axis = 1
    int32_t axis = 1;
    RaggedShape shape = Unsqueeze(src_shape, axis);
    int32_t tot_size = shape.TotSize(axis);
    const std::vector<RaggedShapeDim> &src_axes = src_shape.Axes();
    const std::vector<RaggedShapeDim> &dest_axes = shape.Axes();

    {
      for (auto i = 0; i < axis; ++i) {
        CheckArrayData(src_axes[i].row_splits, dest_axes[i].row_splits);
        CheckArrayData(src_axes[i].row_ids, dest_axes[i].row_ids);
      }
    }

    {
      const Array1<int32_t> &row_splits = dest_axes[axis].row_splits;
      std::vector<int32_t> data(tot_size + 1);
      std::iota(data.begin(), data.end(), 0);
      CheckArrayData(row_splits, data);
    }

    {
      const Array1<int32_t> &row_ids = dest_axes[axis].row_ids;
      std::vector<int32_t> data(tot_size);
      std::iota(data.begin(), data.end(), 0);
      CheckArrayData(row_ids, data);
    }

    {
      for (auto i = axis; i < src_axes.size(); ++i) {
        CheckArrayData(src_axes[i].row_splits, dest_axes[i + 1].row_splits);
        CheckArrayData(src_axes[i].row_ids, dest_axes[i + 1].row_ids);
      }
    }
  }
}

TEST_F(RaggedShapeOpsSuiteTest, TestUnsqueezeCpu) {
  TestUnsqueeze(GetCpuContext(), simple_shape_);
  TestUnsqueeze(GetCpuContext(), random_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestUnsqueezeGpu) {
  TestUnsqueeze(GetCudaContext(), simple_shape_);
  TestUnsqueeze(GetCudaContext(), random_shape_);
}

void TestRemoveAxis(ContextPtr context, const RaggedShape &input_shape) {
  RaggedShape src_shape = input_shape.To(context);
  ASSERT_EQ(src_shape.NumAxes(), 4);
  {
    // axis = 0.
    int32_t axis = 0;
    RaggedShape shape = RemoveAxis(src_shape, axis);
    const std::vector<RaggedShapeDim> &src_axes = src_shape.Axes();
    const std::vector<RaggedShapeDim> &dest_axes = shape.Axes();
    ASSERT_EQ(src_axes.size(), 3);
    ASSERT_EQ(dest_axes.size(), 2);

    {
      for (auto i = 0; i != dest_axes.size(); ++i) {
        CheckArrayData(dest_axes[i].row_splits, src_axes[i + 1].row_splits);
        CheckArrayData(dest_axes[i].row_ids, src_axes[i + 1].row_ids);
      }
    }
  }

  {
    // axis = 1
    int32_t axis = 1;
    RaggedShape shape = RemoveAxis(src_shape, axis);
    const std::vector<RaggedShapeDim> &src_axes = src_shape.Axes();
    const std::vector<RaggedShapeDim> &dest_axes = shape.Axes();
    ASSERT_EQ(src_axes.size(), 3);
    ASSERT_EQ(dest_axes.size(), 2);

    {
      const Array1<int32_t> &row_splits0 = dest_axes[0].row_splits;
      std::vector<int32_t> data = {0, 3, 7, 10};
      CheckArrayData(row_splits0, data);
    }

    {
      const Array1<int32_t> &row_ids0 = dest_axes[0].row_ids;
      std::vector<int32_t> data = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
      CheckArrayData(row_ids0, data);
    }

    {
      for (auto i = 1; i != dest_axes.size(); ++i) {
        CheckArrayData(dest_axes[i].row_splits, src_axes[i + 1].row_splits);
        CheckArrayData(dest_axes[i].row_ids, src_axes[i + 1].row_ids);
      }
    }
  }

  {
    // axis = 3
    int32_t axis = 3;  // the last axis
    RaggedShape shape = RemoveAxis(src_shape, axis);
    const std::vector<RaggedShapeDim> &src_axes = src_shape.Axes();
    const std::vector<RaggedShapeDim> &dest_axes = shape.Axes();
    ASSERT_EQ(src_axes.size(), 3);
    ASSERT_EQ(dest_axes.size(), 2);

    {
      for (auto i = 0; i != dest_axes.size(); ++i) {
        CheckArrayData(dest_axes[i].row_splits, src_axes[i].row_splits);
        CheckArrayData(dest_axes[i].row_ids, src_axes[i].row_ids);
      }
    }
  }
}

TEST_F(RaggedShapeOpsSuiteTest, TestRemoveAxisCpu) {
  TestRemoveAxis(GetCpuContext(), simple_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestRemoveAxisGpu) {
  TestRemoveAxis(GetCudaContext(), simple_shape_);
}

void TestGetOffsets(ContextPtr context) {
  for (int32_t i = 0; i != 2; ++i) {
    int32_t num_shape = RandInt(10, 100);
    int32_t num_axes = RandInt(2, 4);
    std::vector<RaggedShape> shape_vec(num_shape);
    std::vector<RaggedShape *> shapes(num_shape);
    for (int32_t j = 0; j != num_shape; ++j) {
      shape_vec[j] =
          RandomRaggedShape(false, num_axes, num_axes, 0, 1000).To(context);
      shapes[j] = &shape_vec[j];
    }
    RaggedShape **shapes_ptr = shapes.data();
    Array2<int32_t> offsets = GetOffsets(num_shape, shapes_ptr);
    ASSERT_EQ(offsets.Dim0(), num_axes + 1);
    ASSERT_EQ(offsets.Dim1(), num_shape + 1);
    auto acc = offsets.Accessor();
    for (int32_t axis = 0; axis <= num_axes; ++axis) {
      int32_t sum = 0;
      for (int32_t j = 0; j <= num_shape; ++j) {
        EXPECT_EQ(acc(axis, j), sum);
        if (j < num_shape) {
          sum += (axis == 0 ? 1 : shape_vec[j].TotSize(axis - 1));
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, TestGetOffsets) {
  TestGetOffsets(GetCpuContext());
  TestGetOffsets(GetCudaContext());
}

// returns a random ragged shape where the dims on axis 1 are all the same
// (so: can be transposed).
RaggedShape RandomRaggedShapeToTranspose(ContextPtr c) {
  ContextPtr c_cpu = GetCpuContext();

  RaggedShape random = RandomRaggedShape(false, 2, 4, 0, 5000).To(c);

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

  {
    const std::vector<int32_t> row_splits1_vec = {0, 2, 4, 6};
    const std::vector<int32_t> row_splits2_vec = {0, 3, 4, 7, 8, 10, 12};
    Array1<int32_t> row_splits1(context, row_splits1_vec);
    Array1<int32_t> row_splits2(context, row_splits2_vec);
    RaggedShape src_shape =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
    ASSERT_EQ(src_shape.Dim0(), 3);
    ASSERT_EQ(src_shape.TotSize(1), 6);
    RaggedShape shape = Transpose(src_shape);
    EXPECT_EQ(shape.Dim0(), 2);
    ASSERT_EQ(shape.TotSize(1), 6);
    const std::vector<int32_t> expected_row_splits = {0, 3, 6};
    const std::vector<int32_t> expected_row_ids = {0, 0, 0, 1, 1, 1};
    CheckArrayData(shape.RowSplits(1), expected_row_splits);
    CheckArrayData(shape.RowIds(1), expected_row_ids);
    CheckArrayData(shape.RowSplits(2), {0, 3, 6, 8, 9, 10, 12});
    CheckArrayData(shape.RowIds(2), {0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 5});
  }

  {
    // random case
    for (int32_t j = 0; j != 2; ++j) {
      RaggedShape to_transpose = RandomRaggedShapeToTranspose(context);
      RaggedShape transposed = Transpose(to_transpose);

      if (d != kCpu) {
        to_transpose = to_transpose.To(cpu);
        transposed = transposed.To(cpu);
      }

      for (auto iter = transposed.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        int32_t i = transposed[index];  // Just make sure this doesn't crash,
                                        // dont need the value.
        std::swap(index[0], index[1]);
        i = to_transpose[index];  // don't need the value, just need to make
                                  // sure it's an allowable index.
      }
      for (auto iter = to_transpose.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        std::swap(index[0], index[1]);
        int32_t i = transposed[index];  // don't need the value, just need to
                                        // make sure it's an allowable index.
      }
    }
  }
}
TEST(RaggedShapeOpsTest, TestTranspose) {
  TestTranspose<kCpu>();
  TestTranspose<kCuda>();
}

template <DeviceType d, typename T>
void TestTransposeRagged() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    const std::vector<int32_t> row_splits1_vec = {0, 2, 4, 6};
    const std::vector<int32_t> row_splits2_vec = {0, 3, 4, 7, 8, 10, 12};
    Array1<int32_t> row_splits1(context, row_splits1_vec);
    Array1<int32_t> row_splits2(context, row_splits2_vec);
    RaggedShape src_shape =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
    ASSERT_EQ(src_shape.Dim0(), 3);
    ASSERT_EQ(src_shape.TotSize(1), 6);
    std::vector<T> values = {0, 1, 2, 3, 4, 5, 8, 7, 6, 9, 10, 15};
    ASSERT_EQ(values.size(), src_shape.NumElements());
    Array1<T> values_array(context, values);
    Ragged<T> ragged(src_shape, values_array);
    Ragged<T> ans = Transpose(ragged);
    RaggedShape shape = ans.shape;
    // Check shape
    ASSERT_EQ(shape.Dim0(), 2);
    ASSERT_EQ(shape.TotSize(1), 6);
    const std::vector<int32_t> expected_row_splits = {0, 3, 6};
    const std::vector<int32_t> expected_row_ids = {0, 0, 0, 1, 1, 1};
    CheckArrayData(shape.RowSplits(1), expected_row_splits);
    CheckArrayData(shape.RowIds(1), expected_row_ids);
    CheckArrayData(shape.RowSplits(2), {0, 3, 6, 8, 9, 10, 12});
    CheckArrayData(shape.RowIds(2), {0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 5});
    // Check values
    CheckArrayData(ans.values, {0, 1, 2, 4, 5, 8, 6, 9, 3, 7, 10, 15});
  }

  {
    // random case
    for (int32_t j = 0; j != 2; ++j) {
      RaggedShape to_transpose = RandomRaggedShapeToTranspose(context);
      int32_t num_elems = to_transpose.NumElements();
      Array1<T> src_values = RandUniformArray1<T>(context, num_elems, 0, 10000);
      Ragged<T> src(to_transpose, src_values);
      Ragged<T> ans = Transpose(src);
      if (d != kCpu) {
        src = src.To(cpu);
        ans = ans.To(cpu);
        to_transpose = to_transpose.To(cpu);
      }
      RaggedShape transposed = ans.shape;

      for (auto iter = transposed.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        T value = ans[index];
        std::swap(index[0], index[1]);
        EXPECT_EQ(value, src[index]);
      }
      for (auto iter = to_transpose.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        T value = src[index];
        std::swap(index[0], index[1]);
        EXPECT_EQ(value, ans[index]);
      }
    }
  }
}
TEST(RaggedTest, TestTransposeRagged) {
  TestTransposeRagged<kCpu, int32_t>();
  TestTransposeRagged<kCuda, int32_t>();
  TestTransposeRagged<kCpu, double>();
  TestTransposeRagged<kCuda, double>();
}

template <DeviceType d>
void TestRowSplitsPtr() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }
  RaggedShape shape = RandomRaggedShape().To(context);
  ASSERT_GE(shape.NumAxes(), 2);
  Array1<int32_t *> ptrs = GetRowSplitsPtr(shape);
  ASSERT_EQ(ptrs.Dim(), shape.NumAxes() - 1);
  // as num_axes is not so big, access (may copy memory) it in a loop is fine.
  for (int32_t i = 0; i != ptrs.Dim(); ++i) {
    EXPECT_EQ(ptrs[i], shape.RowSplits(i + 1).Data());
  }
}
TEST(RaggedShapeOpsTest, TestRowSplitsPtr) {
  TestRowSplitsPtr<kCpu>();
  TestRowSplitsPtr<kCuda>();
}

void TestRaggedShape2(ContextPtr context, const RaggedShape &shape) {
  RaggedShape src_shape = shape.To(context);
  src_shape.Populate();
  ASSERT_GE(src_shape.NumAxes(), 2);
  Array1<int32_t> row_splits = src_shape.RowSplits(1);
  Array1<int32_t> row_ids = src_shape.RowIds(1);
  int32_t cached_tot_size = src_shape.TotSize(1);

  {
    // both row_splits and row_ids are non-null
    RaggedShape result = RaggedShape2(&row_splits, &row_ids, cached_tot_size);
    CheckArrayData(result.RowSplits(1), row_splits);
    CheckArrayData(result.RowIds(1), row_ids);
    EXPECT_EQ(result.TotSize(1), cached_tot_size);
  }
  {
    // both row_splits and row_ids are non-null, cached_tot_size = -1
    RaggedShape result = RaggedShape2(&row_splits, &row_ids, -1);
    CheckArrayData(result.RowSplits(1), row_splits);
    CheckArrayData(result.RowIds(1), row_ids);
    EXPECT_EQ(result.TotSize(1), cached_tot_size);
  }
  {
    // row_ids is null
    RaggedShape result = RaggedShape2(&row_splits, nullptr, cached_tot_size);
    CheckArrayData(result.RowSplits(1), row_splits);
    CheckArrayData(result.RowIds(1), row_ids);
    EXPECT_EQ(result.TotSize(1), cached_tot_size);
  }
  {
    // row_ids is null, cached_tot_size = -1
    RaggedShape result = RaggedShape2(&row_splits, nullptr, -1);
    CheckArrayData(result.RowSplits(1), row_splits);
    CheckArrayData(result.RowIds(1), row_ids);
    EXPECT_EQ(result.TotSize(1), cached_tot_size);
  }

  // note if row_splits == null, then we suppose there's no empty rows after the
  // last row-id in row_ids
  if (row_splits.Dim() == (row_ids.Dim() == 0 ? 1 : row_ids.Back() + 2)) {
    {
      // row_splits is null
      RaggedShape result = RaggedShape2(nullptr, &row_ids, cached_tot_size);
      CheckArrayData(result.RowSplits(1), row_splits);
      CheckArrayData(result.RowIds(1), row_ids);
      EXPECT_EQ(result.TotSize(1), cached_tot_size);
    }
    {
      // row_splits is null, cached_tot_size = -1
      RaggedShape result = RaggedShape2(nullptr, &row_ids, -1);
      CheckArrayData(result.RowSplits(1), row_splits);
      CheckArrayData(result.RowIds(1), row_ids);
      EXPECT_EQ(result.TotSize(1), cached_tot_size);
    }
  }
}
TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape2Cpu) {
  TestRaggedShape2(GetCpuContext(), simple_shape_);
  TestRaggedShape2(GetCpuContext(), random_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape2Gpu) {
  TestRaggedShape2(GetCudaContext(), simple_shape_);
  TestRaggedShape2(GetCudaContext(), random_shape_);
}

void TestRaggedShape3(ContextPtr context, const RaggedShape &shape) {
  RaggedShape src_shape = shape.To(context);
  src_shape.Populate();
  ASSERT_GE(src_shape.NumAxes(), 3);
  Array1<int32_t> row_splits1 = src_shape.RowSplits(1);
  Array1<int32_t> row_ids1 = src_shape.RowIds(1);
  int32_t cached_tot_size1 = src_shape.TotSize(1);
  Array1<int32_t> row_splits2 = src_shape.RowSplits(2);
  Array1<int32_t> row_ids2 = src_shape.RowIds(2);
  int32_t cached_tot_size2 = src_shape.TotSize(2);

  {
    // both row_splits and row_ids are non-null
    RaggedShape result =
        RaggedShape3(&row_splits1, &row_ids1, cached_tot_size1, &row_splits2,
                     &row_ids2, cached_tot_size2);
    CheckArrayData(result.RowSplits(1), row_splits1);
    CheckArrayData(result.RowIds(1), row_ids1);
    EXPECT_EQ(result.TotSize(1), cached_tot_size1);
    CheckArrayData(result.RowSplits(2), row_splits2);
    CheckArrayData(result.RowIds(2), row_ids2);
    EXPECT_EQ(result.TotSize(2), cached_tot_size2);
  }
  {
    // row_ids is non-null, cached_tot_size = -1
    RaggedShape result =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
    CheckArrayData(result.RowSplits(1), row_splits1);
    CheckArrayData(result.RowIds(1), row_ids1);
    EXPECT_EQ(result.TotSize(1), cached_tot_size1);
    CheckArrayData(result.RowSplits(2), row_splits2);
    CheckArrayData(result.RowIds(2), row_ids2);
    EXPECT_EQ(result.TotSize(2), cached_tot_size2);
  }

  // note if row_splits == null, then we suppose there's no empty rows after the
  // last row-id in row_ids
  bool valid1 =
      (row_splits1.Dim() == (row_ids1.Dim() == 0 ? 1 : row_ids1.Back() + 2));
  bool valid2 =
      (row_splits2.Dim() == (row_ids2.Dim() == 0 ? 1 : row_ids2.Back() + 2));
  if (valid1 && valid2) {
    RaggedShape result =
        RaggedShape3(nullptr, &row_ids1, -1, nullptr, &row_ids2, -1);
    CheckArrayData(result.RowSplits(1), row_splits1);
    CheckArrayData(result.RowIds(1), row_ids1);
    EXPECT_EQ(result.TotSize(1), cached_tot_size1);
    CheckArrayData(result.RowSplits(2), row_splits2);
    CheckArrayData(result.RowIds(2), row_ids2);
    EXPECT_EQ(result.TotSize(2), cached_tot_size2);
  }
  // TODO(haowen): add more cases for other branches
}
TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape3Cpu) {
  TestRaggedShape3(GetCpuContext(), simple_shape_);
  TestRaggedShape3(GetCpuContext(), random_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape3Gpu) {
  TestRaggedShape3(GetCudaContext(), simple_shape_);
  TestRaggedShape3(GetCudaContext(), random_shape_);
}

void TestComposeShape(ContextPtr context, const RaggedShape &shape) {
  RaggedShape src_shape = shape.To(context);
  ASSERT_GE(src_shape.NumAxes(), 3);
  Array1<int32_t> row_splits1 = src_shape.RowSplits(1);
  Array1<int32_t> row_ids1 = src_shape.RowIds(1);
  Array1<int32_t> row_splits2 = src_shape.RowSplits(2);
  Array1<int32_t> row_ids2 = src_shape.RowIds(2);

  RaggedShape shape1 = RaggedShape2(&row_splits1, nullptr, -1);
  RaggedShape shape2 = RaggedShape2(&row_splits2, nullptr, -1);

  RaggedShape result = ComposeRaggedShapes(shape1, shape2);

  ASSERT_EQ(result.NumAxes(), 3);

  CheckArrayData(result.RowSplits(1), row_splits1);
  CheckArrayData(result.RowIds(1), row_ids1);
  CheckArrayData(result.RowSplits(2), row_splits2);
  CheckArrayData(result.RowIds(2), row_ids2);
}
TEST_F(RaggedShapeOpsSuiteTest, TestComposeShapeCpu) {
  TestComposeShape(GetCpuContext(), simple_shape_);
  TestComposeShape(GetCpuContext(), random_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestComposeShapeGpu) {
  TestComposeShape(GetCudaContext(), simple_shape_);
  TestComposeShape(GetCudaContext(), random_shape_);
}

void TestShapeFromTotSize(ContextPtr context, const RaggedShape &shape) {
  RaggedShape src_shape = shape.To(context);
  ASSERT_GE(src_shape.NumAxes(), 2);

  int32_t num_axes = src_shape.NumAxes();
  std::vector<int32_t> tot_sizes(num_axes);
  for (int32_t i = 0; i != num_axes; ++i) {
    tot_sizes[i] = src_shape.TotSize(i);
  }

  RaggedShape result =
      RaggedShapeFromTotSizes(context, num_axes, tot_sizes.data());

  ASSERT_EQ(result.NumAxes(), num_axes);
  for (int32_t i = 0; i < num_axes; ++i) {
    EXPECT_EQ(result.TotSize(i), src_shape.TotSize(i));
    if (i > 0) {
      EXPECT_EQ(result.RowSplits(i).Dim(), src_shape.RowSplits(i).Dim());
      EXPECT_EQ(result.RowIds(i).Dim(), src_shape.RowIds(i).Dim());
    }
  }
}
TEST_F(RaggedShapeOpsSuiteTest, TestShapeFromTotSizeCpu) {
  TestShapeFromTotSize(GetCpuContext(), simple_shape_);
  TestShapeFromTotSize(GetCpuContext(), random_shape_);
}
TEST_F(RaggedShapeOpsSuiteTest, TestShapeFromTotSizeGpu) {
  TestShapeFromTotSize(GetCudaContext(), simple_shape_);
  TestShapeFromTotSize(GetCudaContext(), random_shape_);
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

  Array1<T> unsorted = values.Clone();

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

template <DeviceType d>
void TestAppend() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // simple case
    std::vector<RaggedShape> shapes(2);
    std::vector<RaggedShape *> shapes_ptr(2);
    std::vector<std::vector<Array1<int32_t>>> row_splits_vec(2);
    {
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
      const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
      const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
      Array1<int32_t> splits1(context, row_splits1);
      Array1<int32_t> ids1(context, row_ids1);
      Array1<int32_t> splits2(context, row_splits2);
      Array1<int32_t> ids2(context, row_ids2);
      row_splits_vec[0].push_back(splits1);
      row_splits_vec[1].push_back(splits2);
      shapes[0] = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2, &ids2,
                               ids2.Dim());
      shapes_ptr[0] = &shapes[0];
    }
    {
      const std::vector<int32_t> row_splits1 = {0, 1, 3, 4};
      const std::vector<int32_t> row_ids1 = {0, 1, 1, 2};
      const std::vector<int32_t> row_splits2 = {0, 3, 4, 5, 7};
      const std::vector<int32_t> row_ids2 = {0, 0, 0, 1, 2, 3, 3};
      Array1<int32_t> splits1(context, row_splits1);
      Array1<int32_t> ids1(context, row_ids1);
      Array1<int32_t> splits2(context, row_splits2);
      Array1<int32_t> ids2(context, row_ids2);
      row_splits_vec[0].push_back(splits1);
      row_splits_vec[1].push_back(splits2);
      RaggedShape shape = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2,
                                       &ids2, ids2.Dim());
      shapes[1] = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2, &ids2,
                               ids2.Dim());
      shapes_ptr[1] = &shapes[1];
    }

    {
      // axis == 1
      RaggedShape result = Append(1, 2, shapes_ptr.data());
      std::vector<std::vector<int32_t>> expected_row_splits = {
          {0, 3, 8, 10}, {0, 2, 3, 6, 7, 9, 10, 11, 12, 15, 17}};
      std::vector<std::vector<int32_t>> expected_row_ids = {
          {0, 0, 0, 1, 1, 1, 1, 1, 2, 2},
          {0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8, 9, 9}};
      for (int32_t i = 0; i < 2; ++i) {
        CheckArrayData(result.RowSplits(i + 1), expected_row_splits[i]);
        CheckArrayData(result.RowIds(i + 1), expected_row_ids[i]);
      }
    }

    {
      // axis == 0
      RaggedShape result = Append(0, 2, shapes_ptr.data());

      // get result splits with `SpliceRowSplits` and get result row-ids with
      // `RowSplitsToRowIds``
      std::vector<Array1<int32_t>> result_splits;
      std::vector<Array1<int32_t>> result_ids;
      for (auto i = 0; i < 2; ++i) {
        std::vector<const Array1<int32_t> *> splits_ptr = {
            &row_splits_vec[i][0], &row_splits_vec[i][1]};
        Array1<int32_t> curr_row_splits = SpliceRowSplits(2, splits_ptr.data());
        result_splits.push_back(curr_row_splits);
        Array1<int32_t> curr_row_ids(context, curr_row_splits.Back());
        RowSplitsToRowIds(curr_row_splits, &curr_row_ids);
        result_ids.push_back(curr_row_ids);
      }
      for (int32_t i = 0; i < 2; ++i) {
        CheckArrayData(result.RowSplits(i + 1), result_splits[i]);
        CheckArrayData(result.RowIds(i + 1), result_ids[i]);
      }
    }
  }

  {
    // test with random large size
    for (int32_t i = 0; i < 2; ++i) {
      int32_t num_shape = RandInt(2, 100);
      int32_t num_axes = RandInt(2, 4);
      std::vector<RaggedShape> shape_vec(num_shape);
      std::vector<RaggedShape *> shapes(num_shape);
      for (int32_t j = 0; j != num_shape; ++j) {
        shape_vec[j] =
            RandomRaggedShape(true, num_axes, num_axes, 0, 1000).To(context);
        shapes[j] = &shape_vec[j];
      }
      // only test case axis == 0, test axis==1 with simple case is good enough
      // as it just calls Stack
      RaggedShape result = Append(0, num_shape, shapes.data());
      ASSERT_EQ(result.NumAxes(), num_axes);

      // get result splits with `SpliceRowSplits` and get result row-ids with
      // `RowSplitsToRowIds``
      std::vector<Array1<int32_t>> result_splits;
      std::vector<Array1<int32_t>> result_ids;
      for (int32_t axis = 1; axis < num_axes; ++axis) {
        std::vector<Array1<int32_t>> splits_vec(num_shape);
        std::vector<const Array1<int32_t> *> splits_vec_ptr(num_shape);
        for (int32_t n = 0; n != num_shape; ++n) {
          splits_vec[n] = shape_vec[n].RowSplits(axis);
          splits_vec_ptr[n] = &splits_vec[n];
        }
        Array1<int32_t> curr_row_splits =
            SpliceRowSplits(num_shape, splits_vec_ptr.data());
        result_splits.push_back(curr_row_splits);
        Array1<int32_t> curr_row_ids(context, curr_row_splits.Back());
        RowSplitsToRowIds(curr_row_splits, &curr_row_ids);
        result_ids.push_back(curr_row_ids);
      }

      // check data
      for (int32_t axis = 1; axis < num_axes; ++axis) {
        CheckArrayData(result.RowSplits(axis), result_splits[axis - 1]);
        CheckArrayData(result.RowIds(axis), result_ids[axis - 1]);
      }
    }
  }
}
TEST(RaggedShapeOpsTest, TestAppend) {
  TestAppend<kCpu>();
  TestAppend<kCuda>();
}

template <DeviceType d, typename T>
void TestAppendRagged() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  // TODO(haowen): remove duplicate code in TestAppend above.
  // test with simple case could be good enough, as we have tested
  // Append(RaggedShape&) already.
  std::vector<Ragged<T>> ragged_vec(2);
  std::vector<Ragged<T> *> ragged(2);
  std::vector<std::vector<Array1<int32_t>>> row_splits_vec(2);
  {
    const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
    const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
    const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
    const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
    const std::vector<T> values_vec = {1, 2, 5, 7, 9, 10, 12, 14, 15, 18};
    Array1<int32_t> splits1(context, row_splits1);
    Array1<int32_t> ids1(context, row_ids1);
    Array1<int32_t> splits2(context, row_splits2);
    Array1<int32_t> ids2(context, row_ids2);
    RaggedShape shape =
        RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2, &ids2, ids2.Dim());
    Array1<T> values(context, values_vec);
    ragged_vec[0] = Ragged<T>(shape, values);
    ragged[0] = &ragged_vec[0];
  }

  {
    const std::vector<int32_t> row_splits1 = {0, 1, 3, 4};
    const std::vector<int32_t> row_ids1 = {0, 1, 1, 2};
    const std::vector<int32_t> row_splits2 = {0, 3, 4, 5, 7};
    const std::vector<int32_t> row_ids2 = {0, 0, 0, 1, 2, 3, 3};
    const std::vector<T> values_vec = {20, 21, 23, 28, 30, 32, 35};
    Array1<int32_t> splits1(context, row_splits1);
    Array1<int32_t> ids1(context, row_ids1);
    Array1<int32_t> splits2(context, row_splits2);
    Array1<int32_t> ids2(context, row_ids2);
    RaggedShape shape =
        RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2, &ids2, ids2.Dim());
    Array1<T> values(context, values_vec);
    ragged_vec[1] = Ragged<T>(shape, values);
    ragged[1] = &ragged_vec[1];
  }

  {
    // axis == 0
    Ragged<T> result = Append(0, 2, ragged.data());
    std::vector<std::vector<int32_t>> expected_row_splits = {
        {0, 2, 5, 6, 7, 9, 10}, {0, 2, 3, 4, 6, 7, 10, 13, 14, 15, 17}};
    std::vector<std::vector<int32_t>> expected_row_ids = {
        {0, 0, 1, 1, 1, 2, 3, 4, 4, 5},
        {0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9, 9}};
    for (int32_t i = 0; i < 2; ++i) {
      CheckArrayData(result.RowSplits(i + 1), expected_row_splits[i]);
      CheckArrayData(result.RowIds(i + 1), expected_row_ids[i]);
    }
    std::vector<T> expected_data = {1,  2,  5,  7,  9,  10, 12, 14, 15,
                                    18, 20, 21, 23, 28, 30, 32, 35};
    CheckArrayData(result.values, expected_data);
  }

  {
    // axis == 1
    Ragged<T> result = Append(1, 2, ragged.data());
    std::vector<std::vector<int32_t>> expected_row_splits = {
        {0, 3, 8, 10}, {0, 2, 3, 6, 7, 9, 10, 11, 12, 15, 17}};
    std::vector<std::vector<int32_t>> expected_row_ids = {
        {0, 0, 0, 1, 1, 1, 1, 1, 2, 2},
        {0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8, 9, 9}};
    for (int32_t i = 0; i < 2; ++i) {
      CheckArrayData(result.RowSplits(i + 1), expected_row_splits[i]);
      CheckArrayData(result.RowIds(i + 1), expected_row_ids[i]);
    }
    std::vector<T> expected_data = {1,  2,  5,  20, 21, 23, 7,  9, 10,
                                    12, 28, 30, 14, 15, 18, 32, 35};
    CheckArrayData(result.values, expected_data);
  }
}
TEST(RaggedTest, TestAppendRagged) {
  TestAppendRagged<kCpu, int32_t>();
  TestAppendRagged<kCuda, int32_t>();
  TestAppendRagged<kCpu, double>();
  TestAppendRagged<kCuda, double>();
}

void CheckResultOfIndex(const ContextPtr &context, RaggedShape shape,
                           Array1<int32_t> new2old, RaggedShape result) {
  K2_CHECK(context->IsCompatible(*shape.Context()));
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  int32_t num_axes = shape.NumAxes();
  int32_t src_dim0 = shape.Dim0(),
      result_dim0 = result.Dim0();
  if (result_dim0 == 0) {
    std::vector<int32_t> empty_row_splits = {0};
    for (int32_t i = 0; i < num_axes - 1; ++i) {
      CheckArrayData(result.RowSplits(i + 1), empty_row_splits);
      EXPECT_EQ(result.RowIds(i + 1).Dim(), 0);
    }
    return;
  }
  Array2<int32_t> old_offsets(context, num_axes, src_dim0 + 1);
  auto old_offsets_acc = old_offsets.Accessor();
  Array1<int32_t *> row_splits_ptrs = GetRowSplitsPtr(shape);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();
  // Set old_offsets
  auto lambda_get_old_offsets = [=] __host__ __device__(int32_t i) {
    // 0 <= i <= dim0
    int32_t cur_offset = i;
    for (int32_t axis = 0; axis < num_axes; axis++) {
      old_offsets_acc(axis, i) = cur_offset;
      if (axis + 1 == num_axes) return;
      cur_offset = row_splits_ptrs_data[axis][cur_offset];
    }
  };
  Eval(context, src_dim0 + 1, lambda_get_old_offsets);
  old_offsets = old_offsets.To(cpu);
  auto cpu_offsets_acc = old_offsets.Accessor();
  shape = shape.To(cpu);
  new2old = new2old.To(cpu);
  // get result splits with `SpliceRowSplits` and get result row-ids with
  // `RowSplitsToRowIds``
  std::vector<Array1<int32_t>> result_splits;
  std::vector<Array1<int32_t>> result_ids;
  for (auto axis = 0; axis < num_axes - 1; ++axis) {
    Array1<int32_t> curr_row_splits = shape.RowSplits(axis + 1);
    std::vector<Array1<int32_t>> splits_vec(result_dim0);
    std::vector<const Array1<int32_t> *> splits_vec_ptr(result_dim0);
    for (int32_t m = 0; m != result_dim0; ++m) {
      int32_t old_idx = new2old[m];
      int32_t start = cpu_offsets_acc(axis, old_idx);
      int32_t end = cpu_offsets_acc(axis, old_idx + 1);
      Array1<int32_t> sub_list = curr_row_splits.Range(start, end - start + 1);
      Array1<int32_t> copy_sub_list(cpu, sub_list.Dim());
      copy_sub_list.CopyFrom(sub_list);
      int32_t *data = copy_sub_list.Data();
      int32_t init = data[0];
      for (int32_t n = 0; n != copy_sub_list.Dim(); ++n) {
        data[n] -= init;
      }
      splits_vec[m] = copy_sub_list;
      splits_vec_ptr[m] = &splits_vec[m];
    }
    Array1<int32_t> result_row_splits =
        SpliceRowSplits(result_dim0, splits_vec_ptr.data());
    result_splits.push_back(result_row_splits);
    Array1<int32_t> result_row_ids(cpu, result_row_splits.Back());
    RowSplitsToRowIds(result_row_splits, &result_row_ids);
    result_ids.push_back(result_row_ids);
  }
  for (int32_t i = 0; i < num_axes - 1; ++i) {
    CheckArrayData(result.RowSplits(i + 1), result_splits[i]);
    CheckArrayData(result.RowIds(i + 1), result_ids[i]);
  }
}


void TestIndex(DeviceType d) {
  for (int i = 0; i < 5; i++) {
    ContextPtr cpu = GetCpuContext();  // will use to copy data
    ContextPtr context = nullptr;
    if (d == kCpu) {
      context = GetCpuContext();
    } else {
      K2_CHECK_EQ(d, kCuda);
      context = GetCudaContext();
    }

    {
      // simple case
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
      const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
      const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};


      Array1<int32_t> splits1(context, row_splits1);
      Array1<int32_t> ids1(context, row_ids1);
      Array1<int32_t> splits2(context, row_splits2);
      Array1<int32_t> ids2(context, row_ids2);
      RaggedShape shape = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2,
                                       &ids2, ids2.Dim());

      std::vector<int32_t> new2old_vec = {2, 1};
      Array1<int32_t> new2old(context, new2old_vec);
      Array1<int32_t> value_indexes_out;
      RaggedShape result  = Index(shape, new2old, &value_indexes_out);
      // fsa 2, state_idx01 {5}, arc_idx012 {7, 8, 9}
      // fsa 1, state_idx01 {2, 3, 4}, arc_idx012 {{3},{4, 5}, {6}}
      CheckArrayData(value_indexes_out,
                     std::vector<int32_t>{7, 8, 9, 3, 4, 5, 6});
      CheckResultOfIndex(context, shape, new2old, result);
    }

    {
      // test with random large size
      for (int32_t i = 0; i < 2; ++i) {
        int32_t num_axes = RandInt(2, 4);
        RaggedShape shape =
            RandomRaggedShape(true, num_axes, num_axes, 0, 1000).To(context);
        int32_t dim0 = shape.Dim0(),
            result_dim0 = RandInt(0, 10);
        if (dim0 == 0)
          result_dim0 = 0;
        std::vector<int32_t> new2old_vec(result_dim0);
        for (int i = 0; i < result_dim0; i++)
          new2old_vec[i] = RandInt(0, dim0 - 1);
        Array1<int32_t> new2old(context, new2old_vec);
        Array1<int32_t> value_indexes;
        RaggedShape result = Index(shape, new2old, &value_indexes);
        CheckResultOfIndex(context, shape, new2old, result);
        K2_LOG(INFO) << "Value_indexes = " << value_indexes;
      }
    }
  }
}


TEST(RaggedShapeOpsTest, TestIndex) {
  TestIndex(kCpu);
  TestIndex(kCuda);
}


TEST(GetTransposeReordering, NoDuplicates) {
  //       col0  col1  col2  col3  col4  col5
  // row0                           a0    b1
  // row1   c2    d3                      e4
  // row2                     f5
  // row3   g6          h7          i8
  // row4                                 j9
  // row5         k10               l11
  std::vector<int32_t> col_indexes{4, 5, 0, 1, 5, 3, 0, 2, 4, 5, 1, 4};
  std::vector<int32_t> _row_splits{0, 2, 5, 6, 9, 10, 12};
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> row_splits(context, _row_splits);
    RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
    Array1<int32_t> values(context, col_indexes);

    Ragged<int32_t> ragged(shape, values);
    Array1<int32_t> order = GetTransposeReordering(ragged, 6);
    CheckArrayData(order, {2, 6, 3, 10, 7, 5, 0, 8, 11, 1, 4, 9});
    EXPECT_TRUE(context->IsCompatible(*order.Context()));
  }
}

TEST(GetTransposeReordering, NoDuplicatesThreeAxes) {
  //       col0  col1  col2  col3  col4  col5
  // row0         a0          b1
  // row1   c2          d3
  // row2         e4
  // row3   f5    g6          h7
  // row4                                  i8
  // row5                            j9    k10
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> col_indexes(
        context, std::vector<int32_t>{1, 3, 0, 2, 1, 0, 1, 3, 5, 4, 5});
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 4, 6});
    Array1<int32_t> row_splits2(context,
                                std::vector<int32_t>{0, 2, 4, 5, 8, 9, 11});
    RaggedShape shape =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
    Ragged<int32_t> ragged(shape, col_indexes);
    Array1<int32_t> order = GetTransposeReordering(ragged, 6);
    CheckArrayData(order, {2, 5, 0, 4, 6, 3, 1, 7, 9, 8, 10});
    EXPECT_TRUE(context->IsCompatible(*order.Context()));
  }
}

TEST(GetTransposeReordering, WithDuplicates) {
  //       col0   col1   col2    col3      col4      col5
  // row0         a0,a1         b2,b3,b4
  // row1  c5,c6          d7
  // row2         e8
  // row3   f9   g10,g11         h12
  // row4                                i13,i14,i15
  // row5                        j16                  k17
  std::vector<int32_t> col_indexes{1, 1, 3, 3, 3, 0, 0, 2, 1,
                                   0, 1, 1, 3, 4, 4, 4, 3, 5};
  std::vector<int32_t> _row_splits{0, 5, 8, 9, 13, 16, 18};
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> row_splits(context, _row_splits);
    RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
    Array1<int32_t> values(context, col_indexes);
    Ragged<int32_t> ragged(shape, values);
    Array1<int32_t> order = GetTransposeReordering(ragged, 6);
    CheckArrayData(
        order, {5, 6, 9, 0, 1, 8, 10, 11, 7, 2, 3, 4, 12, 16, 13, 14, 15, 17});
    EXPECT_TRUE(context->IsCompatible(*order.Context()));
  }
}

TEST(GetTransposeReordering, WithDuplicatesThreeAxes) {
  //       col0   col1   col2    col3      col4      col5
  // row0         a0,a1         b2,b3,b4
  // row1  c5,c6          d7
  // row2         e8
  // row3   f9   g10,g11         h12
  // row4                                i13,i14,i15
  // row5                                 j16         k17
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> col_indexes(
        context, std::vector<int32_t>{1, 1, 3, 3, 3, 0, 0, 2, 1, 0, 1, 1, 3, 4,
                                      4, 4, 4, 5});
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 4, 6});
    Array1<int32_t> row_splits2(context,
                                std::vector<int32_t>{0, 5, 8, 9, 13, 16, 18});
    RaggedShape shape =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
    Ragged<int32_t> ragged(shape, col_indexes);
    Array1<int32_t> order = GetTransposeReordering(ragged, 6);
    CheckArrayData(
        order, {5, 6, 9, 0, 1, 8, 10, 11, 7, 2, 3, 4, 12, 13, 14, 15, 16, 17});
    EXPECT_TRUE(context->IsCompatible(*order.Context()));
  }
}

TEST(ChangeSublistSize, TwoAxes) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 2, 5});
    RaggedShape src = RaggedShape2(&row_splits1, nullptr, -1);

    int32_t size_delta = 2;
    RaggedShape dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 4, 9});

    size_delta = -2;
    dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 0, 1});

    size_delta = 0;
    dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 2, 5});
  }
}

TEST(ChangeSublistSize, ThreeAxes) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    /*
     [
       [ [x, x, x], [x, x] ]
       [ [x], [x, x], [x, x, x] ]
     ]
     */
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 2, 5});
    Array1<int32_t> row_splits2(context,
                                std::vector<int32_t>{0, 3, 5, 6, 8, 11});
    RaggedShape src =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);

    int32_t size_delta = 2;
    RaggedShape dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(2), std::vector<int32_t>{0, 5, 9, 12, 16, 21});

    // it is an error to use -2 here
    // because the state (state_idx01 == 2) has only 1 entry
    size_delta = -1;

    dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(2), std::vector<int32_t>{0, 2, 3, 3, 4, 6});

    size_delta = 0;
    dst = ChangeSublistSize(src, size_delta);
    CheckArrayData(dst.RowSplits(2), std::vector<int32_t>{0, 3, 5, 6, 8, 11});
  }
}

template <DeviceType d>
void TestGetCountsPartitioned() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  // Testing with simple case is good enough as we have tested GetCounts()
  // with random large size and GetCountsPartitioned just calls GetCounts.
  std::vector<int32_t> src_row_splits_vec = {0, 3, 4, 6, 10};
  Array1<int32_t> src_row_splits(context, src_row_splits_vec);
  RaggedShape src_shape = RaggedShape2(&src_row_splits, nullptr, -1);
  std::vector<int32_t> src_values_vec = {0, 1, 0, 2, 5, 5, 7, 7, 9, 7};
  Array1<int32_t> src_values(context, src_values_vec);
  Ragged<int32_t> src(src_shape, src_values);

  std::vector<int32_t> ans_row_splits_vec = {0, 2, 4, 7, 10};
  Array1<int32_t> ans_row_splits(context, ans_row_splits_vec);
  RaggedShape ans_shape = RaggedShape2(&ans_row_splits, nullptr, -1);

  Ragged<int32_t> result = GetCountsPartitioned(src, ans_shape);

  ASSERT_EQ(result.NumAxes(), 2);
  // Check row_splits
  Array1<int32_t> row_splits = result.shape.RowSplits(1).To(cpu);
  std::vector<int32_t> result_row_splits(row_splits.Data(),
                                         row_splits.Data() + row_splits.Dim());
  EXPECT_EQ(result_row_splits, ans_row_splits_vec);
  // check values
  std::vector<int32_t> expected_data = {2, 1, 1, 0, 0, 2, 0, 3, 0, 1};
  Array1<int32_t> values = result.values.To(cpu);
  std::vector<int32_t> data(values.Data(), values.Data() + values.Dim());
  EXPECT_EQ(data, expected_data);
}

TEST(RaggedShapeOpsTest, TestGetCountsPartitioned) {
  TestGetCountsPartitioned<kCpu>();
  TestGetCountsPartitioned<kCuda>();
}

template <DeviceType d>
void TestStack() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // simple case
    std::vector<RaggedShape> shapes(2);
    std::vector<RaggedShape *> shapes_ptr(2);
    std::vector<std::vector<Array1<int32_t>>> row_splits_vec(2);
    {
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
      Array1<int32_t> splits1(context, row_splits1);
      Array1<int32_t> splits2(context, row_splits2);
      row_splits_vec[0].push_back(splits1);
      row_splits_vec[1].push_back(splits2);
      shapes[0] = RaggedShape3(&splits1, nullptr, -1, &splits2, nullptr, -1);
      shapes_ptr[0] = &shapes[0];
    }
    {
      const std::vector<int32_t> row_splits1 = {0, 1, 3, 4};
      const std::vector<int32_t> row_splits2 = {0, 3, 4, 5, 7};
      Array1<int32_t> splits1(context, row_splits1);
      Array1<int32_t> splits2(context, row_splits2);
      row_splits_vec[0].push_back(splits1);
      row_splits_vec[1].push_back(splits2);
      shapes[1] = RaggedShape3(&splits1, nullptr, -1, &splits2, nullptr, -1);
      shapes_ptr[1] = &shapes[1];
    }
    std::vector<std::vector<int32_t>> expected_row_splits = {
        {0, 3, 6},
        {0, 2, 5, 6, 7, 9, 10},
        {0, 2, 3, 4, 6, 7, 10, 13, 14, 15, 17}};

    {
      // axis == 0
      int32_t axis = 0;
      RaggedShape result = Stack(axis, 2, shapes_ptr.data());
      for (int32_t i = 0; i != 3; ++i) {
        CheckArrayData(result.RowSplits(i + 1), expected_row_splits[i]);
      }
    }
    {
      // axis == 1
      int32_t axis = 1;
      RaggedShape result = Stack(axis, 2, shapes_ptr.data());
      RaggedShape transpose = Transpose(result);
      for (int32_t i = 0; i != 3; ++i) {
        CheckArrayData(transpose.RowSplits(i + 1), expected_row_splits[i]);
      }
    }
  }

  {
    // test with random large size
    for (int32_t m = 0; m < 2; ++m) {
      int32_t num_shape = RandInt(2, 100);
      int32_t num_axes = RandInt(2, 4);
      int32_t dim0 = RandInt(1, 100);
      std::vector<RaggedShape> shape_vec(num_shape);
      std::vector<RaggedShape *> shapes(num_shape);
      for (int32_t j = 0; j != num_shape; ++j) {
        RaggedShape shape =
            RandomRaggedShape(false, num_axes, num_axes, 0, 1000).To(context);
        int32_t src_dim0 = shape.Dim0();
        std::vector<int32_t> row_splits_vec(dim0 + 1);
        row_splits_vec[0] = 0;
        for (int32_t n = 1; n < dim0; ++n) {
          row_splits_vec[n] = RandInt(0, src_dim0);
        }
        row_splits_vec[dim0] = src_dim0;
        std::sort(row_splits_vec.begin(), row_splits_vec.end());
        Array1<int32_t> row_splits(context, row_splits_vec);
        RaggedShape first = RaggedShape2(&row_splits, nullptr, -1);
        RaggedShape new_shape = ComposeRaggedShapes(first, shape);
        shape_vec[j] = new_shape;
        shapes[j] = &shape_vec[j];
      }
      std::vector<RaggedShape> cpu_shapes(num_shape);
      for (auto i = 0; i != num_shape; ++i) {
        cpu_shapes[i] = shape_vec[i].To(cpu);
      }

      {
        // axis == 0
        int32_t axis = 0;
        RaggedShape result = Stack(axis, num_shape, shapes.data());
        ASSERT_EQ(result.NumAxes(),
                  num_axes + 2);  // note we append one axis in each shape in
                                  // `shapes` before `Stack`
        ASSERT_EQ(result.Dim0(), num_shape);
        result = result.To(cpu);
        for (auto iter = result.Iterator(); !iter.Done(); iter.Next()) {
          std::vector<int32_t> index = iter.Value();
          int32_t t = result[index];  // don't need the value, just make sure
                                      // it's a valid index.
          int32_t i = index[0];
          index.erase(index.begin());
          // result[i,j,k,l] = (shape[i])[j,k,l]
          i = cpu_shapes[i][index];  // don't need the value, just need to
                                     // make sure it's an allowable index.
        }
      }
      {
        // axis == 1
        int32_t axis = 1;
        RaggedShape result = Stack(axis, num_shape, shapes.data());
        ASSERT_EQ(result.NumAxes(),
                  num_axes + 2);  // note we append one axis in each shape in
                                  // `shapes` before `Stack`
        ASSERT_EQ(result.Dim0(), dim0);
        result = result.To(cpu);
        for (auto iter = result.Iterator(); !iter.Done(); iter.Next()) {
          std::vector<int32_t> index = iter.Value();
          int32_t t = result[index];  // don't need the value, just make sure
                                      // it's a valid index.
          int32_t i = index[1];
          index.erase(index.begin() + 1);
          // result[i,j,k,l] = (shape[j])[i,k,l]
          i = cpu_shapes[i][index];  // don't need the value, just need to
                                     // make sure it's an allowable index.
        }
      }
    }
  }
}
TEST(RaggedShapeOpsTest, TestStack) {
  TestStack<kCpu>();
  TestStack<kCuda>();
}

template <DeviceType d, typename T>
void TestStackRagged() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  // test with random large size
  for (int32_t m = 0; m < 2; ++m) {
    int32_t num_shape = RandInt(2, 100);
    int32_t num_axes = RandInt(2, 4);
    int32_t dim0 = RandInt(1, 100);
    std::vector<Ragged<T>> ragged_vec(num_shape);
    std::vector<Ragged<T> *> ragged(num_shape);
    for (int32_t j = 0; j != num_shape; ++j) {
      RaggedShape shape =
          RandomRaggedShape(false, num_axes, num_axes, 0, 1000).To(context);
      int32_t src_dim0 = shape.Dim0();
      std::vector<int32_t> row_splits_vec(dim0 + 1);
      row_splits_vec[0] = 0;
      for (int32_t n = 1; n < dim0; ++n) {
        row_splits_vec[n] = RandInt(0, src_dim0);
      }
      row_splits_vec[dim0] = src_dim0;
      std::sort(row_splits_vec.begin(), row_splits_vec.end());
      Array1<int32_t> row_splits(context, row_splits_vec);
      RaggedShape first = RaggedShape2(&row_splits, nullptr, -1);
      RaggedShape new_shape = ComposeRaggedShapes(first, shape);
      int32_t num_elems = new_shape.NumElements();
      Array1<T> src_values = RandUniformArray1<T>(context, num_elems, 0, 10000);
      ragged_vec[j] = Ragged<T>(new_shape, src_values);
      ragged[j] = &ragged_vec[j];
    }
    std::vector<Ragged<T>> cpu_ragged_vec(num_shape);
    for (auto j = 0; j != num_shape; ++j) {
      cpu_ragged_vec[j] = ragged_vec[j].To(cpu);
    }

    {
      // axis == 0
      int32_t axis = 0;
      Ragged<T> result = Stack(axis, num_shape, ragged.data());
      ASSERT_EQ(result.NumAxes(),
                num_axes + 2);  // note we append one axis in each shape in
                                // `shapes` before `Stack`
      ASSERT_EQ(result.Dim0(), num_shape);
      result = result.To(cpu);
      RaggedShape &shape = result.shape;
      for (auto iter = shape.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        T value = result[index];
        int32_t i = index[0];
        index.erase(index.begin());
        // result[i,j,k,l] = (shape[i])[j,k,l]
        EXPECT_EQ(value, cpu_ragged_vec[i][index]);
      }
    }
    {
      // axis == 1
      int32_t axis = 1;
      Ragged<T> result = Stack(axis, num_shape, ragged.data());
      ASSERT_EQ(result.NumAxes(),
                num_axes + 2);  // note we append one axis in each shape in
                                // `shapes` before `Stack`
      ASSERT_EQ(result.Dim0(), dim0);
      result = result.To(cpu);
      RaggedShape &shape = result.shape;
      for (auto iter = shape.Iterator(); !iter.Done(); iter.Next()) {
        std::vector<int32_t> index = iter.Value();
        T value = result[index];
        int32_t j = index[1];
        index.erase(index.begin() + 1);
        // result[i,j,k,l] = (shape[j])[i,k,l]
        EXPECT_EQ(value, cpu_ragged_vec[j][index]);
      }
    }
  }
}
TEST(RaggedTest, TestStackRagged) {
  TestStackRagged<kCpu, int32_t>();
  TestStackRagged<kCuda, int32_t>();
  TestStackRagged<kCpu, double>();
  TestStackRagged<kCuda, double>();
}

template <DeviceType d>
void TestMakeTransposable() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // simple case
    const std::vector<int32_t> row_splits1 = {0, 2, 5, 6, 8};
    // const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2, 3, 3};
    const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10, 12, 13};
    // const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 6,
    //                                        6, 7};
    Array1<int32_t> row_splits1_array(context, row_splits1);
    Array1<int32_t> row_splits2_array(context, row_splits2);
    RaggedShape shape = RaggedShape3(&row_splits1_array, nullptr, -1,
                                     &row_splits2_array, nullptr, -1);

    std::vector<std::vector<int32_t>> expected_row_splits = {
        {0, 3, 6, 9, 12}, {0, 2, 3, 3, 4, 6, 7, 10, 10, 10, 12, 13, 13}};
    std::vector<std::vector<int32_t>> expected_row_ids = {
        {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3},
        {0, 0, 1, 3, 4, 4, 5, 6, 6, 6, 9, 9, 10}};

    RaggedShape result = MakeTransposable(shape);
    for (int32_t i = 1; i != 3; ++i) {
      CheckArrayData(result.RowSplits(i), expected_row_splits[i - 1]);
      CheckArrayData(result.RowIds(i), expected_row_ids[i - 1]);
    }
  }

  {
    // test with random large size
    for (int32_t i = 0; i < 2; ++i) {
      int32_t num_axes = RandInt(2, 4);
      RaggedShape shape =
          RandomRaggedShape(true, num_axes, num_axes, 0, 1000).To(context);
      int32_t dim0 = shape.Dim0();
      int32_t max_size = shape.MaxSize(1);
      RaggedShape result = MakeTransposable(shape);
      shape = shape.To(cpu);
      result = result.To(cpu);
      EXPECT_EQ(result.Dim0(), dim0);
      EXPECT_EQ(result.TotSize(1), dim0 * max_size);
      // check if every sub list in axis 1 has the same size
      int32_t *row_splits1 = result.RowSplits(1).Data();
      for (int32_t j = 0; j != dim0 + 1; ++j) {
        EXPECT_EQ(row_splits1[j], j * max_size);
      }
      if (num_axes > 2) {
        for (auto iter = shape.Iterator(); !iter.Done(); iter.Next()) {
          std::vector<int32_t> index = iter.Value();
          EXPECT_EQ(shape[index], result[index]);
        }
      }
    }
  }
}
TEST(RaggedShapeOpsTest, TestMakeTransposable) {
  TestMakeTransposable<kCpu>();
  TestMakeTransposable<kCuda>();
}

}  // namespace k2
