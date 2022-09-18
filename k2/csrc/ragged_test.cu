/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu
 *                                                   Wei Kang)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Yiming Wang
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(RaggedShapeOpsTest, CatMoreAxes) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape1 = RaggedShape(c,
                                     "[ [ [ [ x x ] ] [ [ x ] ] ]"
                                     "  [ [ [ x ] ] ] ]"),
                shape2 = RaggedShape(c,
                                     "[ [ [ [ x ] ] [ [ x ] ] ]"
                                     "  [ [ [ x x ] ] ] ]"),
                shape3 = RaggedShape(c,
                                     "[ [ [ [ ] ] [ [ x ] ] ]"
                                     "  [ [ [ ] ] ] ]");

    RaggedShape cat_axis2_ref =
        RaggedShape(c,
                    "[ [ [ [ x x ] [ x ] [ ] ] [ [ x ] [ x ] [ x ] ] ]"
                    "  [ [ [ x ] [ x x ] [ ] ] ] ]");
    RaggedShape cat_axis3_ref = RaggedShape(c,
                                            "[ [ [ [ x x x ] ] [ [ x x x ] ] ]"
                                            "  [ [ [ x x x ] ] ] ]");
    RaggedShape *srcs[] = {&shape1, &shape2, &shape3};
    Array1<uint32_t> merge_map2;
    Array1<uint32_t> merge_map3;
    RaggedShape cat_axis2 = Cat(2, 3, srcs, &merge_map2);
    RaggedShape cat_axis3 = Cat(3, 3, srcs, &merge_map3);

    K2_CHECK(Equal(cat_axis2, cat_axis2_ref));
    K2_CHECK(Equal(cat_axis3, cat_axis3_ref));

    std::vector<uint32_t> merge_values = {0, 3, 1, 6, 4, 2, 9, 7, 10};
    CheckArrayData(merge_map2, merge_values);
    CheckArrayData(merge_map3, merge_values);
  }
}

TEST(RaggedShapeOpsTest, StackMoreAxes) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape1 = RaggedShape(c,
                                     "[ [ [ [ x x ] ] [ [ x ] ] ]"
                                     "  [ [ [ x ] ] ] ]"),
                shape2 = RaggedShape(c,
                                     "[ [ [ [ x ] ] [ [ x ] ] ]"
                                     "  [ [ [ x x ] ] ] ]"),
                shape3 = RaggedShape(c,
                                     "[ [ [ [ ] ] [ [ x ] ] ]"
                                     "  [ [ [ ] ] ] ]");

    RaggedShape stacked2_ref =
        RaggedShape(c,
                    "[ [ [ [ [ x x ] ] [ [ x ] ] [ [ ] ] ]"
                    "    [ [ [ x ] ] [ [ x ] ] [ [ x ] ] ] ]"
                    "  [ [ [ [ x ] ] [ [ x x ] ] [ [ ] ] ] ] ]");
    RaggedShape stacked3_ref = RaggedShape(c,
                                           "[ [ [ [ [ x x ] [ x ] [ ] ] ]"
                                           "    [ [ [ x ] [ x ] [ x ] ] ] ]"
                                           "  [ [ [ [ x ] [ x x ] [ ] ] ] ] ]");
    RaggedShape *srcs[] = {&shape1, &shape2, &shape3};
    Array1<uint32_t> merge_map2;
    Array1<uint32_t> merge_map3;
    RaggedShape stacked_axis2 = Stack(2, 3, srcs, &merge_map2);
    RaggedShape stacked_axis3 = Stack(3, 3, srcs, &merge_map3);

    K2_CHECK(Equal(stacked_axis2, stacked2_ref));
    K2_CHECK(Equal(stacked_axis3, stacked3_ref));

    std::vector<uint32_t> merge_values = {0, 3, 1, 6, 4, 2, 9, 7, 10};
    CheckArrayData(merge_map2, merge_values);
    CheckArrayData(merge_map3, merge_values);
  }
}

TEST(RaggedShapeOpsTest, Unstack2Axes) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    auto shape = RaggedShape(c, "[ [ x x ] [ x x x ] [ x ] ]");

    std::vector<RaggedShape> out;
    std::vector<Array1<int32_t>> out_map;

    // axis = 0
    Unstack(shape, 0, &out, &out_map);
    K2_CHECK(Equal(out[0], RaggedShape(c, "[ [ x x ] ]")));
    K2_CHECK(Equal(out_map[0], Array1<int32_t>(c, std::vector<int32_t>{0, 1})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ x x x ] ]")));
    K2_CHECK(
        Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{2, 3, 4})));
    K2_CHECK(Equal(out[2], RaggedShape(c, "[ [ x ] ]")));
    K2_CHECK(Equal(out_map[2], Array1<int32_t>(c, std::vector<int32_t>{5})));

    std::vector<RaggedShape *> out_ptr;
    out_ptr.clear();
    for (size_t i = 0; i < out.size(); ++i) out_ptr.emplace_back(&(out[i]));
    auto dest = Stack(0, out.size(), out_ptr.data());
    dest = RemoveAxis(dest, 1);
    K2_CHECK(Equal(dest, shape));

    // axis = 1
    Unstack(shape, 1, &out, &out_map);
    K2_CHECK(Equal(out[0], RaggedShape(c, "[ [ x x x ] ]")));
    K2_CHECK(
        Equal(out_map[0], Array1<int32_t>(c, std::vector<int32_t>{0, 2, 5})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ x x ] ]")));
    K2_CHECK(Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{1, 3})));
    K2_CHECK(Equal(out[2], RaggedShape(c, "[ [ x ] ]")));
    K2_CHECK(Equal(out_map[2], Array1<int32_t>(c, std::vector<int32_t>{4})));
    // can not test Stack here, because the element numbers of axis 1 is not
    // the same
  }
}

TEST(RaggedShapeOpsTest, Unstack) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape(c,
                      "[ [ [ [ x x ] [ x ] ] [ [ x ] [ x x ] ] ]"
                      "  [ [ [ x x x ] ] ] ]");
    std::vector<RaggedShape> out;
    std::vector<Array1<int32_t>> out_map;
    Unstack(shape, 0, &out, &out_map);

    // axis = 0
    K2_CHECK(Equal(out[0],
                   RaggedShape(c, "[ [ [ x x ] [ x ] ] [ [ x ] [ x x ] ] ]")));
    K2_CHECK(Equal(out_map[0],
                   Array1<int32_t>(c, std::vector<int32_t>{0, 1, 2, 3, 4, 5})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ [ x x x ] ] ]")));
    K2_CHECK(
        Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{6, 7, 8})));

    std::vector<RaggedShape *> out_ptr;
    for (size_t i = 0; i < out.size(); ++i) out_ptr.emplace_back(&(out[i]));
    auto dest = Stack(0, out.size(), out_ptr.data());
    K2_CHECK(Equal(dest, shape));

    // axis = 1
    Unstack(shape, 1, &out, &out_map);
    K2_CHECK(
        Equal(out[0], RaggedShape(c, "[ [ [ x x ] [ x ] ] [ [ x x x ] ] ]")));
    K2_CHECK(Equal(out_map[0],
                   Array1<int32_t>(c, std::vector<int32_t>{0, 1, 2, 6, 7, 8})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ [ x ] [ x x ] ] [ ] ]")));
    K2_CHECK(
        Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{3, 4, 5})));

    out_ptr.clear();
    for (size_t i = 0; i < out.size(); ++i) out_ptr.emplace_back(&(out[i]));
    dest = Stack(1, out.size(), out_ptr.data());
    dest = RemoveEmptyLists(dest, 1);
    K2_CHECK(Equal(dest, shape));

    // axis = 2
    Unstack(shape, 2, &out, &out_map);
    K2_CHECK(
        Equal(out[0], RaggedShape(c, "[ [ [ x x ] [ x ] ] [ [ x x x ] ] ]")));
    K2_CHECK(Equal(out_map[0],
                   Array1<int32_t>(c, std::vector<int32_t>{0, 1, 3, 6, 7, 8})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ [ x ] [ x x ] ] [ [ ] ] ]")));
    K2_CHECK(
        Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{2, 4, 5})));

    out_ptr.clear();
    for (size_t i = 0; i < out.size(); ++i) out_ptr.emplace_back(&(out[i]));
    dest = Stack(2, out.size(), out_ptr.data());
    dest = RemoveEmptyLists(dest, 2);
    K2_CHECK(Equal(dest, shape));

    // axis = 3
    Unstack(shape, 3, &out, &out_map);
    K2_CHECK(
        Equal(out[0], RaggedShape(c, "[ [ [ x x ] [ x x ] ] [ [ x ] ] ]")));
    K2_CHECK(Equal(out_map[0],
                   Array1<int32_t>(c, std::vector<int32_t>{0, 2, 3, 4, 6})));
    K2_CHECK(Equal(out[1], RaggedShape(c, "[ [ [ x ] [ x ] ] [ [ x ] ] ]")));
    K2_CHECK(
        Equal(out_map[1], Array1<int32_t>(c, std::vector<int32_t>{1, 5, 7})));
    K2_CHECK(Equal(out[2], RaggedShape(c, "[ [ [ ] [ ] ] [ [ x ] ] ]")));
    K2_CHECK(Equal(out_map[2], Array1<int32_t>(c, std::vector<int32_t>{8})));
    // can not test Stack here, because the element numbers of axis 3 is not
    // the same
  }
}

TEST(RaggedShapeOpsTest, UnstackMoreAxes) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape(c,
                      "[ [ [ [ [ x ] [ ] ] [ [ x x x ] ] ] ]"
                      "  [ [ [ [ x x x ] ] [ [ x x ] ] [ [ x ] ] ]"
                      "    [ [ [ x x ] [ x ] [ ] [ x ] ] ]"
                      "    [ [ [ x ] ] [ [ x ] [ x x x x ] ] ] ]"
                      "  [ [ [ [ x ] ] [ ] ]"
                      "    [ [ [ x x ] ] ] ] ]");

    std::vector<RaggedShape> out;
    std::vector<Array1<int32_t>> out_map;
    std::vector<RaggedShape *> out_ptr;

    for (int32_t axis = 0; axis < 4; axis++) {
      for (bool pad_right : {true, false}) {
        Unstack(shape, axis, pad_right, &out, &out_map);

        out_ptr.clear();
        for (size_t i = 0; i < out.size(); ++i) out_ptr.emplace_back(&(out[i]));
        auto dest = Stack(axis, out.size(), out_ptr.data());
        dest = RemoveEmptyLists(dest, axis);
        K2_CHECK(Equal(dest, RemoveEmptyLists(shape, axis)));
      }
    }
  }
}

TEST(RaggedShapeOpsTest, UnstackRandom) {
  RaggedShape random_shape_ = RandomRaggedShape(true,  // set_row_ids
                                                5,     // min_num_axes
                                                5,     // max_num_axes
                                                1,     // min_num_elements
                                                100);  // max_num_elements
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    auto random_shape0 = random_shape_.To(c);
    std::vector<RaggedShape> out;
    std::vector<RaggedShape *> out_ptr;
    for (int32_t axis = 0; axis < 4; axis++) {
      auto random_shape = RemoveEmptyLists(random_shape0, axis);
      for (bool pad_right : {true, false}) {
        Unstack(random_shape, axis, pad_right, &out, nullptr);

        out_ptr.clear();
        for (size_t i = 0; i < out.size(); ++i) {
          out_ptr.emplace_back(&(out[i]));
        }
        auto dest = Stack(axis, out.size(), out_ptr.data());
        dest = RemoveEmptyLists(dest, axis);

        K2_CHECK(Equal(dest, random_shape));
      }
    }
  }
}

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
    std::vector<RaggedShapeLayer> axes;
    axes.emplace_back(RaggedShapeLayer{Array1<int32_t>(context, row_splits1),
                                       Array1<int32_t>(context, row_ids1),
                                       static_cast<int32_t>(row_ids1.size())});
    axes.emplace_back(RaggedShapeLayer{Array1<int32_t>(context, row_splits2),
                                       Array1<int32_t>(context, row_ids2),
                                       static_cast<int32_t>(row_ids2.size())});
    axes.emplace_back(RaggedShapeLayer{Array1<int32_t>(context, row_splits3),
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

  ASSERT_THROW(RaggedShape(" [ [ x x ] [x] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ [ x x ] [[x]]] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ [ x [] x ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ x ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ x ] [ x ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ x | x ] "), std::runtime_error);

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

  Ragged<int32_t> rs2(" [ [ [ 0 5 ] ] [[10]] ]");
  K2_LOG(INFO) << "rs2 = " << rs2;

  ASSERT_THROW(RaggedShape(" [ [ 0 0 ] [0] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ [ 0 0 ] [[0]]] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ [ 0 [] 0 ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ 0 ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ 0 ] [ 0 ] "), std::runtime_error);
  ASSERT_THROW(RaggedShape(" [ 0 | 0 ] "), std::runtime_error);

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

template <typename T>
void TestMaxPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      const std::vector<int32_t> row_splits = {0};
      RaggedShapeLayer shape_dim;
      shape_dim.row_splits = Array1<int32_t>(context, row_splits);
      shape_dim.cached_tot_size = 0;
      std::vector<RaggedShapeLayer> axes = {shape_dim};
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
      RaggedShapeLayer shape_dim;
      shape_dim.row_splits = Array1<int32_t>(context, row_splits);
      shape_dim.cached_tot_size = row_splits.back();
      std::vector<RaggedShapeLayer> axes = {shape_dim};
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
      max_values.Context()->CopyDataTo(
          max_values.Dim() * max_values.ElementSize(), max_values.Data(), cpu,
          cpu_data.data());
      std::vector<T> expected_data = {3, default_value, 8, default_value};
      EXPECT_EQ(cpu_data, expected_data);
    }

    {
      // test with random large size
      const int32_t min_num_elements = 2000;
      // not random shape is on CPU
      RaggedShape shape =
          RandomRaggedShape(false, 2, 2, min_num_elements, 5000);
      ASSERT_EQ(shape.NumAxes(), 2);
      RaggedShape gpu_shape;
      if (context->GetDeviceType() == kCuda) {
        // copy shape to GPU
        const Array1<T> &row_splits = shape.RowSplits(1);
        RaggedShapeLayer shape_dim;
        shape_dim.row_splits = row_splits.To(GetCudaContext());
        shape_dim.cached_tot_size = shape.NumElements();
        std::vector<RaggedShapeLayer> axes = {shape_dim};
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
        Ragged<T> ragged(context->GetDeviceType() == kCuda ? gpu_shape : shape,
                         values);
        Array1<T> max_values(context, num_rows);
        T default_value = 0;
        MaxPerSublist(ragged, default_value, &max_values);
        EXPECT_EQ(max_values[row], num_elems);
      }
    }
  }
}

TEST(RaggedShapeOpsTest, MaxPerSubListTest) {
  TestMaxPerSubListTest<int32_t>();
}

template <typename T>
void TestArgMaxPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      const std::vector<int32_t> row_splits_vec = {0};
      Array1<int32_t> row_splits(context, row_splits_vec);
      RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
      Array1<T> values(context, 0);
      Ragged<T> ragged(shape, values);

      int32_t num_rows = ragged.shape.Dim0();
      ASSERT_EQ(num_rows, 0);
      Array1<int32_t> argmax_values(context, num_rows);
      // just run to check if there's any error
      ArgMaxPerSublist(ragged, 1, &argmax_values);
      EXPECT_EQ(argmax_values.Dim(), 0);
    }
    {
      // empty case 2
      const std::vector<int32_t> row_splits_vec = {0, 0};
      Array1<int32_t> row_splits(context, row_splits_vec);
      RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
      Array1<T> values(context, 0);
      Ragged<T> ragged(shape, values);

      int32_t num_rows = ragged.shape.Dim0();
      ASSERT_EQ(num_rows, 1);
      Array1<int32_t> argmax_values(context, num_rows);
      // just run to check if there's any error
      ArgMaxPerSublist(ragged, 1, &argmax_values);
      EXPECT_EQ(argmax_values.Dim(), 1);
      std::vector<T> expected_data = {-1};
      CheckArrayData(argmax_values, expected_data);
    }
    {
      const std::vector<int32_t> row_splits_vec = {0, 3, 3, 6, 7};
      Array1<int32_t> row_splits(context, row_splits_vec);
      RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
      const std::vector<T> values_vec = {1, 3, 3, 2, 1, 0, -1};
      Array1<T> values(context, values_vec);
      Ragged<T> ragged(shape, values);

      int32_t num_rows = ragged.shape.Dim0();
      Array1<T> argmax_values(context, num_rows);
      T default_value = 2;
      ArgMaxPerSublist(ragged, default_value, &argmax_values);
      std::vector<T> expected_data = {2, -1, 3, -1};
      CheckArrayData(argmax_values, expected_data);
    }

    {
      // test with random large size
      ContextPtr cpu = GetCpuContext();
      for (int32_t i = 0; i != 10; ++i) {
        Ragged<int32_t> ragged =
            RandomRagged<int32_t>(0, 1000, 2, 4, 0, 5000).To(context);
        int32_t last_axis = ragged.NumAxes() - 1;
        Array1<int32_t> argmax_values(context,
                                      ragged.RowSplits(last_axis).Dim() - 1);
        int32_t default_value = 2;
        ArgMaxPerSublist(ragged, default_value, &argmax_values);

        ragged = ragged.To(cpu);
        argmax_values = argmax_values.To(cpu);
        Array1<int32_t> row_splits = ragged.RowSplits(last_axis);
        int32_t rows = row_splits.Dim() - 1;
        for (int32_t row = 0; row < rows; row++) {
          int32_t begin = row_splits[row], end = row_splits[row + 1];
          int32_t max_val = 2, best_pos = -1;
          for (int32_t pos = begin; pos < end; pos++) {
            if (ragged.values[pos] >= max_val) {
              max_val = ragged.values[pos];
              best_pos = pos;
            }
          }
          EXPECT_EQ(argmax_values[row], best_pos);
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, ArgMaxPerSubListTest) {
  TestArgMaxPerSubListTest<int32_t>();
}

template <typename T>
void TestMinPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    // May add tests for random large size? (but maybe it's fine to not add as
    // we have tested large cases in MaxPerSubList)
  }
}

TEST(RaggedShapeOpsTest, MinPerSubListTest) {
  TestMinPerSubListTest<int32_t>();
}

template <typename T>
void TestAndOrPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // And
      const std::vector<int32_t> row_splits = {0, 2, 2, 5, 6};
      RaggedShapeLayer shape_dim;
      shape_dim.row_splits = Array1<int32_t>(context, row_splits);
      shape_dim.cached_tot_size = row_splits.back();
      std::vector<RaggedShapeLayer> axes = {shape_dim};
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
      RaggedShapeLayer shape_dim;
      shape_dim.row_splits = Array1<int32_t>(context, row_splits);
      shape_dim.cached_tot_size = row_splits.back();
      std::vector<RaggedShapeLayer> axes = {shape_dim};
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
}

TEST(RaggedShapeOpsTest, AndOrPerSubListTest) {
  TestAndOrPerSubListTest<int32_t>();
}

void TestUnsqueeze(const RaggedShape &input_shape) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape src_shape = input_shape.To(context);
    src_shape.Populate();  // set row_ids
    {
      // axis = 0.
      RaggedShape shape = Unsqueeze(src_shape, 0);
      int32_t dim0 = src_shape.Dim0();
      const std::vector<RaggedShapeLayer> &src_axes = src_shape.Layers();
      const std::vector<RaggedShapeLayer> &dest_axes = shape.Layers();

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
        for (size_t i = 0; i != src_axes.size(); ++i) {
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
      const std::vector<RaggedShapeLayer> &src_axes = src_shape.Layers();
      const std::vector<RaggedShapeLayer> &dest_axes = shape.Layers();

      {
        for (int32_t i = 0; i < axis; ++i) {
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
        for (std::size_t i = axis; i < src_axes.size(); ++i) {
          CheckArrayData(src_axes[i].row_splits, dest_axes[i + 1].row_splits);
          CheckArrayData(src_axes[i].row_ids, dest_axes[i + 1].row_ids);
        }
      }
    }
  }
}

TEST_F(RaggedShapeOpsSuiteTest, TestUnsqueeze) {
  TestUnsqueeze(simple_shape_);
  TestUnsqueeze(random_shape_);
}

TEST(RaggedShapeOpsTest, TestUnsqueezeParallel) {
  for (int32_t i = 0; i < 10; i++) {
    ContextPtr c = (i % 2 == 0 ? GetCpuContext() : GetCudaContext());
    int32_t num_shapes = RandInt(0, 10);

    std::vector<RaggedShape *> orig_shapes;
    for (int32_t i = 0; i < num_shapes; i++)
      orig_shapes.push_back(
          new RaggedShape(RandomRaggedShape(false, 2, 5, 0, 1000).To(c)));
    int32_t axis = 0;  // only one supported for now.
    std::vector<RaggedShape> unsqueezed =
        UnsqueezeParallel(num_shapes, orig_shapes.data(), axis);
    for (int32_t i = 0; i < num_shapes; i++) {
      unsqueezed[i].Check();
      RaggedShape temp = RemoveAxis(unsqueezed[i], axis);
      ASSERT_EQ(Equal(temp, *(orig_shapes[i])), true);
      delete orig_shapes[i];
    }
  }
}

void TestRemoveAxis(const RaggedShape &input_shape) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape src_shape = input_shape.To(context);
    ASSERT_EQ(src_shape.NumAxes(), 4);
    {
      // axis = 0.
      int32_t axis = 0;
      RaggedShape shape = RemoveAxis(src_shape, axis);
      const std::vector<RaggedShapeLayer> &src_axes = src_shape.Layers();
      const std::vector<RaggedShapeLayer> &dest_axes = shape.Layers();
      ASSERT_EQ(src_axes.size(), 3);
      ASSERT_EQ(dest_axes.size(), 2);

      {
        for (std::size_t i = 0; i != dest_axes.size(); ++i) {
          CheckArrayData(dest_axes[i].row_splits, src_axes[i + 1].row_splits);
          CheckArrayData(dest_axes[i].row_ids, src_axes[i + 1].row_ids);
        }
      }
    }

    {
      // axis = 1
      int32_t axis = 1;
      RaggedShape shape = RemoveAxis(src_shape, axis);
      const std::vector<RaggedShapeLayer> &src_axes = src_shape.Layers();
      const std::vector<RaggedShapeLayer> &dest_axes = shape.Layers();
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
        for (std::size_t i = 1; i != dest_axes.size(); ++i) {
          CheckArrayData(dest_axes[i].row_splits, src_axes[i + 1].row_splits);
          CheckArrayData(dest_axes[i].row_ids, src_axes[i + 1].row_ids);
        }
      }
    }

    {
      // axis = 3
      int32_t axis = 3;  // the last axis
      RaggedShape shape = RemoveAxis(src_shape, axis);
      const std::vector<RaggedShapeLayer> &src_axes = src_shape.Layers();
      const std::vector<RaggedShapeLayer> &dest_axes = shape.Layers();
      ASSERT_EQ(src_axes.size(), 3);
      ASSERT_EQ(dest_axes.size(), 2);

      {
        for (std::size_t i = 0; i != dest_axes.size(); ++i) {
          CheckArrayData(dest_axes[i].row_splits, src_axes[i].row_splits);
          CheckArrayData(dest_axes[i].row_ids, src_axes[i].row_ids);
        }
      }
    }
  }
}

TEST_F(RaggedShapeOpsSuiteTest, TestRemoveAxis) {
  TestRemoveAxis(simple_shape_);
}

TEST(RaggedShapeOpsTest, TestGetOffsets) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

TEST(RaggedShapeOpsTest, TestTranspose) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

        if (context->GetDeviceType() != kCpu) {
          to_transpose = to_transpose.To(cpu);
          transposed = transposed.To(cpu);
        }

        for (auto iter = transposed.Iterator(); !iter.Done(); iter.Next()) {
          std::vector<int32_t> index = iter.Value();
          int32_t i = transposed[index];  // Just make sure this doesn't crash,
                                          // don't need the value.
          std::swap(index[0], index[1]);
          i = to_transpose[index];  // don't need the value, just need to make
                                    // sure it's an allowable index.
          ++i;  // this line just suppresses the warning `variable i set but not
                // used`
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
}

template <typename T>
void TestTransposeRagged() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    // empty case, fsavec with a empty fsa
    {
      const std::vector<int32_t> row_splits1_vec = {0, 0};
      const std::vector<int32_t> row_splits2_vec = {0};
      Array1<int32_t> row_splits1(context, row_splits1_vec);
      Array1<int32_t> row_splits2(context, row_splits2_vec);
      RaggedShape src_shape =
          RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
      ASSERT_EQ(src_shape.Dim0(), 1);
      ASSERT_EQ(src_shape.TotSize(1), 0);

      Array1<T> values_array(context, 0);
      ASSERT_EQ(values_array.Dim(), src_shape.NumElements());

      Ragged<T> ragged(src_shape, values_array);
      Ragged<T> ans = Transpose(ragged);
      RaggedShape shape = ans.shape;
      // Check shape
      ASSERT_EQ(shape.Dim0(), 0);
      ASSERT_EQ(shape.TotSize(1), 0);
      CheckArrayData(shape.RowSplits(1), std::vector<int32_t>({0}));
      CheckArrayData(shape.RowSplits(2), std::vector<int32_t>({0}));
      K2_CHECK_EQ(shape.RowIds(1).Dim(), 0);
      K2_CHECK_EQ(shape.RowIds(2).Dim(), 0);
      // Check values
      K2_CHECK_EQ(ans.values.Dim(), 0);
    }

    // empty case, fsavec without any fsa
    {
      const std::vector<int32_t> row_splits1_vec = {0};
      const std::vector<int32_t> row_splits2_vec = {0};
      Array1<int32_t> row_splits1(context, row_splits1_vec);
      Array1<int32_t> row_splits2(context, row_splits2_vec);
      RaggedShape src_shape =
          RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);
      ASSERT_EQ(src_shape.Dim0(), 0);
      ASSERT_EQ(src_shape.TotSize(1), 0);

      Array1<T> values_array(context, 0);
      ASSERT_EQ(values_array.Dim(), src_shape.NumElements());

      Ragged<T> ragged(src_shape, values_array);
      Ragged<T> ans = Transpose(ragged);
      RaggedShape shape = ans.shape;
      // Check shape
      ASSERT_EQ(shape.Dim0(), 0);
      ASSERT_EQ(shape.TotSize(1), 0);
      CheckArrayData(shape.RowSplits(1), std::vector<int32_t>({0}));
      CheckArrayData(shape.RowSplits(2), std::vector<int32_t>({0}));
      K2_CHECK_EQ(shape.RowIds(1).Dim(), 0);
      K2_CHECK_EQ(shape.RowIds(2).Dim(), 0);
      // Check values
      K2_CHECK_EQ(ans.values.Dim(), 0);
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
        Array1<T> src_values =
            RandUniformArray1<T>(context, num_elems, 0, 10000);
        Ragged<T> src(to_transpose, src_values);
        Ragged<T> ans = Transpose(src);
        if (context->GetDeviceType() == kCuda) {
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
}
TEST(RaggedTest, TestTransposeRagged) {
  TestTransposeRagged<int32_t>();
  TestTransposeRagged<double>();
}

void TestRaggedShape2(const RaggedShape &shape) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    // note if row_splits == null, then we suppose there's no empty rows after
    // the last row-id in row_ids
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
}

TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape2) {
  TestRaggedShape2(simple_shape_);
  TestRaggedShape2(random_shape_);
}

void TestRaggedShape3(const RaggedShape &shape) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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

    // note if row_splits == null, then we suppose there's no empty rows after
    // the last row-id in row_ids
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
}
TEST_F(RaggedShapeOpsSuiteTest, TestRaggedShape3) {
  TestRaggedShape3(simple_shape_);
  TestRaggedShape3(random_shape_);
}

void TestComposeShape(const RaggedShape &shape) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
}
TEST_F(RaggedShapeOpsSuiteTest, TestComposeShape) {
  TestComposeShape(simple_shape_);
  TestComposeShape(random_shape_);
}

void TestShapeFromTotSize(const RaggedShape &shape) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
}
TEST_F(RaggedShapeOpsSuiteTest, TestShapeFromTotSize) {
  TestShapeFromTotSize(simple_shape_);
  TestShapeFromTotSize(random_shape_);
}

template <typename T>
void TestRagged() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
      std::vector<RaggedShapeLayer> axes;
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits1),
                           Array1<int32_t>(context, row_ids1),
                           static_cast<int32_t>(row_ids1.size())});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits2),
                           Array1<int32_t>(context, row_ids2),
                           static_cast<int32_t>(row_ids2.size())});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits3),
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
      if (context->GetDeviceType() == kCpu) {
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
  TestRagged<int32_t>();
  TestRagged<double>();

  TestSortSublists<int32_t>();
  TestSortSublists<double>();
}

TEST(RaggedShapeOpsTest, TestCat) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
        RaggedShape result = Cat(1, 2, shapes_ptr.data());
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
        RaggedShape result = Cat(0, 2, shapes_ptr.data());

        // get result splits with `SpliceRowSplits` and get result row-ids with
        // `RowSplitsToRowIds``
        std::vector<Array1<int32_t>> result_splits;
        std::vector<Array1<int32_t>> result_ids;
        for (auto i = 0; i < 2; ++i) {
          std::vector<const Array1<int32_t> *> splits_ptr = {
              &row_splits_vec[i][0], &row_splits_vec[i][1]};
          Array1<int32_t> curr_row_splits =
              SpliceRowSplits(2, splits_ptr.data());
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
        // only test case axis == 0, test axis==1 with simple case is good
        // enough as it just calls Stack
        RaggedShape result = Cat(0, num_shape, shapes.data());
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
}

template <typename T>
void TestCatRagged() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    // TODO(haowen): remove duplicate code in TestCat above.
    // test with simple case could be good enough, as we have tested
    // Cat(RaggedShape&) already.
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
      RaggedShape shape = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2,
                                       &ids2, ids2.Dim());
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
      RaggedShape shape = RaggedShape3(&splits1, &ids1, ids1.Dim(), &splits2,
                                       &ids2, ids2.Dim());
      Array1<T> values(context, values_vec);
      ragged_vec[1] = Ragged<T>(shape, values);
      ragged[1] = &ragged_vec[1];
    }

    {
      // axis == 0
      Ragged<T> result = Cat(0, 2, ragged.data());
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
      Ragged<T> result = Cat(1, 2, ragged.data());
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
}
TEST(RaggedTest, TestCatRagged) {
  TestCatRagged<int32_t>();
  TestCatRagged<double>();
}

void CheckResultOfIndex(const ContextPtr &context, RaggedShape shape,
                        Array1<int32_t> new2old, RaggedShape result) {
  K2_CHECK(context->IsCompatible(*shape.Context()));
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  int32_t num_axes = shape.NumAxes();
  int32_t src_dim0 = shape.Dim0(), result_dim0 = result.Dim0();
  EXPECT_EQ(result_dim0, new2old.Dim());

  result.Check();

  for (int32_t i = 0; i < result_dim0; i++) {
    RaggedShape result_part = Arange(result, 0, i, i + 1);
    if (new2old[i] == -1) {
      K2_CHECK_EQ(0, result_part.TotSize(1));
    } else {
      RaggedShape src_part = Arange(shape, 0, new2old[i], new2old[i] + 1);
      K2_CHECK_EQ(true, Equal(src_part, result_part));
    }
  }
}

TEST(RaggedShapeOpsTest, TestIndex) {
  for (int i = 0; i < 5; i++) {
    ContextPtr cpu = GetCpuContext();  // will be used to copy data
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
        RaggedShape result = Index(shape, 0, new2old, &value_indexes_out);
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
          int32_t dim0 = shape.Dim0(), result_dim0 = RandInt(0, 10);
          if (dim0 == 0) result_dim0 = 0;
          std::vector<int32_t> new2old_vec(result_dim0);
          for (int i = 0; i < result_dim0; i++)
            new2old_vec[i] = RandInt(-1, dim0 - 1);
          Array1<int32_t> new2old(context, new2old_vec);
          Array1<int32_t> value_indexes;
          RaggedShape result = Index(shape, 0, new2old, &value_indexes);
          CheckResultOfIndex(context, shape, new2old, result);
          K2_LOG(INFO) << "Value_indexes = " << value_indexes;
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, TestIndexAxis1) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      Ragged<int32_t> input =
          Ragged<int32_t>(" [ [ 1 2 ] [ 3 4 5 ] [ 6 7 ] [ ] ]")
              .To(context);  // NOLINT
      Array1<int32_t> indexes = Array1<int32_t>(" [ 1 0 4 2 6 5 ]").To(context);
      Ragged<int32_t> output =
          Ragged<int32_t>(" [ [ 2 1 ] [ 5 3 ] [ 7 6 ] [ ] ]")
              .To(context);  // NOLINT

      Ragged<int32_t> indexed = Index(input, 1, indexes);
      EXPECT_EQ(Equal(output, indexed), true);
    }
  }
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

TEST(GetTransposeReordering, ThreeAxesEmptyCase) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Ragged<int32_t> ragged("[ [ [ ] ] ]");
    ragged = ragged.To(context);
    Array1<int32_t> order = GetTransposeReordering(ragged, 0);
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

TEST(GetTransposeReordering, RandomFsaVecTest) {
  for (int32_t iter = 0; iter != 8; ++iter) {
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      int n = RandInt(100, 200);
      int32_t min_num_fsas = n;
      int32_t max_num_fsas = n * 2;
      bool acyclic = false;
      int32_t max_symbol = 100;
      int32_t min_num_arcs = min_num_fsas * 10;
      int32_t max_num_arcs = max_num_fsas * 20;

      FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic,
                                 max_symbol, min_num_arcs, max_num_arcs);
      fsas = fsas.To(context);
      Array1<int32_t> dest_states = GetDestStates(fsas, true);
      Ragged<int32_t> dest_states_tensor(fsas.shape, dest_states);
      int32_t num_states = fsas.TotSize(1);
      int32_t num_arcs = fsas.TotSize(2);
      Array1<int32_t> order =
          GetTransposeReordering(dest_states_tensor, num_states);
      Sort(&order);
      ASSERT_EQ(order.Dim(), num_arcs);
      Array1<int32_t> expected = Range<int32_t>(context, num_arcs, 0);
      CheckArrayData(order, expected);
    }
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

TEST(ChangeSublistSizePinned, TwoAxes) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 2, 5, 5});
      RaggedShape src = RaggedShape2(&row_splits1, nullptr, -1);

      int32_t size_delta = 2;
      RaggedShape dst = ChangeSublistSizePinned(src, size_delta);
      CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 4, 9, 9});

      size_delta = -3;
      dst = ChangeSublistSizePinned(src, size_delta);
      CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 0, 0, 0});

      size_delta = 0;
      dst = ChangeSublistSizePinned(src, size_delta);
      CheckArrayData(dst.RowSplits(1), std::vector<int32_t>{0, 2, 5, 5});
    }
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

TEST(ChangeSublistSizePinned, ThreeAxes) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    /*
     [
       [ [x, x, x], [x, x] ]
       [ [x], [x, x], [], [x, x, x] ]
     ]
     */
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 2, 6});
    Array1<int32_t> row_splits2(context,
                                std::vector<int32_t>{0, 3, 5, 6, 8, 8, 11});
    RaggedShape src =
        RaggedShape3(&row_splits1, nullptr, -1, &row_splits2, nullptr, -1);

    int32_t size_delta = 2;
    RaggedShape dst = ChangeSublistSizePinned(src, size_delta);
    CheckArrayData(dst.RowSplits(2),
                   std::vector<int32_t>{0, 5, 9, 12, 16, 16, 21});

    size_delta = -2;

    dst = ChangeSublistSizePinned(src, size_delta);
    CheckArrayData(dst.RowSplits(2), std::vector<int32_t>{0, 1, 1, 1, 1, 1, 2});

    size_delta = 0;
    dst = ChangeSublistSizePinned(src, size_delta);
    CheckArrayData(dst.RowSplits(2),
                   std::vector<int32_t>{0, 3, 5, 6, 8, 8, 11});
  }
}

TEST(RaggedShapeOpsTest, TestGetCountsPartitioned) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
    std::vector<int32_t> result_row_splits(
        row_splits.Data(), row_splits.Data() + row_splits.Dim());
    EXPECT_EQ(result_row_splits, ans_row_splits_vec);
    // check values
    std::vector<int32_t> expected_data = {2, 1, 1, 0, 0, 2, 0, 3, 0, 1};
    Array1<int32_t> values = result.values.To(cpu);
    std::vector<int32_t> data(values.Data(), values.Data() + values.Dim());
    EXPECT_EQ(data, expected_data);
  }
}

TEST(RaggedShapeOpsTest, TestStack) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
        RaggedShape result2 = Stack(axis, 1, shapes_ptr.data());
        RaggedShape orig = result2.Index(0, 0);
        EXPECT_TRUE(Equal(orig, shapes[0]));
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
}

template <typename T>
void TestStackRagged() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
        Array1<T> src_values =
            RandUniformArray1<T>(context, num_elems, 0, 10000);
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
}

TEST(RaggedTest, TestStackRagged) {
  TestStackRagged<int32_t>();
  TestStackRagged<double>();
}

template <typename T>
void TestUnstackRagged() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    // two axes
    auto ragged = Ragged<T>(c, "[ [ 10 20 ] [ 30 40 50 ] [ 60 ] ]");
    std::vector<Ragged<T>> out;

    // axis = 0
    Unstack(ragged, 0, &out);
    K2_CHECK(Equal(out[0], Ragged<T>(c, "[ [ 10 20 ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ 30 40 50 ] ]")));
    K2_CHECK(Equal(out[2], Ragged<T>(c, "[ [ 60 ] ]")));

    // axis = 1
    Unstack(ragged, 1, &out);
    K2_CHECK(Equal(out[0], Ragged<T>(c, "[ [ 10 30 60 ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ 20 40 ] ]")));
    K2_CHECK(Equal(out[2], Ragged<T>(c, "[ [ 50 ] ]")));

    // more axes
    ragged = Ragged<T>(c,
                       "[ [ [ [ 1 11 21 ] [ 21 22 ] [ 31 ] ]"
                       "    [ [ 41 ] [ 51 ] ] ]"
                       "  [ [ [ 61 62 63 ] ] ] ]");

    // axis = 0
    Unstack(ragged, 0, &out);
    K2_CHECK(Equal(
        out[0],
        Ragged<T>(c,
                  "[ [ [ 1 11 21 ] [ 21 22 ] [ 31 ] ] [ [ 41 ] [ 51 ] ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ [ 61 62 63 ] ] ]")));

    // axis = 1
    Unstack(ragged, 1, &out);
    K2_CHECK(Equal(
        out[0],
        Ragged<T>(c, "[ [ [ 1 11 21 ] [ 21 22 ] [ 31 ] ] [ [ 61 62 63 ] ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ [ 41 ] [ 51 ] ] [ ] ]")));

    // axis = 2
    Unstack(ragged, 2, &out);
    K2_CHECK(Equal(
        out[0], Ragged<T>(c, "[ [ [ 1 11 21 ] [ 41 ] ] [ [ 61 62 63 ] ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ [ 21 22 ] [ 51 ] ] [ [ ] ] ]")));
    K2_CHECK(Equal(out[2], Ragged<T>(c, "[ [ [ 31 ] [ ] ] [ [ ] ] ]")));

    // axis = 3
    Unstack(ragged, 3, &out);
    K2_CHECK(Equal(out[0],
                   Ragged<T>(c, "[ [ [ 1 21 31 ] [ 41 51 ] ] [ [ 61 ] ] ]")));
    K2_CHECK(Equal(out[1], Ragged<T>(c, "[ [ [ 11 22 ] [ ] ] [ [ 62 ] ] ]")));
    K2_CHECK(Equal(out[2], Ragged<T>(c, "[ [ [ 21 ] [ ] ] [ [ 63 ] ] ]")));
  }
}

TEST(RaggedTest, TestUnstack) {
  TestUnstackRagged<int32_t>();
  TestUnstackRagged<float>();
  TestUnstackRagged<double>();
}

TEST(RaggedTest, TestMaxSize) {
  for (int32_t i = 0; i <= 10; i++) {
    ContextPtr c = (i % 2 == 0 ? GetCpuContext() : GetCudaContext());
    int32_t num_axes = RandInt(2, 4);
    RaggedShape shape =
        RandomRaggedShape(true, num_axes, num_axes, 0, 1000).To(c);
    int32_t axis = RandInt(1, num_axes - 1);
    int32_t max_size = shape.MaxSize(axis);
    if (axis == 0) {
      K2_CHECK(max_size == shape.Dim0());
    } else {
      Array1<int32_t> row_splits = shape.RowSplits(axis).To(GetCpuContext());
      int32_t *row_splits_data = row_splits.Data();
      int32_t m = 0;
      for (int32_t i = 0; i + 1 < row_splits.Dim(); i++) {
        int32_t size = row_splits_data[i + 1] - row_splits_data[i];
        if (size > m) m = size;
      }
      ASSERT_EQ(m, max_size);
    }
  }
}

TEST(RaggedShapeOpsTest, TestMakeTransposable) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
            const std::vector<int32_t> &index = iter.Value();
            EXPECT_EQ(shape[index], result[index]);
          }
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, PrefixTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // simple case
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6, 8};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10, 12, 13};
      Array1<int32_t> row_splits1_array(context, row_splits1);
      Array1<int32_t> row_splits2_array(context, row_splits2);
      RaggedShape shape = RaggedShape3(&row_splits1_array, nullptr, -1,
                                       &row_splits2_array, nullptr, -1);
      int32_t dim0 = shape.Dim0();
      int32_t num_axes = shape.NumAxes();
      EXPECT_EQ(dim0, 4);
      EXPECT_EQ(num_axes, 3);
      {
        // n == 0
        int32_t n = 0;
        std::vector<std::vector<int32_t>> expected_row_splits = {{0}, {0}};
        RaggedShape result = Prefix(shape, n);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), n);
        EXPECT_EQ(result.NumAxes(), num_axes);
        for (int32_t i = 1; i != num_axes; ++i) {
          CheckArrayData(result.RowSplits(i), expected_row_splits[i - 1]);
        }
      }

      {
        // n > 0 && n < dim0
        int32_t n = 2;
        std::vector<std::vector<int32_t>> expected_row_splits = {
            {0, 2, 5}, {0, 2, 3, 4, 6, 7}};
        RaggedShape result = Prefix(shape, n);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), n);
        EXPECT_EQ(result.NumAxes(), num_axes);
        for (int32_t i = 1; i != num_axes; ++i) {
          CheckArrayData(result.RowSplits(i), expected_row_splits[i - 1]);
        }
      }

      {
        // n == dim0
        int32_t n = 4;
        std::vector<std::vector<int32_t>> expected_row_splits = {
            {0, 2, 5}, {0, 2, 3, 4, 6, 7}};
        RaggedShape result = Prefix(shape, n);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), n);
        EXPECT_EQ(result.NumAxes(), num_axes);
        CheckArrayData(result.RowSplits(1), row_splits1);
        CheckArrayData(result.RowSplits(2), row_splits2);
      }
    }

    {
      // test with random large size
      for (int32_t i = 0; i < 2; ++i) {
        RaggedShape shape = RandomRaggedShape(false, 2, 4, 0, 1000).To(context);
        int32_t dim0 = shape.Dim0();
        int32_t num_axes = shape.NumAxes();
        int32_t n = RandInt(0, dim0);
        RaggedShape result = Prefix(shape, n);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), n);
        EXPECT_EQ(result.NumAxes(), num_axes);
        // just check row_splits1 here would be fine, as we have tested it with
        // simple case. We just confirm it can run successfully with kinds of
        // different random shapes.
        CheckArrayData(result.RowSplits(1), shape.RowSplits(1).Range(0, n + 1));
      }
    }
  }
}

TEST(RaggedShapeOpsTest, GetPrefixesTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // test with random large size
      for (int32_t i = 0; i < 2; ++i) {
        RaggedShape shape = RandomRaggedShape(false, 2, 4, 0, 1000).To(context);
        int32_t dim0 = shape.Dim0();
        int32_t num_axes = shape.NumAxes();
        int32_t ans_num = RandInt(0, 10);
        std::vector<int32_t> sizes;
        for (int32_t j = 0; j != ans_num; ++j)
          sizes.push_back(RandInt(0, dim0));
        ASSERT_EQ(sizes.size(), ans_num);
        std::vector<RaggedShape> ans = GetPrefixes(shape, sizes);
        ASSERT_EQ(ans.size(), ans_num);

        for (int32_t j = 0; j != ans_num; ++j) {
          int32_t n = sizes[j];

          RaggedShape ans_j = ans[j];
          EXPECT_TRUE(IsCompatible(shape, ans_j));
          EXPECT_EQ(ans_j.Dim0(), n);
          EXPECT_EQ(ans_j.NumAxes(), num_axes);

          RaggedShape result = Prefix(shape, n);
          EXPECT_TRUE(IsCompatible(shape, result));
          EXPECT_EQ(result.Dim0(), n);
          EXPECT_EQ(result.NumAxes(), num_axes);

          for (int32_t m = 1; m != num_axes; ++m) {
            EXPECT_TRUE(Equal(result.RowSplits(m), ans_j.RowSplits(m)));
          }
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, ArangeTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // simple case
      const std::vector<int32_t> row_splits1 = {0, 2, 3, 4, 6, 7, 10};
      // const std::vector<int32_t> row_ids1 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
      const std::vector<int32_t> row_splits2 = {0,  2,  3,  5,  8, 9,
                                                12, 13, 15, 15, 16};
      // const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 2, 3, 3, 3,
      // 4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_splits1_array(context, row_splits1);
      Array1<int32_t> row_splits2_array(context, row_splits2);
      RaggedShape shape = RaggedShape3(&row_splits1_array, nullptr, -1,
                                       &row_splits2_array, nullptr, -1);
      std::vector<int32_t> values(shape.NumElements());
      std::iota(values.begin(), values.end(), 10);
      Array1<int32_t> values_array(context, values);
      Ragged<int32_t> ragged(shape, values_array);
      int32_t dim0 = shape.Dim0();
      int32_t num_axes = shape.NumAxes();
      EXPECT_EQ(dim0, 6);
      EXPECT_EQ(num_axes, 3);
      {
        // axis == 0, begin == end
        int32_t axis = 0;
        int32_t begin = 1, end = 1;
        std::vector<std::vector<int32_t>> expected_row_splits = {{0}, {0}};
        std::pair<int32_t, int32_t> value_range;
        RaggedShape result = Arange(shape, axis, begin, end, &value_range);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), 0);
        EXPECT_EQ(result.NumAxes(), num_axes);
        for (int32_t i = 1; i != num_axes; ++i) {
          CheckArrayData(result.RowSplits(i), expected_row_splits[i - 1]);
        }
        std::pair<int32_t, int32_t> expected_value_range = {1, 1};
        EXPECT_EQ(value_range, expected_value_range);
        EXPECT_EQ(result.NumElements(), value_range.second - value_range.first);

        // test `Arange` for ragged array
        Ragged<int32_t> ragged_result = Arange(ragged, axis, begin, end);
        EXPECT_EQ(ragged_result.values.Dim(), 0);
      }

      {
        // axis == 0, begin  < end == Dim0() + 1
        int32_t axis = 0;
        int32_t begin = 3, end = 6;
        std::vector<std::vector<int32_t>> expected_row_splits = {
            {0, 2, 3, 6}, {0, 1, 4, 5, 7, 7, 8}};
        std::pair<int32_t, int32_t> value_range;
        RaggedShape result = Arange(shape, axis, begin, end, &value_range);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.NumAxes(), num_axes);
        for (int32_t i = 1; i != num_axes; ++i) {
          CheckArrayData(result.RowSplits(i), expected_row_splits[i - 1]);
        }
        std::pair<int32_t, int32_t> expected_value_range = {8, 16};
        EXPECT_EQ(value_range, expected_value_range);
        EXPECT_EQ(result.NumElements(), value_range.second - value_range.first);

        // test `Arange` for ragged array
        Ragged<int32_t> ragged_result = Arange(ragged, axis, begin, end);
        std::vector<int32_t> expected_values = {18, 19, 20, 21, 22, 23, 24, 25};
        CheckArrayData(ragged_result.values, expected_values);
      }

      {
        // axis == 1
        int32_t axis = 1;
        int32_t begin = 6, end = 8;
        std::vector<int32_t> expected_row_splits = {0, 1, 3};
        std::pair<int32_t, int32_t> value_range;
        RaggedShape result = Arange(shape, axis, begin, end, &value_range);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.NumAxes(), 2);
        CheckArrayData(result.RowSplits(1), expected_row_splits);
        std::pair<int32_t, int32_t> expected_value_range = {12, 15};
        EXPECT_EQ(value_range, expected_value_range);
        EXPECT_EQ(result.NumElements(), value_range.second - value_range.first);

        // test `Arange` for ragged array
        Ragged<int32_t> ragged_result = Arange(ragged, axis, begin, end);
        std::vector<int32_t> expected_values = {22, 23, 24};
        CheckArrayData(ragged_result.values, expected_values);
      }
    }

    {
      // test with random large size
      for (int32_t i = 0; i < 2; ++i) {
        RaggedShape shape = RandomRaggedShape(false, 2, 4, 0, 1000).To(context);
        int32_t num_axes = shape.NumAxes();
        int32_t axis = RandInt(0, num_axes - 2);
        int32_t tot_size = shape.TotSize(axis);
        int32_t begin = RandInt(0, tot_size);
        int32_t end = RandInt(begin, tot_size);
        std::pair<int32_t, int32_t> value_range;
        RaggedShape result = Arange(shape, axis, begin, end, &value_range);
        EXPECT_TRUE(IsCompatible(shape, result));
        EXPECT_EQ(result.Dim0(), std::max(0, end - begin));
        EXPECT_EQ(result.NumAxes(), num_axes - axis);
        // just check row_splits1 here would be fine, as we have tested it with
        // simple case. We just confirm it can run successfully with kinds of
        // different random shapes.
        if (begin == end) {
          CheckArrayData(result.RowSplits(1), std::vector<int32_t>{0});
        } else {
          Array1<int32_t> row_splits1 =
              shape.RowSplits(axis + 1).Arange(begin, end + 1);
          row_splits1 = Minus(row_splits1, row_splits1[0]);
          CheckArrayData(result.RowSplits(1), row_splits1);
        }
        EXPECT_EQ(result.NumElements(), value_range.second - value_range.first);
      }
    }
  }
}

TEST(RaggedShapeOpsTest, Merge) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape1 = RaggedShape("[ [ x x ] [ x ] [] ]")
                             .To(c),  // m: 0 3 6, m_out:  0 3, 6,
        shape2 = RaggedShape("[ [ x] [ x x x ] ]")
                     .To(c),  // m: 1 4, m_out: 1, 4 7 10
        shape3 =
            RaggedShape("[ [ ] [ x x ] [] ]").To(c);  // m: 2 5 8, m_out: ,2 5,

    RaggedShape ans_ref =
        RaggedShape("[ [] [x] [x x x] [] [] [x x] [x x] [x] ]").To(c);

    // This is a mixed-up kind of merge map that doesn't appear naturally (they
    // are always in-order from each source, right now) but it should still
    // work.
    std::vector<uint32_t> merge_map_data = {6, 1, 4, 8, 2, 5, 0, 3};
    Array1<uint32_t> merge_map_in(c, merge_map_data);
    RaggedShape *srcs[] = {&shape1, &shape2, &shape3};

    Array1<uint32_t> merge_map_out;
    RaggedShape merged = Merge(3, srcs, merge_map_in, &merge_map_out);

    ASSERT_EQ(true, Equal(ans_ref, merged));

    std::vector<uint32_t> merge_map_out_data = {1, 4, 7, 10, 2, 5, 0, 3, 6};
    CheckArrayData(merge_map_out, merge_map_out_data);
  }
}

TEST(RaggedTest, AddSuffixToRaggedTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // test with random large size
      for (int32_t i = 0; i < 10; ++i) {
        Ragged<int32_t> src = RandomRagged<int32_t>().To(context);
        int32_t num_axes = src.NumAxes();
        Array1<int32_t> suffix = RandUniformArray1<int32_t>(
            context, src.TotSize(num_axes - 2), 0, 100);
        Ragged<int32_t> dst = AddSuffixToRagged(src, suffix);
        EXPECT_EQ(dst.NumAxes(), num_axes);
        EXPECT_EQ(dst.NumElements(), src.NumElements() + suffix.Dim());
        Ragged<int32_t> src_cpu = src.To(GetCpuContext());
        Ragged<int32_t> dst_cpu = dst.To(GetCpuContext());
        for (RaggedShapeIndexIterator src_iter = src_cpu.shape.Iterator();
             !src_iter.Done(); src_iter.Next()) {
          const std::vector<int32_t> &src_indexes = src_iter.Value();
          EXPECT_EQ(dst_cpu[src_indexes], src_cpu[src_indexes]);
        }
        Array1<int32_t> src_row_splits = src_cpu.RowSplits(num_axes - 1);
        Array1<int32_t> suffix_cpu = suffix.To(GetCpuContext());
        for (int32_t i = 0; i < suffix.Dim(); ++i) {
          EXPECT_EQ(dst_cpu.values[src_row_splits[i + 1] + i], suffix_cpu[i]);
        }
      }
    }
  }
}

TEST(RaggedTest, AddPrefixToRaggedTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // test with random large size
      for (int32_t i = 0; i < 10; ++i) {
        Ragged<int32_t> src = RandomRagged<int32_t>().To(context);
        int32_t num_axes = src.NumAxes();
        Array1<int32_t> prefix = RandUniformArray1<int32_t>(
            context, src.TotSize(num_axes - 2), 0, 100);
        Ragged<int32_t> dst = AddPrefixToRagged(src, prefix);
        EXPECT_EQ(dst.NumAxes(), num_axes);
        EXPECT_EQ(dst.NumElements(), src.NumElements() + prefix.Dim());
        Ragged<int32_t> src_cpu = src.To(GetCpuContext());
        Ragged<int32_t> dst_cpu = dst.To(GetCpuContext());
        for (RaggedShapeIndexIterator src_iter = src_cpu.shape.Iterator();
             !src_iter.Done(); src_iter.Next()) {
          const std::vector<int32_t> &src_indexes = src_iter.Value();
          std::vector<int32_t> dst_indexes(src_indexes);
          dst_indexes.back() += 1;  // increase the last index by 1
          EXPECT_EQ(dst_cpu[dst_indexes], src_cpu[src_indexes]);
        }
        Array1<int32_t> src_row_splits = src_cpu.RowSplits(num_axes - 1);
        Array1<int32_t> prefix_cpu = prefix.To(GetCpuContext());
        for (int32_t i = 0; i < prefix.Dim(); ++i) {
          EXPECT_EQ(dst_cpu.values[src_row_splits[i] + i], prefix_cpu[i]);
        }
      }
    }
  }
}

TEST(RaggedTest, RemoveValuesLeq) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<int32_t> r = Ragged<int32_t>(" [ [ 3 4 ] [ 5 7 8 ] ]").To(c),
                    s3 = Ragged<int32_t>(" [ [4] [5 7 8]]").To(c),
                    s5 = Ragged<int32_t>(" [ [] [ 7 8]]").To(c);
    Ragged<int32_t> ans1 = RemoveValuesLeq(r, 3), ans2 = RemoveValuesLeq(r, 5);
    K2_LOG(INFO) << "ans2 = " << ans2;
    EXPECT_EQ(true, Equal(ans1, s3));
    EXPECT_EQ(true, Equal(ans2, s5));
  }
}

TEST(RaggedTest, IndexArrayRagged) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<int32_t> r = Ragged<int32_t>(" [ [ 2 0 ] [ 1 2 3 ] ]").To(c);
    Array1<float> f(c, std::vector<float>({0.0, 1.0, 2.0, 3.0, 4.0}));

    Ragged<float> fr = Ragged<float>(" [ [ 2.0 0.0 ] [ 1.0 2.0 3.0 ] ]").To(c),
                  ans = Index(f, r);
    EXPECT_EQ(true, Equal(ans, fr));
  }
}

TEST(RaggedTest, IndexRaggedRagged) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<int32_t> r = Ragged<int32_t>(" [ [ 2 0 ] [ 1 2 3 ] ]").To(c);

    Ragged<int32_t> s =
        Ragged<int32_t>(" [ [ 10 10 ] [ 11 ] [ 12 12 ] [ 13 ] [ 14 14] ]")
            .To(c);  // NOLINT

    Ragged<int32_t> sr1 =
        Ragged<int32_t>(" [ [ [12 12] [10 10] ] [ [11] [12 12] [13] ] ]")
            .To(c);  // NOLINT

    Ragged<int32_t> sr2 =
        Ragged<int32_t>(" [ [ 12 12 10 10 ] [ 11 12 12 13 ] ]")
            .To(c);  // NOLINT

    EXPECT_EQ(true, Equal(Index(s, r, false), sr1));
    EXPECT_EQ(true, Equal(Index(s, r, true), sr2));
  }
}

TEST(RaggedShapeOpsTest, CoveringShape) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    {
      // simple case
      RaggedShape shape1 = RaggedShape("[ [ x x ] [] [ x ] ]").To(c),
                  shape2 = RaggedShape("[ [ x] [] [ x x x ] ]").To(c),
                  shape3 = RaggedShape("[ [] [] [ x x ] ]").To(c);

      RaggedShape expected = RaggedShape("[ [x x] [] [x x x] ]").To(c);
      RaggedShape *srcs[] = {&shape1, &shape2, &shape3};
      RaggedShape ans = CoveringShape(3, srcs);
      EXPECT_TRUE(Equal(expected, ans));

      // test CoveringShapeForwardMap
      {
        Array1<int32_t> elem_map = CoveringShapeForwardMap(shape1, ans);
        std::vector<int32_t> expected_map = {0, 1, 2, -1, -1};
        CheckArrayData(elem_map, expected_map);
      }
      {
        Array1<int32_t> elem_map = CoveringShapeForwardMap(shape2, ans);
        std::vector<int32_t> expected_map = {0, -1, 1, 2, 3};
        CheckArrayData(elem_map, expected_map);
      }
      {
        Array1<int32_t> elem_map = CoveringShapeForwardMap(shape3, ans);
        std::vector<int32_t> expected_map = {-1, -1, 0, 1, -1};
        CheckArrayData(elem_map, expected_map);
      }
    }
    {
      // another simple case: only one src
      RaggedShape shape1 = RaggedShape("[ [ x x ] [ x ] [] ]").To(c);
      RaggedShape *srcs[] = {&shape1};
      RaggedShape ans = CoveringShape(1, srcs);
      EXPECT_TRUE(Equal(shape1, ans));

      // test CoveringShapeForwardMap
      Array1<int32_t> elem_map = CoveringShapeForwardMap(shape1, ans);
      std::vector<int32_t> expected_map = {0, 1, 2};
      CheckArrayData(elem_map, expected_map);
    }
    {
      // random case
      for (int32_t i = 0; i != 1; ++i) {
        int32_t num_shape = RandInt(1, 100);
        int32_t dim0 = RandInt(1, 1000);
        std::vector<RaggedShape> shape_vec(num_shape);
        std::vector<RaggedShape *> shapes(num_shape);
        for (int32_t j = 0; j != num_shape; ++j) {
          Array1<int32_t> row_sizes =
              RandUniformArray1<int32_t>(c, dim0 + 1, 0, 100);
          ExclusiveSum(row_sizes, &row_sizes);
          shape_vec[j] = RaggedShape2(&row_sizes, nullptr, -1);
          ASSERT_TRUE(shape_vec[j].Context()->IsCompatible(*c));
          ASSERT_EQ(shape_vec[j].Dim0(), dim0);
          shapes[j] = &shape_vec[j];
        }
        RaggedShape ans = CoveringShape(num_shape, shapes.data());
        std::vector<Array1<int32_t>> elem_map(num_shape);
        for (int32_t j = 0; j != num_shape; ++j) {
          elem_map[j] = CoveringShapeForwardMap(shape_vec[j], ans);
        }
        // check ans
        ASSERT_EQ(ans.NumAxes(), 2);
        ASSERT_EQ(ans.Dim0(), dim0);
        ASSERT_TRUE(ans.Context()->IsCompatible(*c));
        ContextPtr cpu = GetCpuContext();
        ans = ans.To(cpu);
        for (int32_t j = 0; j != num_shape; ++j)
          shape_vec[j] = shape_vec[j].To(cpu);
        for (int32_t d = 0; d != dim0; ++d) {
          int32_t max_row_size = 0;
          for (int32_t j = 0; j != num_shape; ++j)
            max_row_size = std::max(
                shape_vec[j].RowSplits(1)[d + 1] - shape_vec[j].RowSplits(1)[d],
                max_row_size);
          EXPECT_EQ(max_row_size,
                    ans.RowSplits(1)[d + 1] - ans.RowSplits(1)[d]);
        }

        // test CoveringShapeForwardMap
        for (int32_t j = 0; j != num_shape; ++j) {
          Array1<int32_t> cur_elem_map = elem_map[j].To(cpu);
          ASSERT_EQ(cur_elem_map.Dim(), ans.NumElements());
          int32_t n = 0;
          for (RaggedShapeIndexIterator ans_iter = ans.Iterator();
               !ans_iter.Done(); ans_iter.Next()) {
            const std::vector<int32_t> &ans_indexes = ans_iter.Value();
            int32_t src_shape_linear_index = cur_elem_map[n];
            if (src_shape_linear_index != -1) {
              EXPECT_EQ(src_shape_linear_index, shape_vec[j][ans_indexes]);
            }
            ++n;
          }
        }
      }
    }
  }
}

TEST(RaggedShapeOpsTest, RaggedShapeAxis0Splitter) {
  for (int32_t i = 0; i < 20; i++) {
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      RaggedShape random = RandomRaggedShape(false, 3, 6, 0, 2000);
      int32_t dim0 = random.Dim0();
      RaggedShapeAxis0Splitter splitter(random);
      for (int32_t i = 0; i < dim0; i++) {
        int32_t offset, offset2, offset3;
        RaggedShape sub_shape1 = random.Index(0, i, &offset),
                    sub_shape2 = splitter.GetElement(i, &offset2);
        offset3 = splitter.GetOffset(i, random.NumAxes() - 1);
        EXPECT_EQ(offset, offset2);
        EXPECT_EQ(offset, offset3);
        EXPECT_EQ(Equal(sub_shape1, sub_shape2), true);
      }
    }
  }
}
template <typename T>
static void TestSegmentedExclusiveSum() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    {
      // simple case
      Ragged<T> src("[ [1 2 3 -1] [3 4 -1] [] [5 6 7 -1] ]");
      src = src.To(c);
      Array1<T> dst(c, src.NumElements());
      SegmentedExclusiveSum(src, &dst);

      std::vector<T> expected = {0, 1, 3, 6,
                                 //
                                 0, 3, 7,
                                 //
                                 0, 5, 11, 18};
      CheckArrayData(dst, expected);

      // &src.values == dst
      SegmentedExclusiveSum(src, &src.values);
      CheckArrayData(src.values, expected);
    }
    {
      // random case, we assume the implementation for cpu is correct and only
      // test for Cuda version
      if (c->GetDeviceType() == kCuda) {
        for (int32_t i = 0; i != 2; ++i) {
          Ragged<T> cpu_ragged = RandomRagged<T>(-1000, 1000, 2, 4, 0, 5000);
          int32_t dim = cpu_ragged.NumElements();
          Array1<T> cpu_dst(GetCpuContext(), dim);
          SegmentedExclusiveSum(cpu_ragged, &cpu_dst);

          Ragged<T> ragged = cpu_ragged.To(c);
          Array1<T> dst(c, dim);
          SegmentedExclusiveSum(ragged, &dst);
          CheckArrayData(dst, cpu_dst, (T)0.1);
        }
      }
    }
  }
}

TEST(RaggedOpsTest, SegmentedExclusiveSum) {
  TestSegmentedExclusiveSum<int32_t>();
  TestSegmentedExclusiveSum<float>();
  TestSegmentedExclusiveSum<double>();
}

TEST(RaggedOpsTest, TestComputeHash) {
  for (int32_t i = 0; i < 20; i++) {
    Ragged<int32_t> src = RandomRagged<int32_t>(
                        std::numeric_limits<int32_t>::min(),
                        std::numeric_limits<int32_t>::max(), 2, 4, 0, 20000),
                    src_gpu = src.To(GetCudaContext());
    {
      Array1<int64_t> hash1 = ComputeHash<int64_t>(src),
                      hash2 = ComputeHash<int64_t>(src_gpu).To(GetCpuContext());
      EXPECT_EQ(Equal(hash1, hash2), true);
    }

    {
      Array1<int32_t> hash1 = ComputeHash<int32_t>(src),
                      hash2 = ComputeHash<int32_t>(src_gpu).To(GetCpuContext());
      EXPECT_EQ(Equal(hash1, hash2), true);
    }
  }
}

TEST(RaggedOpsTest, TestUniqueSequences) {
  for (int32_t i = 0; i < 20; i++) {
    for (auto &c : {GetCpuContext(), GetCudaContext()}) {
      Ragged<int32_t> src = RandomRagged<int32_t>(0, 3, 2, 4, 0, 20000).To(c);

      Ragged<int32_t> unique = UniqueSequences(src);

      if (src.NumAxes() == 2) {
        src = Unsqueeze(src, 0);
        unique = Unsqueeze(unique, 0);
      }

      ContextPtr cpu = GetCpuContext();
      Array1<int32_t> hash_src = ComputeHash<int32_t>(src).To(cpu),
                      hash_unique = ComputeHash<int32_t>(unique).To(cpu);

      RaggedShape src_hash_shape =
          RemoveAxis(src.shape, src.NumAxes() - 1).To(cpu);
      src_hash_shape = GetLayer(src_hash_shape, src_hash_shape.NumLayers() - 1);

      RaggedShape unique_hash_shape =
          RemoveAxis(unique.shape, unique.NumAxes() - 1).To(cpu);
      unique_hash_shape =
          GetLayer(unique_hash_shape, unique_hash_shape.NumLayers() - 1);

      K2_CHECK_EQ(src_hash_shape.Dim0(), unique_hash_shape.Dim0());

      const int32_t *src_hash_row_splits = src_hash_shape.RowSplits(1).Data(),
                    *unique_hash_row_splits =
                        unique_hash_shape.RowSplits(1).Data();
      const int32_t *src_hash_data = hash_src.Data(),
                    *unique_hash_data = hash_unique.Data();

      for (int32_t r = 0; r < src_hash_shape.Dim0(); r++) {
        int32_t src_begin = src_hash_row_splits[r],
                src_end = src_hash_row_splits[r + 1],
                unique_begin = unique_hash_row_splits[r],
                unique_end = unique_hash_row_splits[r + 1];
        std::set<int32_t> src_set(src_hash_data + src_begin,
                                  src_hash_data + src_end),
            unique_set(unique_hash_data + unique_begin,
                       unique_hash_data + unique_end);
        EXPECT_EQ((src_set == unique_set), true);
      }
    }
  }
}

TEST(RaggedIntTest, TestCreateRagged2Int) {
  std::vector<std::vector<int32_t>> vecs{{7, 9}, {10, 12, 13}, {}};
  std::vector<int32_t> expected_values{7, 9, 10, 12, 13};
  std::vector<int32_t> expected_row_splits = {0, 2, 5, 5};
  Ragged<int32_t> r = CreateRagged2(vecs);
  EXPECT_EQ(r.Context()->GetDeviceType(), kCpu);
  CheckArrayData(r.RowSplits(1), expected_row_splits);
  EXPECT_EQ(r.NumAxes(), 2);
  CheckArrayData(r.values, expected_values);

  Ragged<int32_t> r2("[ [7 9] [10 12 13] [] ]");
  K2_CHECK(Equal(r, r2));
}

TEST(RaggedFloatTest, TestCreateRagged2Float) {
  std::vector<std::vector<float>> vecs{{1.2, 2.3}, {}, {3.4, 5.6}};
  std::vector<float> expected_values{1.2, 2.3, 3.4, 5.6};
  std::vector<int32_t> expected_row_splits = {0, 2, 2, 4};
  Ragged<float> r = CreateRagged2(vecs);

  Ragged<float> &r2 = r.Generic().Specialize<float>();

  EXPECT_EQ(r.Context()->GetDeviceType(), kCpu);
  EXPECT_EQ(r2.Context()->GetDeviceType(), kCpu);
  CheckArrayData(r.RowSplits(1), expected_row_splits);
  EXPECT_EQ(r.NumAxes(), 2);
  EXPECT_EQ(r2.NumAxes(), 2);
  CheckArrayData(r.values, expected_values);
}

template <typename T>
static void TestPadRagged() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    {
      Ragged<T> src(c, "[ [1 2] [3 4 3] [] [5 6 7 8] ]");
      T padding_value = 0;
      Array2<T> res = PadRagged(src, "constant", padding_value);
      Array1<T> dst = res.Flatten();
      std::vector<T> expected = {1, 2, 0, 0, 3, 4, 3, 0,
                                 0, 0, 0, 0, 5, 6, 7, 8};
      CheckArrayData(dst, expected);
    }
    {
      Ragged<T> src(c, "[ [1 2] [3 4 3] [] [5 6 7 8] ]");
      T padding_value = -1;
      Array2<T> res = PadRagged(src, "constant", padding_value);
      Array1<T> dst = res.Flatten();
      std::vector<T> expected = {1,  2,  -1, -1, 3, 4, 3, -1,
                                 -1, -1, -1, -1, 5, 6, 7, 8};
      CheckArrayData(dst, expected);
    }
    {
      Ragged<T> src(c, "[ [1 2] [3 4 3] [] [5 6 7 8] ]");
      T padding_value = 100;
      Array2<T> res = PadRagged(src, "replicate", padding_value);
      Array1<T> dst = res.Flatten();
      std::vector<T> expected = {1,   2,   2,   2,   3, 4, 3, 3,
                                 100, 100, 100, 100, 5, 6, 7, 8};
      CheckArrayData(dst, expected);
    }
    {
      Ragged<T> src(c, "[ [ ] [ ] [ ] ]");
      T padding_value = 100;
      Array2<T> res = PadRagged(src, "constant", padding_value);
      K2_CHECK_EQ(res.Dim0(), 3);
      K2_CHECK_EQ(res.Dim1(), 0);
    }
    {
      Ragged<T> src(c, "[ [ ] [ ] [ ] ]");
      T padding_value = 100;
      Array2<T> res = PadRagged(src, "replicate", padding_value);
      K2_CHECK_EQ(res.Dim0(), 3);
      K2_CHECK_EQ(res.Dim1(), 0);
    }
  }
}

TEST(RaggedTest, TestPadRagged) {
  TestPadRagged<int32_t>();
  TestPadRagged<double>();
}

template <typename T>
static void TestPruneRagged() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<T> src(c,
                  "[ [ [ 1.1 2.1 5.2 ] [ 1.0 5.1 ] [ 6.1 ] ] "
                  "  [ [ 1.2 ] [ 2.2 6.3 ] [ ] ] "
                  "  [ [ 1.3 4.4 ] [ 2.3 5.0 ] ] ]");

    T beam = 2.0;
    auto renumbering = PruneRagged(src, 0, beam, 2);
    // best_score=6.3, max scores for sublists are [6.1, 6.3, 5.0]
    // no sublist is pruned by beam, 5.0 is pruned by max-elems
    // keep : [ [ [ 1.1 2.1 5.2 ] [ 1.0 5.1 ] [ 6.1 ] ]
    //          [ [ 1.2 ] [ 2.2 6.3 ] [ ] ] ]
    Array1<char> keep_ref(c, std::vector<char>{1, 1, 0});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));

    beam = 0.1;
    renumbering = PruneRagged(src, 0, beam, 3);
    // best_score=6.3, max scores for sublists are [6.1, 6.3, 5.0]
    // 6.1 & 5.0 are pruned by beam
    // keep : [ [ [ 1.2 ] [ 2.2 6.3 ] [ ] ] ]
    keep_ref = Array1<char>(c, std::vector<char>{0, 1, 0});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));

    beam = 2.0;
    renumbering = PruneRagged(src, 1, beam, 2);
    // best_score=[6.1, 6.3, 5.0], max scores for sublists are
    // [[5.2, 5.1, 6.1], [1.2, 6.3, -inf], [4.4, 5.0]]
    // 1.2 & -inf are pruned by beam, 5.1 is pruned by max-elems.
    // keep : [ [ [ 1.1 2.1 5.2 ] [ 6.1 ] ] [ [ 2.2 6.3 ] ]
    //          [ [1.3 4.4] [ 2.3 5.0 ] ] ]
    keep_ref = Array1<char>(c, std::vector<char>{1, 0, 1, 0, 1, 0, 1, 1});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));

    beam = 0.5;
    renumbering = PruneRagged(src, 1, beam, 2);
    // best_score=[6.1, 6.3, 5.0], max scores for sublists are
    // [[5.2, 5.1, 6.1], [1.2, 6.3, -inf], [4.4, 5.0]]
    // all sublists are pruned by beam, except 6.1 & 6.3 & 5.0
    // keep : [ [ [ 6.1 ] ] [ [ 2.2 6.3 ] ] [ [ 2.3 5.0 ] ] ]
    keep_ref = Array1<char>(c, std::vector<char>{0, 0, 1, 0, 1, 0, 0, 1});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));

    beam = 4.0;
    renumbering = PruneRagged(src, 2, beam, 3);
    // best scores for sublists are
    // [5.2, 5.1, 6.1, 1.2, 6.3, -inf, 4.4, 5.0]
    // 1.1, 1.0, 2.2 are pruned by beam.
    // keep : [ [ [ 2.1 5.2 ] [ 5.1 ] [ 6.1 ] ] [ [ 1.2 ] [ 6.3 ] ]
    //          [ [ 1.3 4.4 ] [ 2.3 5.0 ] ] ]
    keep_ref = Array1<char>(
        c, std::vector<char>{0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));

    beam = 5.0;
    renumbering = PruneRagged(src, 2, beam, 2);
    // best scores for sublists are
    // [5.2, 5.1, 6.1, 1.2, 6.3, -inf, 4.4, 5.0]
    // 1.1 is pruned by max-elems.
    // keep : [ [ [ 2.1 5.2 ] [ 1.0 5.1 ] [ 6.1 ] ] [ [ 1.2 ] [ 2.2 6.3 ] ]
    //          [ [ 1.3 4.4 ] [ 2.3 5.0 ] ] ]
    keep_ref = Array1<char>(
        c, std::vector<char>{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    K2_CHECK(Equal(renumbering.Keep(), keep_ref));
  }
}

TEST(RaggedTest, TestPruneRagged) {
  TestPruneRagged<float>();
  TestPruneRagged<double>();
}

template <typename T>
static void TestPruneRaggedAndSubsetRagged() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<T> src(c,
                  "[ [ [ 1.1 4.2 2.1 1.8 ] [ 5.0 3.1 ] ] "
                  "  [ [ 1.2 ] [ 2.2 6.3 ] [ 2.4 6.1 ] [ 5.1 ] ] "
                  "  [ [ 1.3 4.4 ] [ 1.4 0.8 2.3 5.2 3.6 ] ] ]");
    T beam = 1.0;
    auto renumbering = PruneRagged(src, 0, beam, 3);
    Array1<int32_t> new2old;
    auto dest = SubsetRagged(src, renumbering, 0, &new2old);
    Ragged<T> dest_ref(c, "[ [ [ 1.2 ] [ 2.2 6.3 ] [ 2.4 6.1 ] [ 5.1 ] ] ]");
    Array1<int32_t> new2old_ref(c, std::vector<int32_t>{6, 7, 8, 9, 10, 11});
    K2_CHECK(Equal(dest, dest_ref));
    K2_CHECK(Equal(new2old, new2old_ref));

    beam = 1.0;
    renumbering = PruneRagged(src, 1, beam, 2);
    dest = SubsetRagged(src, renumbering, 1, &new2old);
    // [5.0, 6.3, 5.2]
    // [ [4.2 5.0] [1.2 6.3 6.1 5.1] [4.4 5.2]]
    dest_ref =
        Ragged<T>(c,
                  "[ [ [ 1.1 4.2 2.1 1.8] [ 5.0 3.1 ] ]"
                  "  [ [ 2.2 6.3 ] [ 2.4 6.1 ] ] "
                  "  [ [ 1.3 4.4 ] [ 1.4 0.8 2.3 5.2 3.6 ] ] ]");
    new2old_ref = Array1<int32_t>(
        c, std::vector<int32_t>{
        0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18});
    K2_CHECK(Equal(dest, dest_ref));
    K2_CHECK(Equal(new2old, new2old_ref));

    beam = 3.0;
    renumbering = PruneRagged(src, 2, beam, 3);
    dest = SubsetRagged(src, renumbering, 2, &new2old);
    dest_ref = Ragged<T>(
        c,
        "[ [ [ 4.2 2.1 1.8 ] [ 5.0 3.1 ] ] [ [ 1.2 ] [ 6.3 ] [ 6.1 ] [ 5.1 ] ]"
        "  [ [ 4.4 ] [ 2.3 5.2 3.6 ] ] ]");
    new2old_ref = Array1<int32_t>(
        c, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 16, 17, 18});
    K2_CHECK(Equal(dest, dest_ref));
    K2_CHECK(Equal(new2old, new2old_ref));
  }
}

TEST(RaggedTest, TestPruneRaggedAndSubsetRagged) {
  TestPruneRaggedAndSubsetRagged<float>();
  TestPruneRaggedAndSubsetRagged<double>();
}

}  // namespace k2
