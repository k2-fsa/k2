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
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor.h"

namespace {
static void CheckRowSplitsOrIds(k2::RaggedShape &shape,
                                const std::vector<std::vector<int32_t>> &target,
                                bool row_splits) {
  for (int32_t i = 1; i < shape.NumAxes(); ++i) {
    k2::Array1<int32_t> curr_row_splits =
        row_splits ? shape.RowSplits(i) : shape.RowIds(i);
    // copy data from CPU/GPU to CPU
    std::vector<int32_t> cpu_data(curr_row_splits.Dim());
    curr_row_splits.Context()->CopyDataTo(
        curr_row_splits.Dim() * curr_row_splits.ElementSize(),
        curr_row_splits.Data(), k2::GetCpuContext(), cpu_data.data());

    EXPECT_EQ(cpu_data, target[i - 1]);
  }
}
}  // namespace

namespace k2 {
TEST(RaggedShapeTest, RaggedShape) {
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
      if (context->GetDeviceType() == kCpu) {
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

    {
      // constructed with row_splits
      // RaggedTensor4 t = [
      //  [ [[ 1, 2], [4]],  [[3, 0]] ],
      //  [ [[7, 8, 9]], [[6], [3, 5, 7]], [[2]] ],
      //  [ [[3, 4], [], [8]] ]
      // ]
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
      const std::vector<int32_t> row_splits3 = {0,  2,  3,  5,  8, 9,
                                                12, 13, 15, 15, 16};
      Array1<int32_t> row_ids;  // invalid row_ids as it has no context,
                                // shape.RowIds(axis) will create it.
      std::vector<RaggedShapeLayer> axes;
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits1), row_ids, -1});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits2), row_ids, -1});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits3), row_ids, -1});
      RaggedShape shape(axes, true);

      EXPECT_EQ(shape.NumAxes(), 4);
      EXPECT_EQ(shape.Dim0(), 3);
      EXPECT_EQ(shape.NumElements(), row_splits3.back());

      // test RowIds()
      const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
      const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
      const std::vector<int32_t> row_ids3 = {0, 0, 1, 2, 2, 3, 3, 3,
                                             4, 5, 5, 5, 6, 7, 7, 9};
      const std::vector<std::vector<int32_t>> row_ids_vec = {row_ids1, row_ids2,
                                                             row_ids3};
      CheckRowSplitsOrIds(shape, row_ids_vec, false);
    }

    {
      // constructed with row_splits and test Populate()
      // RaggedTensor4 t = [
      //  [ [[ 1, 2], [4]],  [[3, 0]] ],
      //  [ [[7, 8, 9]], [[6], [3, 5, 7]], [[2]] ],
      //  [ [[3, 4], [], [8]] ]
      // ]
      const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
      const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
      const std::vector<int32_t> row_splits3 = {0,  2,  3,  5,  8, 9,
                                                12, 13, 15, 15, 16};
      Array1<int32_t> row_ids;  // invalid row_ids as it has no context,
                                // shape.RowIds(axis) will create it.
      std::vector<RaggedShapeLayer> axes;
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits1), row_ids, -1});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits2), row_ids, -1});
      axes.emplace_back(
          RaggedShapeLayer{Array1<int32_t>(context, row_splits3), row_ids, -1});
      RaggedShape shape(axes, true);

      // test Populate(), it will create row_ids and cached_tot_size from
      // row_splits
      shape.Populate();

      const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
      const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
      const std::vector<int32_t> row_ids3 = {0, 0, 1, 2, 2, 3, 3, 3,
                                             4, 5, 5, 5, 6, 7, 7, 9};
      const std::vector<std::vector<int32_t>> row_ids_vec = {row_ids1, row_ids2,
                                                             row_ids3};

      const auto &curr_axes = shape.Layers();
      for (int32_t i = 1; i < shape.NumAxes(); ++i) {
        const Array1<int32_t> &curr_row_ids = curr_axes[i - 1].row_ids;
        // copy data from CPU/GPU to CPU
        std::vector<int32_t> cpu_data(curr_row_ids.Dim());
        curr_row_ids.Context()->CopyDataTo(
            curr_row_ids.Dim() * curr_row_ids.ElementSize(),
            curr_row_ids.Data(), cpu, cpu_data.data());
        EXPECT_EQ(cpu_data, row_ids_vec[i - 1]);
        EXPECT_EQ(curr_axes[i - 1].cached_tot_size, row_ids_vec[i - 1].size());
      }
    }
  }
}


TEST(RaggedShapeTest, DecomposeRaggedShape) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    {
      RaggedShape s(c, "[ [ x x ] [ x ] ]"),
          t(c, "[ [ x ] [ x x ] [ x x x x ] ]"),
          u = ComposeRaggedShapes(s, t);

      RaggedShape s2, t2;
      DecomposeRaggedShape(u, 1, &s2, &t2);
      EXPECT_EQ(Equal(s, s2), true);
      EXPECT_EQ(Equal(t, t2), true);
    }
  }
}



TEST(RaggedShapeTest, RemoveEmptyListsAxis0) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    {
      RaggedShape s(c, "[ [ [ x ] [x ] ] [ ] [ [ x ] ] [ ]  ]"),
          t(c, "[ [ [ x ] [x ] ] [ [ x ] ] ]");

      Renumbering r;
      RaggedShape t2 = RemoveEmptyListsAxis0(s, &r);
      EXPECT_EQ(Equal(t, t2), true);
    }

    {
      RaggedShape s(c, "[ [ x x ] [ ] [ x ] [ ]  ]"),
          t(c, "[ [ x x ] [ x ] ]");
      RaggedShape t2 = RemoveEmptyListsAxis0(s);
      EXPECT_EQ(Equal(t, t2), true);
    }

    {
      RaggedShape s(c, "[ [ x x ] [ ] [ x ] [ ]  ]"),
          t(c, "[ [ x x ] [ x ] ]");
      Renumbering r;
      RaggedShape t2 = RemoveEmptyLists(s, 0, &r);
      EXPECT_EQ(Equal(t, t2), true);
    }


    {
      RaggedShape s(c, "[ [ x x ] [ ] [ x ] [ ]  ]"),
          t(c, "[ [ x x ] [ ] [ x ] ]");

      Array1<char> keep(c, std::vector<char>({ (char)1, (char)1, (char)1, (char)0 }));
      Renumbering r(c, 4);
      Assign(keep, &r.Keep());
      RaggedShape t2 = RenumberAxis0Simple(s, r);
      EXPECT_EQ(Equal(t, t2), true);
    }

    {
      RaggedShape s(c, "[ [ x x ] [ ] [ x ] [ ]  ]"),
          t(c, "[ [ x x ] [ ] [ x ] ]");

      Array1<char> keep(c, std::vector<char>({ (char)0, (char)1, (char)1, (char)0 }));
      Renumbering r(c, 4);
      Assign(keep, &r.Keep());
#ifndef NDEBUG
      ASSERT_DEATH(RenumberAxis0Simple(s, r), "");
#endif
    }
  }
}


TEST(RaggedShapeTest, RemoveEmptyLists) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    {
      RaggedShape s(c, "[ [ [ x ] [ ] ]  [ ] [ [ x ] ] [ ]  ]"),
          t(c, "[ [ [ x ] ]  [ ] [ [ x ] ] [ ]  ]");

      Renumbering r;
      RaggedShape t2 = RemoveEmptyLists(s, 1, &r);
      EXPECT_EQ(Equal(t, t2), true);
    }
  }
}






TEST(RaggedShapeTest, RaggedShapeIterator) {
  // note RaggedShapeIndexIterator works only for CPU
  ContextPtr context = GetCpuContext();
  // constructed with row_splits
  // RaggedTensor4 t = [
  //  [ [[ 1, 2], [4]],  [[3, 0]] ],
  //  [ [[7, 8, 9]], [[6], [3, 5, 7]], [[2]] ],
  //  [ [[3, 4], [], [8]] ]
  // ]
  const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
  const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
  const std::vector<int32_t> row_splits3 = {0,  2,  3,  5,  8, 9,
                                            12, 13, 15, 15, 16};
  Array1<int32_t> row_ids;  // invalid row_ids as it has no context,
                            // shape.RowIds(axis) will create it.
  std::vector<RaggedShapeLayer> axes;
  axes.emplace_back(
      RaggedShapeLayer{Array1<int32_t>(context, row_splits1), row_ids, -1});
  axes.emplace_back(
      RaggedShapeLayer{Array1<int32_t>(context, row_splits2), row_ids, -1});
  axes.emplace_back(
      RaggedShapeLayer{Array1<int32_t>(context, row_splits3), row_ids, -1});
  RaggedShape shape(axes, true);

  int32_t index = 0;
  for (RaggedShapeIndexIterator iter = shape.Iterator(); !iter.Done();
       iter.Next()) {
    const std::vector<int32_t> &vec = iter.Value();
    int32_t linear_index = shape[vec];
    EXPECT_EQ(linear_index, index);
    ++index;
  }
  EXPECT_EQ(index, row_splits3.back());
}

TEST(RaggedShapeTest, RandomRaggedShape) {
  {
    RaggedShape shape = RandomRaggedShape(false, 2, 4, 0, 0);
    EXPECT_GE(shape.NumAxes(), 2);
    EXPECT_EQ(shape.NumElements(), 0);
  }
  {
    RaggedShape shape = RandomRaggedShape();
    EXPECT_GE(shape.NumAxes(), 2);
    EXPECT_GE(shape.NumElements(), 0);
  }
  {
    RaggedShape shape = RandomRaggedShape(false, 3, 5, 100);
    EXPECT_GE(shape.NumAxes(), 3);
    EXPECT_GE(shape.NumElements(), 100);
  }
  {
    RaggedShape shape = RandomRaggedShape(true, 3, 5, 100);
    EXPECT_GE(shape.NumAxes(), 3);
    EXPECT_GE(shape.NumElements(), 100);
    const auto &axes = shape.Layers();
    EXPECT_GE(axes.back().row_ids.Dim(), 100);
  }
}

}  // namespace k2
