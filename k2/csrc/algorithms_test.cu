/**
 * @brief
 * algorithms_test
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
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"

namespace k2 {
TEST(AlgorithmsTest, TestRenumbering) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      int32_t num_old_elems = 0;
      Renumbering numbering(context, num_old_elems);
      EXPECT_EQ(numbering.NumOldElems(), num_old_elems);
      Array1<char> &keep = numbering.Keep();
      EXPECT_EQ(keep.Dim(), num_old_elems);
      Array1<int32_t> old2new = numbering.Old2New();
      EXPECT_EQ(old2new.Dim(), num_old_elems);
      Array1<int32_t> new2old = numbering.New2Old();
      EXPECT_EQ(new2old.Dim(), 0);
      EXPECT_EQ(numbering.NumNewElems(), 0);
    }

    {
      // num_old_elems != 0 but num_new_elems = 0
      int32_t num_old_elems = 5;
      Renumbering numbering(context, num_old_elems);
      EXPECT_EQ(numbering.NumOldElems(), num_old_elems);
      Array1<char> &keep = numbering.Keep();
      EXPECT_EQ(keep.Dim(), num_old_elems);
      std::vector<char> keep_data(num_old_elems, 0);
      Array1<char> keep_array(context, keep_data);
      keep.CopyFrom(keep_array);
      Array1<int32_t> old2new = numbering.Old2New();
      old2new = old2new.To(cpu);
      EXPECT_EQ(old2new.Dim(), num_old_elems);

      std::vector<int32_t> cpu_old2new(old2new.Data(),
                                       old2new.Data() + old2new.Dim());
      EXPECT_THAT(cpu_old2new, ::testing::ElementsAre(0, 0, 0, 0, 0));
      Array1<int32_t> new2old = numbering.New2Old();
      EXPECT_EQ(new2old.Dim(), 0);
      EXPECT_EQ(numbering.NumNewElems(), 0);
    }

    {
      // normal case
      int32_t num_old_elems = 7;
      Renumbering numbering(context, num_old_elems);
      EXPECT_EQ(numbering.NumOldElems(), num_old_elems);
      Array1<char> &keep = numbering.Keep();
      EXPECT_EQ(keep.Dim(), num_old_elems);
      std::vector<char> keep_data = {1, 0, 1, 1, 0, 0, 1};
      ASSERT_EQ(keep_data.size(), num_old_elems);
      Array1<char> keep_array(context, keep_data);
      keep.CopyFrom(keep_array);
      Array1<int32_t> old2new = numbering.Old2New();
      old2new = old2new.To(cpu);
      EXPECT_EQ(old2new.Dim(), num_old_elems);
      std::vector<int32_t> cpu_old2new(old2new.Data(),
                                       old2new.Data() + old2new.Dim());
      EXPECT_THAT(cpu_old2new, ::testing::ElementsAre(0, 1, 1, 2, 3, 3, 3));
      Array1<int32_t> new2old = numbering.New2Old();
      EXPECT_EQ(new2old.Dim(), 4);
      EXPECT_EQ(numbering.NumNewElems(), 4);
      new2old = new2old.To(cpu);
      std::vector<int32_t> cpu_new2old(new2old.Data(),
                                       new2old.Data() + new2old.Dim());
      EXPECT_THAT(cpu_new2old, ::testing::ElementsAre(0, 2, 3, 6));
    }
  }
}

}  // namespace k2
