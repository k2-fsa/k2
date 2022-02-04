/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
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
      new2old = numbering.New2Old(true);
      EXPECT_EQ(new2old.Dim(), 1);
      EXPECT_EQ(new2old.Back(), 0);
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
      new2old = numbering.New2Old(true);
      EXPECT_EQ(new2old.Dim(), 1);
      EXPECT_EQ(new2old.Back(), 5);
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
      new2old = numbering.New2Old(true);
      EXPECT_EQ(new2old.Dim(), 5);
      EXPECT_EQ(new2old.Back(), 7);
    }
  }
}

void TestGetNew2OldAndRowIds() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i < 10; i++) {
      int32_t num_rows = RandInt(0, 2048);

      Array1<int32_t> elems_per_row = RandUniformArray1(c, num_rows + 1, 0, 15);
      Array1<int32_t> row_splits(c, num_rows + 1);
      ExclusiveSum(elems_per_row, &row_splits);
      Array1<int32_t> row_ids(c, row_splits.Back());
      RowSplitsToRowIds(row_splits, &row_ids);
      int32_t num_elems = row_ids.Dim();
      Array1<int32_t> random_keep = RandUniformArray1(c, num_elems, 0, 1);
      const int32_t *random_keep_data = random_keep.Data();

      Renumbering r(c, num_elems);
      char *keep_data = r.Keep().Data();
      K2_EVAL(c, num_elems, lambda_set_keep, (int32_t i) -> void {
          keep_data[i] = (char)random_keep_data[i];
        });

      auto lambda_keep = [=] __host__ __device__(int32_t i, int32_t row) {
        return random_keep_data[i];
      };

      Array1<int32_t> new2old,
          row_ids_subsampled;

      GetNew2OldAndRowIds(row_splits, num_elems, lambda_keep,
                          &new2old,
                          &row_ids_subsampled,
                          256);

      Array1<int32_t> new2old_ref = r.New2Old(),
           row_ids_subsampled_ref = row_ids[new2old_ref];
      K2_CHECK_EQ(Equal(new2old, new2old_ref), true);
      K2_CHECK_EQ(Equal(row_ids_subsampled, row_ids_subsampled_ref), true);
    }
  }
}

TEST(AlgorithmsTest, TestGetNew2OldAndRowIds) {
  TestGetNew2OldAndRowIds();
}


}  // namespace k2
