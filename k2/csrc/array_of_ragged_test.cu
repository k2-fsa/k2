/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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

#include "gtest/gtest.h"
#include "k2/csrc/array_of_ragged.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/ragged_utils.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

template <typename T>
void TestArray1OfRaggedConstruct() {
  int32_t num_srcs = 5;
  int32_t num_axes = 4;

  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    std::vector<Ragged<T>> raggeds;
    for (int32_t i = 0; i < num_srcs; ++i) {
      raggeds.emplace_back(
          RandomRagged<T>(0 /*min_value*/, 100 /*max_value*/,
                          num_axes /*min_num_axes*/, num_axes /*max_num_axes*/,
                          0 /*min_num_elements*/, 100 /*max_num_elements*/)
              .To(c, true /*copy_all*/));
    }
    auto array_of_ragged =
        Array1OfRagged<T>(raggeds.data(), num_srcs, true /*populate_meta*/);
    for (int32_t j = 1; j < num_axes; ++j) {
      const int32_t **row_splits = array_of_ragged.shape.RowSplits(j);
      const int32_t **row_ids = array_of_ragged.shape.RowIds(j);

      Array1<int32_t *> expected_row_splits(GetCpuContext(), num_srcs);
      Array1<int32_t *> expected_row_ids(GetCpuContext(), num_srcs);
      int32_t **expected_row_splits_data = expected_row_splits.Data();
      int32_t **expected_row_ids_data = expected_row_ids.Data();
      for (int32_t i = 0; i < num_srcs; ++i) {
        expected_row_splits_data[i] = raggeds[i].RowSplits(j).Data();
        expected_row_ids_data[i] = raggeds[i].RowIds(j).Data();
      }
      expected_row_splits = expected_row_splits.To(c);
      expected_row_ids = expected_row_ids.To(c);
      expected_row_splits_data = expected_row_splits.Data();
      expected_row_ids_data = expected_row_ids.Data();

      Array1<int32_t> flags(c, 2, 1);
      int32_t *flags_data = flags.Data();
      K2_EVAL(
          c, num_srcs, lambda_check_pointer, (int32_t i) {
            if (row_splits[i] != expected_row_splits_data[i]) flags_data[0] = 0;
            if (row_ids[i] != expected_row_ids_data[i]) flags_data[1] = 0;
          });
      K2_CHECK(Equal(flags, Array1<int32_t>(c, std::vector<int32_t>{1, 1})));
    }
    for (int32_t i = 0; i < num_srcs; ++i) {
      K2_CHECK_EQ(array_of_ragged.values[i], raggeds[i].values.Data());
    }

    for (int32_t j = 0; j < num_axes; ++j) {
      Array1<int32_t> meta_row_splits(array_of_ragged.shape.MetaRowSplits(j));
      Array1<int32_t> meta_row_ids(array_of_ragged.shape.MetaRowIds(j));
      Array1<int32_t> offsets(
          array_of_ragged.shape.Offsets().RowArange(j + 1, j + 2).Row(0));

      Array1<int32_t> expected_meta_row_splits(GetCpuContext(), num_srcs + 1);
      int32_t *expected_meta_row_splits_data = expected_meta_row_splits.Data();
      for (int32_t i = 0; i < num_srcs; ++i) {
        expected_meta_row_splits_data[i] = raggeds[i].TotSize(j);
      }
      ExclusiveSum(expected_meta_row_splits, &expected_meta_row_splits);
      expected_meta_row_splits = expected_meta_row_splits.To(c);
      Array1<int32_t> expected_meta_row_ids(c,
                                            array_of_ragged.shape.TotSize(j));
      RowSplitsToRowIds(expected_meta_row_splits, &expected_meta_row_ids);

      K2_CHECK(Equal(meta_row_splits, expected_meta_row_splits));
      K2_CHECK(Equal(meta_row_ids, expected_meta_row_ids));
      K2_CHECK(Equal(offsets, expected_meta_row_splits));
    }

    Array1<int32_t> expected_offsets_1st_row(GetCpuContext(), num_srcs + 1);
    int32_t *expected_offsets_1st_row_data = expected_offsets_1st_row.Data();
    for (int32_t i = 0; i <= num_srcs; ++i) {
      expected_offsets_1st_row_data[i] = i;
    }
    expected_offsets_1st_row = expected_offsets_1st_row.To(c);
    Array1<int32_t> offsets_1st_row(
        array_of_ragged.shape.Offsets().RowArange(0, 1).Row(0));
    K2_CHECK(Equal(offsets_1st_row, expected_offsets_1st_row));
  }
}

TEST(Array1OfRagged, Construct) {
  TestArray1OfRaggedConstruct<int32_t>();
  TestArray1OfRaggedConstruct<float>();
}

}  // namespace k2
