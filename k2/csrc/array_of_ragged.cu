/**
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey)
 *                2022  ASLP@NWPU          (authors: Hang Lyu)
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

#include "k2/csrc/array_of_ragged.h"

namespace k2 {

ArrayOfRaggedShape::ArrayOfRaggedShape(RaggedShape *srcs, int32_t num_srcs) :
  num_srcs_(num_srcs) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK(srcs);

  // Initialize context and num_axes_.
  c_ = srcs[0].Context();
  num_axes_ = srcs[0].NumAxes();

  // Check if they have same num-axes and compatible context.
  for (int32_t i = 1; i < num_srcs_; ++i) {
    K2_CHECK_EQ(num_axes_, srcs[i].NumAxes());
    K2_CHECK(IsCompatible(c_, srcs[i].Context()));
  }

  // Initialize row_splits, row_ids_ and tot_sizes_.
  row_splits_ = Array2<int32_t *>(c_, num_axes_ - 1, num_srcs_);
  row_ids_ = Array2<int32_t *>(c_, num_axes_ - 1, num_srcs_);
  tot_sizes_ = Array1<int32_t>(c_, num_axes_, 0);

  Array2Accessor<int32_t *> row_splits_acc = row_splits_.Accessor(),
                            row_ids_acc = row_ids_.Accessor();
  // Bear in mind, when axis == 0, the TotSize() is row_splits.Dim() - 1.
  // When 0 < axis < NumAxes(), the TotSize() is row_splits.Back().
  int32_t tot_sizes_data = tot_sizes_.Data();

  for (int32_t i = 1; i < num_axes_; ++i) {
    for (int32_t j = 0; j < num_srcs_; ++j) {
      row_splits_acc(i - 1, j) = srcs[j].RowSplits(i).Data();
      row_ids_acc(i - 1, j) = srcs[j].RowIds(i).Data();
      tot_sizes_[i] += srcs[j].TotSize(i);
    }
  }
  // Deal with the special axis == 0.
  for (int32_t i = 0; i < num_srcs_; ++i) {
    tot_sizes_[0] += srcs[i].TotSize(0);
  }

  // Initialize meat_row_splits_
  // We populate this on CPU and transfer to GPU.
  meta_row_splits_ = Array2<int32_t>(GetCpuContext(), num_axes_, num_srcs_ + 1);
  offsets_ = Array2<int32_t>(GetCpuContext(), num_axes_ + 1, num_srcs_ + 1);

  Array2Accessor<int32_t> meta_row_splits_acc = meta_row_splits_.Accessor(),
                          offsets_acc = offsets_.Accessor();
  // Initialize the 1st row/col of offsets_ and meta_row_splits_
  for (int32_t col = 0; col <= num_srcs_; ++col) {
    offsets_acc(0, col) = col;
  }
  for (int32_t row = 1; row <= num_axes_; ++ row) {
    offsets_acc(row, 0) = 0;
    meta_row_splits_acc(row, 0) = 0;
  }
  // The meta_row_splits_ is the cumulative sum of the tot-sizes of the
  // individual arrays.
  for (int32_t i = 0; i < num_axes_; ++i) {
    for (int32_t j = 1; j <= num_srcs_; ++j) {
      meta_row_splits_acc(i, j) = meta_row_splits_acc(i, j - 1) +
                                  srcs[j - 1].TotSize(i);
      offsets_acc(i + 1, j) = meta_row_splits_acc(i, j);
    }
  }

  meta_row_splits_ = meta_row_splits_.To(c_);
  offsets_ = offsets_.To(c_);

  // Initialize meta_row_ids_
  // Elements are in [0, NumSrcs() - 1]
  meta_row_ids_.resize(num_axes_);
  for (int32_t axis = 0; axis < num_axes_; ++axis) {
    // The length equals to TotSize(axis)
    meta_row_ids_.at(axis) = Array1<int32_t>(
        GetCpuContext(), meta_row_splits_acc(axis, num_srcs_));
    int32_t meta_row_ids_data = meta_row_ids_[axis].Data();

    int32_t cur_row_start = meta_row_splits_acc(axis, 0);
    for (int32_t src = 0; src < num_srcs_; ++src) {
      int32_t next_row_start = meta_row_splits_acc(axis, src + 1);
      for (; cur_row_start < next_row_start; ++cur_row_start) {
        meta_row_ids_data[cur_row_start] = src;
      }
    }
    meta_row_ids_[axis].To(c_);
  }
}

ArrayOfRagged::ArrayOfRagged(Ragged<T> *srcs, int32_t num_srcs) :
  values(srcs->Context(), num_srcs, nullptr) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK(srcs);

  T **values_data = values.Data();
  std::array<RaggedShape, num_srcs> shapes;

  for (int32_t i = 0; i < num_srcs; i++) {
    // Initialize values
    values_data[i] = srcs[i].values.Data();
    shapes[i] = srcs[i].shape;
  }

  // Initialize shape
  shape = ArrayOfRaggedShape(shapes.data(), num_srcs);
}

}  // namespeace k2
