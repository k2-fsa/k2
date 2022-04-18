/**
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
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

Array1OfRaggedShape::Array1OfRaggedShape(RaggedShape *srcs, int32_t num_srcs,
                                         bool populate_meta)
    : num_srcs_(num_srcs), populate_meta_(populate_meta) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK(srcs);

  // Initialize context and num_axes_.
  c_ = srcs[0].Context();
  num_axes_ = srcs[0].NumAxes();

  // Check if they have same num-axes and compatible context.
  for (int32_t i = 1; i < num_srcs_; ++i) {
    K2_CHECK_EQ(num_axes_, srcs[i].NumAxes());
    K2_CHECK(c_->IsCompatible(*(srcs[i].Context())));
  }

  // Initialize row_splits__, row_ids_ and tot_sizes_.
  //
  // Notice: since the Data() function is a __host__ function, it cannot be
  // called on GPU. It limits us to work on CPU so that the row_splits_ and
  // row_ids_ are populated on CPU, although the operator() of Array2 is a
  // __host__ and __device__ function. Bear in mind, we cannot access the
  // GPU data on CPU.
  row_splits_ =
      Array2<const int32_t *>(GetCpuContext(), num_axes_ - 1, num_srcs_);
  row_ids_ = Array2<const int32_t *>(GetCpuContext(), num_axes_ - 1, num_srcs_);

  // Notice: no matter the return value of TotSize() is from 'cached_tot_size'
  //  or the Back() function (i.e. operator[]) of array1, it it a CPU value.
  tot_sizes_ = Array1<int32_t>(GetCpuContext(), num_axes_, 0);

  auto row_splits_acc = row_splits_.Accessor(),
       row_ids_acc = row_ids_.Accessor();
  // Bear in mind, when axis == 0, the TotSize() is row_splits.Dim() - 1.
  // When 0 < axis < NumAxes(), the TotSize() is row_splits.Back().
  int32_t *tot_sizes_data = tot_sizes_.Data();

  for (int32_t i = 0; i < num_srcs_; ++i) {
    for (int32_t j = 1; j < num_axes_; ++j) {
      row_splits_acc(j - 1, i) = srcs[i].RowSplits(j).Data();
      row_ids_acc(j - 1, i) = srcs[i].RowIds(j).Data();
      tot_sizes_data[j] += srcs[i].TotSize(j);
    }
    tot_sizes_data[0] += srcs[i].TotSize(0);
  }

  row_splits_ = row_splits_.To(c_);
  row_ids_ = row_ids_.To(c_);

  if (populate_meta_) {
    // Initialize meta_row_splits_
    // We populate this on CPU and transfer to GPU.
    meta_row_splits_ =
        Array2<int32_t>(GetCpuContext(), num_axes_, num_srcs_ + 1);
    offsets_ = Array2<int32_t>(GetCpuContext(), num_axes_ + 1, num_srcs_ + 1);

    auto meta_row_splits_acc = meta_row_splits_.Accessor(),
         offsets_acc = offsets_.Accessor();

    // Initialize the 1st row of offsets_, which contains 0,1,2,...
    for (int32_t col = 0; col <= num_srcs_; ++col) {
      offsets_acc(0, col) = col;
    }
    // Initialize the 1st col of meta_row_splits_ and offsets_
    for (int32_t row = 0; row < num_axes_; ++row) {
      meta_row_splits_acc(row, 0) = 0;
      offsets_acc(row + 1, 0) = 0;
    }

    // The meta_row_splits_ is the cumulative sum of the tot-sizes of the
    // individual arrays.
    for (int32_t i = 0; i < num_axes_; ++i) {
      for (int32_t j = 1; j <= num_srcs_; ++j) {
        meta_row_splits_acc(i, j) =
            meta_row_splits_acc(i, j - 1) + srcs[j - 1].TotSize(i);
        offsets_acc(i + 1, j) = meta_row_splits_acc(i, j);
      }
    }

    // Initialize meta_row_ids_
    // Elements are in [0, NumSrcs() - 1]
    meta_row_ids_.resize(num_axes_);

    for (int32_t axis = 0; axis < num_axes_; ++axis) {
      // The length equals to TotSize(axis)
      meta_row_ids_.at(axis) = Array1<int32_t>(
          GetCpuContext(), meta_row_splits_acc(axis, num_srcs_));
      int32_t *meta_row_ids_data = meta_row_ids_[axis].Data();

      int32_t cur_row_start = meta_row_splits_acc(axis, 0);
      for (int32_t src = 0; src < num_srcs_; ++src) {
        int32_t next_row_start = meta_row_splits_acc(axis, src + 1);
        for (; cur_row_start < next_row_start; ++cur_row_start) {
          meta_row_ids_data[cur_row_start] = src;
        }
      }
      meta_row_ids_[axis] = meta_row_ids_[axis].To(c_);
    }

    meta_row_splits_ = meta_row_splits_.To(c_);
    offsets_ = offsets_.To(c_);
  }
}

}  // namespace k2
