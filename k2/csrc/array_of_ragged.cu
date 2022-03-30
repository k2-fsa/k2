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

#include "k2/csrc/array_of_ragged.h"

namespace k2 {

Array1OfRaggedShape::Array1OfRaggedShape(RaggedShape *src, int32_t num_srcs)
    : num_srcs_(num_srcs) {
  K2_CHECK_GE(num_srcs, 1);
  K2_CHECK(src);
  num_axes_ = src[0].NumAxes();
  c_ = src[0].Context();

  row_splits_ =
      Array2<const int32_t *>(GetCpuContext(), num_axes_ - 1, num_srcs_);
  row_ids_ = Array2<const int32_t *>(GetCpuContext(), num_axes_ - 1, num_srcs_);
  tot_sizes_ = Array1<int32_t>(GetCpuContext(), num_axes_, 0);

  auto row_splits_acc = row_splits_.Accessor(),
       row_ids_acc = row_ids_.Accessor();
  int32_t *tot_sizes_data = tot_sizes_.Data();

  for (int32_t i = 0; i < num_srcs_; ++i) {
    K2_CHECK_EQ(src[i].NumAxes(), num_axes_);
    K2_CHECK(c_->IsCompatible(*(src[i].Context())));
    for (int32_t j = 1; j < num_axes_; ++j) {
      row_splits_acc(j - 1, i) = src[i].RowSplits(j).Data();
      row_ids_acc(j - 1, i) = src[i].RowIds(j).Data();
      tot_sizes_data[j] += src[i].TotSize(j);
    }
    tot_sizes_data[0] += src[i].TotSize(0);
  }

  row_splits_ = row_splits_.To(c_);
  row_ids_ = row_ids_.To(c_);
}

}  // namespace k2
