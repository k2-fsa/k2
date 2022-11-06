/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/torch_util.h"
#include "k2/torch/csrc/torch_api.h"

namespace k2 {

void ExclusiveSum(torch::Tensor src, torch::Tensor *dst) {
  Array1<int32_t> src_arr = FromTorch<int32_t>(src);
  Array1<int32_t> dst_arr = FromTorch<int32_t>(*dst);

  ExclusiveSum(src_arr, &dst_arr);
}

RaggedShapePtr RaggedShape2(torch::Tensor row_splits, torch::Tensor row_ids,
                            int32_t cached_tot_size /*=-1*/) {
  if (!row_splits.defined()) {
    K2_CHECK(row_ids.defined())
        << "You have to provide row_ids if row_splits is empty";
  }

  Array1<int32_t> row_splits_arr, row_ids_arr;

  if (row_splits.defined()) {
    row_splits_arr = FromTorch<int32_t>(row_splits);
  }

  if (row_ids.defined()) {
    row_ids_arr = FromTorch<int32_t>(row_ids);
  }

  return std::make_shared<RaggedShape>(RaggedShape2(
      row_splits.defined() ? &row_splits_arr : nullptr,
      row_ids.defined() ? &row_ids_arr : nullptr, cached_tot_size));
}

int32_t TotSize(RaggedShapePtr shape, int32_t axis) {
  return shape->TotSize(axis);
}

torch::Tensor RowIds(RaggedShapePtr shape, int32_t axis) {
  return ToTorch(shape->RowIds(axis));
}

torch::Tensor RowSplits(RaggedShapePtr shape, int32_t axis) {
  return ToTorch(shape->RowSplits(axis));
}

}  // namespace k2
