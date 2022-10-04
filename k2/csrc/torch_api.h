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

#ifndef K2_CSRC_TORCH_API_H_
#define K2_CSRC_TORCH_API_H_

#include <memory>

#include "torch/script.h"

namespace k2 {

class RaggedShape;
using RaggedShapePtr = std::shared_ptr<RaggedShape>;

/** Compute the exclusive sum of "src".
 *
 * @param src A 1-D tensor of dtype torch.int32.
 * @param dst A 1-D tensor of dtype torch.int32 that should have the same
 *            number of elements as src. On return, dst[0] is always 0.
 *            dst[i] = sum_{j=0}^{i-1} src[j] for i > 0
 *            Note: src and dst can share the same address.
 */
void ExclusiveSum(torch::Tensor src, torch::Tensor *dst);

/** Create a ragged shape by specifying its row_splits and row_ids.
 *
 * Note: You have to provide at least one of them.
 *
 * @param row_splits If not empty, it is a 1-D tensor with dtype torch.int32.
 * @param row_ids If not empty, it is a 1-D tensor with dtype torch.int32
 * @param cached_tot_size If not -1, it contains the total number of elements
 *                        in the shape.
 *
 * @return Return a ragged shape with 2 axes.
 */
RaggedShapePtr RaggedShape2(torch::Tensor row_splits, torch::Tensor row_ids,
                            int32_t cached_tot_size = -1);

/** Return shape->TotSize(axis);
 *
 * Refer to the help information of RaggedShape::TotSize().
 */
int32_t TotSize(RaggedShapePtr shape, int32_t axis);

/** Return shape->RowIds(axis);
 *
 * Refer to the help information of RaggedShape::RowIds().
 */
torch::Tensor RowIds(RaggedShapePtr shape, int32_t axis);

/** Return shape->RowSplits(axis);
 *
 * Refer to the help information of RaggedShape::RowSplits().
 */
torch::Tensor RowSplits(RaggedShapePtr shape, int32_t axis);

}  // namespace k2

#endif  // K2_CSRC_TORCH_API_H_
