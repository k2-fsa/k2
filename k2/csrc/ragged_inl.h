/**
 *
 * This is to be included only from ragged_ops.h.
 *
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_RAGGED_INL_H_
#define K2_CSRC_RAGGED_INL_H_

#ifndef IS_IN_K2_CSRC_RAGGED_H_
#error "this file is supposed to be included only by ragged.h"
#endif


namespace k2 {

template<int MAX_LAYERS>
RowSplitsAccessor<MAX_LAYERS>::RowSplitsAccessor(RaggedShape &src) {
  int32_t num_layers = src.NumLayers();
  K2_CHECK_LE(src.NumLayers(), MAX_LAYERS);
  for (int i = 0; i < num_layers; i++)
    ptrs[i] = src.RowSplits(i + 1).Data();
}

template<int MAX_LAYERS>
RowIdsAccessor<MAX_LAYERS>::RowIdsAccessor(RaggedShape &src) {
  int32_t num_layers = src.NumLayers();
  K2_CHECK_LE(src.NumLayers(), MAX_LAYERS);
  for (int i = 0; i < num_layers; i++)
    ptrs[i] = src.RowIds(i + 1).Data();
}


}  // namespace k2

#endif  // K2_CSRC_RAGGED_INL_H_
