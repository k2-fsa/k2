/**
 *
 * This is to be included only from ragged_ops.h.
 *
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_INL_H_
#define K2_CSRC_RAGGED_INL_H_

#ifndef IS_IN_K2_CSRC_RAGGED_H_
#error "this file is supposed to be included only by ragged_ops.h"
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
