/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "torch/extension.h"

namespace k2 {

torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index) {
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<int32_t>::value)

  torch::Tensor ans;
  return ans;
}

}  // namespace k2
