/**
 * @brief
 * ragged_tensor_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_TENSOR_OPS_H_
#define K2_CSRC_RAGGED_TENSOR_OPS_H_

#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/utils.h"

namespace k2 {
// This file declares ops involving RaggedShape, Ragged<int32_t>,
// and Tensor.  They are implemented in ragged_tensor_ops.cu
// (they don't need to be in the header as Tensor doesn't have type
// information, so these functions are not templated).







}  // namespace k2


#endif  // K2_CSRC_RAGGED_TENSOR_OPS_H_
