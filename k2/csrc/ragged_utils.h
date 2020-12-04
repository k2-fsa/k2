/**
 * @brief
 * ragged_utils
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_UTILS_H_
#define K2_CSRC_RAGGED_UTILS_H_

#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

// ragged_utils.h is intended for operations that are somewhat internal to the
// ragged implementation, and not user-facing-- for example, operations that are
// limited to a RaggedShape with NumAxes() == 2 (i.e. a single ragged axis).
namespace k2 {


}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_UTILS_H_
#include "k2/csrc/ragged_utils_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_UTILS_H_

#endif  // K2_CSRC_RAGGED_UTILS_H_
