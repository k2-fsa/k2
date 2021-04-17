/**
 * Copyright (c)  2021  xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 */

#ifndef K2_CSRC_CUB_H_
#define K2_CSRC_CUB_H_

// See
// https://github.com/k2-fsa/k2/issues/698
// and
// https://github.com/pytorch/pytorch/issues/54245#issuecomment-805707551
// for why we need the following two macros
//
// NOTE: We define the following two macros so
// that k2 and PyTorch use a different copy
// of CUB.

#define CUB_NS_PREFIX namespace k2 {
#define CUB_NS_POSTFIX }

#include "cub/cub.cuh"  // NOLINT

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#endif  // K2_CSRC_CUB_H_
