/**
 * @brief This is an allocator for moderngpu only.
 *
 * Currently it is used by `SortSublists`.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_MODERNGPU_ALLOCATOR_H_
#define K2_CSRC_MODERNGPU_ALLOCATOR_H_

#include <memory>

#include "k2/csrc/context.h"
#include "moderngpu/context.hxx"

namespace k2 {
// Return a context for moderngpu that is a warpper of the given CUDA allocator
// than mgpu::standard_context_t
std::unique_ptr<mgpu::context_t> GetModernGpuAllocator(ContextPtr c);

}  // namespace k2

#endif  // K2_CSRC_MODERNGPU_ALLOCATOR_H_
