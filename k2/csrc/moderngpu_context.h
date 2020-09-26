/**
 * @brief This is a Context for moderngpu only.
 *
 * Currently it is used by `SortSublists`.
 *
 * @copyright
 * Copyright (c)  2020  AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_MODERNGPU_CONTEXT_H_
#define K2_CSRC_MODERNGPU_CONTEXT_H_

#include <memory>

#include "moderngpu/context.hxx"

namespace k2 {
// Return a context for moderngpu that has a better memory allocator
// than mgpu::standard_context_t
std::unique_ptr<mgpu::context_t> GetModernGpuContext(int32_t device_id = -1);

}  // namespace k2

#endif  // K2_CSRC_MODERNGPU_CONTEXT_H_
