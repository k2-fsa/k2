/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_MODERNGPU_ALLOCATOR_H_
#define K2_CSRC_MODERNGPU_ALLOCATOR_H_

#include <memory>

#include "k2/csrc/context.h"
#include "k2/csrc/moderngpu.h"

namespace k2 {

/* Return an allocator for moderngpu.

   Caution: The returned pointer is NOT owned by the caller and it
   should NOT be freed!

   @param  [in]  context  It is a CUDA context that will be used to
                          allocate device memory for moderngpu.

   @return  Return a pointer to mgpu::context_t. The user should NOT
            free it.
 */
mgpu::context_t *GetModernGpuAllocator(ContextPtr context);

}  // namespace k2

#endif  // K2_CSRC_MODERNGPU_ALLOCATOR_H_
