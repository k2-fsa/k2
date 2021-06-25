/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
