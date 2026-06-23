/**
 * Copyright (c) 2026  Advanced Micro Devices, Inc. (authors: Jeff Daily <jeff.daily@amd.com>)
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

// HIP replacement for moderngpu_allocator.cu. The HIP mgpu shim
// (moderngpu_shim.h) allocates and launches directly through the k2 Context, so
// the moderngpu allocator subclass is unnecessary; GetModernGpuAllocator just
// hands back a per-device shim context_t wrapping the k2 ContextPtr.

#include <mutex>  // NOLINT
#include <utility>

#include "k2/csrc/context.h"
#include "k2/csrc/moderngpu_allocator.h"

namespace k2 {

static mgpu::context_t *mgpu_contexts[kMaxNumGpus];
static std::once_flag mgpu_once_flags[kMaxNumGpus];

static void InitModernGpuAllocator(ContextPtr context) {
  int32_t device_index = context->GetDeviceId();
  K2_CHECK_GE(device_index, 0);
  K2_CHECK_LT(device_index, kMaxNumGpus);
  // It is never freed (same lifetime policy as the CUDA build).
  mgpu_contexts[device_index] = new mgpu::context_t(context);
}

mgpu::context_t *GetModernGpuAllocator(ContextPtr context) {
  K2_CHECK_EQ(context->GetDeviceType(), kCuda);

  int32_t device_index = context->GetDeviceId();
  K2_CHECK_GE(device_index, 0);
  K2_CHECK_LT(device_index, kMaxNumGpus);

  std::call_once(mgpu_once_flags[device_index], InitModernGpuAllocator,
                 context);

  return mgpu_contexts[device_index];
}

}  // namespace k2
