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

#include <mutex>  // NOLINT
#include <utility>

#include "k2/csrc/context.h"
#include "k2/csrc/moderngpu_allocator.h"

namespace {

class ModernGpuAllocator : public mgpu::standard_context_t {
 public:
  explicit ModernGpuAllocator(k2::ContextPtr context)
      : mgpu::standard_context_t(false, context->GetCudaStream()),
        context_(std::move(context)) {}

  void *alloc(size_t size, mgpu::memory_space_t space) override {
    K2_DCHECK_EQ(space, mgpu::memory_space_device);
    void *deleter_ = nullptr;
    void *p = context_->Allocate(size, &deleter_);
    K2_DCHECK(deleter_ == nullptr);
    return p;
  }

  void free(void *p, mgpu::memory_space_t space) override {
    K2_DCHECK_EQ(space, mgpu::memory_space_device);
    context_->Deallocate(p, nullptr);
  }

 private:
  k2::ContextPtr context_;
};

}  // namespace

namespace k2 {

static mgpu::context_t *mgpu_contexts[kMaxNumGpus];
static std::once_flag mgpu_once_flags[kMaxNumGpus];

static void InitModernGpuAllocator(ContextPtr context) {
  int32_t device_index = context->GetDeviceId();
  K2_CHECK_GE(device_index, 0);
  K2_CHECK_LT(device_index, kMaxNumGpus);
  // it is never freed
  mgpu_contexts[device_index] = new ModernGpuAllocator(context);
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
