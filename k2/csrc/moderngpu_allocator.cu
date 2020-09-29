/**
 * @brief A better memory allocator for moderngpu.
 *
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <utility>

#include "k2/csrc/context.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "moderngpu/context.hxx"

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

std::unique_ptr<mgpu::context_t> GetModernGpuAllocator(
    int32_t device_id /*= -1*/) {
  return std::make_unique<ModernGpuAllocator>(GetCudaContext(device_id));
}

}  // namespace k2
