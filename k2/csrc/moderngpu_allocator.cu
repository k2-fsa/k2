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

#include <mutex>  // NOLINT
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

// maximum number of GPUs supported by k2
static constexpr int32_t kMaxNumGpus = 16;

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
