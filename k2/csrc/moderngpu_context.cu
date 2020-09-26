/**
 * @brief A context for moderngpu with a better memory allocator.
 *
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"
#include "k2/csrc/moderngpu_context.h"
#include "moderngpu/context.hxx"

namespace {

class ModernGpuContext : public mgpu::standard_context_t {
 public:
  explicit ModernGpuContext(k2::ContextPtr context)
      : mgpu::standard_context_t(false, context->GetCudaStream()),
        context_(std::move(context)) {}

  void *alloc(size_t size, mgpu::memory_space_t /*space*/) override {
    void *deleter_ = nullptr;
    void *p = context_->Allocate(size, &deleter_);
    K2_DCHECK(deleter_ == nullptr);
    return p;
  }

  void free(void *p, mgpu::memory_space_t /*space*/) override {
    context_->Deallocate(p, nullptr);
  }

 private:
  k2::ContextPtr context_;
};

}  // namespace

namespace k2 {

std::unique_ptr<mgpu::context_t> GetModernGpuContext(
    DeviceType type, int32_t device_id /*= -1*/) {
  ContextPtr context;
  if (type == kCuda)
    context = GetCudaContext(device_id);
  else if (type == kCpu)
    context = GetCpuContext();
  else
    K2_LOG(FATAL) << "Unknown device type: " << type;

  return std::make_unique<ModernGpuContext>(context);
}

}  // namespace k2
