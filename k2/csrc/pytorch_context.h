/**
 * @brief
 * pytorch_context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Fangjun Kuang (csukuangfj@gmail.com)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_PYTORCH_CONTEXT_H_
#define K2_CSRC_PYTORCH_CONTEXT_H_

#include "c10/cuda/CUDACachingAllocator.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "torch/torch.h"

namespace k2 {

class PytorchContext : public Context {
 public:
  explicit PytorchContext(int32_t gpu_id) : gpu_id_(gpu_id) {
    if (gpu_id < 0) gpu_id_ = 0;  // TODO(fangjun): select a gpu
    auto ret = cudaSetDevice(gpu_id_);
    K2_CHECK_CUDA_ERROR(ret);
    // TODO(fangjun): invoke init only once
    c10::cuda::CUDACachingAllocator::init(gpu_id_ + 1);

    allocator_ = c10::cuda::CUDACachingAllocator::get();
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

  ContextPtr GetCpuContext() override { return nullptr; }

  ContextPtr GetPinnedContext() override { return nullptr; }

  DeviceType GetDeviceType() const override { return kCuda; }

  int32_t GetDeviceId() const override { return gpu_id_; }

  cudaStream_t GetCudaStream() const override {
    return c10::cuda::getCurrentCUDAStream(gpu_id_);
  }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = allocator_->raw_allocate(bytes);
    if (deleter_context) *deleter_context = nullptr;
    return p;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
    allocator_->raw_deallocate(data);
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Sync() const override {
    auto ret = cudaStreamSynchronize(GetCudaStream());
    K2_CHECK_CUDA_ERROR(ret);
  }

 private:
  torch::Allocator *allocator_;  // NOT owned here
  int32_t gpu_id_;
};

}  // namespace k2

#endif  // K2_CSRC_PYTORCH_CONTEXT_H_
