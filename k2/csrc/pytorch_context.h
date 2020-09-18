/**
 * @brief
 * pytorch_context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
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
  // if device_id < 0, then this is a cpu context;
  // otherwise, it is a cuda context.
  explicit PytorchContext(int32_t device_id) : device_id_(device_id) {
    if (device_id_ < 0)
      InitCpu();
    else
      InitCuda();
  }

  ContextPtr GetCpuContext() override {
    // TODO(fangjun): return `this` if it's cpu ?
    return nullptr;
  }

  ContextPtr GetPinnedContext() override { return nullptr; }

  DeviceType GetDeviceType() const override {
    return device_id_ >= 0 ? kCuda : kCpu;
  }

  int32_t GetDeviceId() const override { return device_id_; }

  cudaStream_t GetCudaStream() const override {
    return device_id_ >= 0 ? c10::cuda::getCurrentCUDAStream(device_id_)
                           : kCudaStreamInvalid;
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
    return other.GetDeviceType() == GetDeviceType() &&
           other.GetDeviceId() == device_id_;
  }

  void Sync() const override {
    if (device_id_ >= 0) {
      auto ret = cudaStreamSynchronize(GetCudaStream());
      K2_CHECK_CUDA_ERROR(ret);
    }
  }

 private:
  void InitCpu() {
    allocator_ = torch::GetAllocator(torch::kCPU);
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

  void InitCuda() {
    auto ret = cudaSetDevice(device_id_);
    K2_CHECK_CUDA_ERROR(ret);
    // TODO(fangjun): invoke init only once
    c10::cuda::CUDACachingAllocator::init(device_id_ + 1);

    allocator_ = c10::cuda::CUDACachingAllocator::get();
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

 private:
  torch::Allocator *allocator_;  // NOT owned here
  int32_t device_id_;
};

}  // namespace k2

#endif  // K2_CSRC_PYTORCH_CONTEXT_H_
