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

#include <memory>

#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "torch/torch.h"

namespace k2 {

class ManagedTensor {
 public:
  explicit ManagedTensor(torch::Tensor &tensor) : handle_(tensor) {}

 private:
  torch::Tensor handle_;  // retain a copy of the tensor passed from Python
};

class PytorchCpuContext : public Context {
 private:
  PytorchCpuContext() {
    allocator_ = torch::GetAllocator(torch::kCPU);
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

 public:
  static ContextPtr Make() {
    auto p = new PytorchCpuContext();
    return ContextPtr{p};
  }

  // since the constructor is private, the only way to create an instance
  // of PytorchCpuContext is via `Make`, which returns a `shared_ptr`.
  // Thus it is safe to call `shared_from_this`.
  ContextPtr GetCpuContext() override { return shared_from_this(); }

  ContextPtr GetPinnedContext() override { return nullptr; }

  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = allocator_->raw_allocate(bytes);
    if (deleter_context) *deleter_context = nullptr;
    return p;
  }

  void Deallocate(void *data, void *deleter_context) override {
    if (deleter_context) {
      // a non-empty `deleter_context` indicates that
      // the memory is passed from a `torch::Tensor`
      delete reinterpret_cast<ManagedTensor *>(deleter_context);
    } else {
      allocator_->raw_deallocate(data);
    }
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

 private:
  torch::Allocator *allocator_;  // NOT owned here
};

class PytorchCudaContext : public Context {
 public:
  explicit PytorchCudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
    K2_CHECK_GE(gpu_id, 0);
    K2_CHECK_LT(gpu_id, c10::cuda::device_count());

    c10::cuda::set_device(gpu_id);

    // The internals of `lazyInitCUDA` are executed only once
    // so it is fine to invoke lazyInitCUDA() multiple times.
    // The call will be inlined since it is defined in the header
    // aten/src/ATen/Context.h
    at::globalContext().lazyInitCUDA();

    allocator_ = c10::cuda::CUDACachingAllocator::get();
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

  ContextPtr GetCpuContext() override { return k2::GetCpuContext(); }

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

  void Deallocate(void *data, void *deleter_context) override {
    if (deleter_context) {
      // a non-empty `deleter_context` indicates that
      // the memory is passed from a `torch::Tensor`
      delete reinterpret_cast<ManagedTensor *>(deleter_context);
    } else {
      allocator_->raw_deallocate(data);
    }
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

// Construct a region from a `torch::Tensor`.
//
// The resulting region shares the underlying memory with
// the given tensor.
RegionPtr NewRegion(torch::Tensor &tensor);

}  // namespace k2

#endif  // K2_CSRC_PYTORCH_CONTEXT_H_
