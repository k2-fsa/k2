/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
 *                      Xiaomi Corporation (author: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cstdlib>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

static constexpr std::size_t kAlignment = 64;

namespace k2 {

// TODO(haowen): most of implementations below should be updated later.
class CpuContext : public Context {
 public:
  ContextPtr GetCpuContext() override { return nullptr; }
  ContextPtr GetPinnedContext() override { return nullptr; }
  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      int32_t ret = posix_memalign(&p, kAlignment, bytes);
      K2_DCHECK_EQ(ret, 0);
    }
    if (deleter_context) *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
    free(data);
  }
};

class CudaContext : public Context {
 public:
  explicit CudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
    if (gpu_id_ != -1) {
      auto ret = cudaSetDevice(gpu_id_);
      K2_CHECK_CUDA_ERROR(ret);
    }
    // TODO(haowen): choose one from available GPUs if gpu_id == -1?
    // and handle GPU ids from multiple machines.
    auto ret = cudaStreamCreate(&stream_);
    K2_CHECK_CUDA_ERROR(ret);
  }
  ContextPtr GetCpuContext() override { return nullptr; }
  ContextPtr GetPinnedContext() override { return nullptr; }
  DeviceType GetDeviceType() const override { return kCuda; }
  int GetDeviceId() const override { return gpu_id_; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      auto ret = cudaMalloc(&p, bytes);
      K2_CHECK_CUDA_ERROR(ret);
    }
    if (deleter_context) *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
    auto ret = cudaFree(data);
    K2_CHECK_CUDA_ERROR(ret);
  }

  cudaStream_t GetCudaStream() const override { return stream_; }

  void Sync() const override {
    auto ret = cudaStreamSynchronize(stream_);
    K2_CHECK_CUDA_ERROR(ret);
  }

  ~CudaContext() {
    auto ret = cudaStreamDestroy(stream_);
    K2_CHECK_CUDA_ERROR(ret);
  }

 private:
  int32_t gpu_id_;
  cudaStream_t stream_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  return std::make_shared<CudaContext>(gpu_id);
}

RegionPtr NewRegion(ContextPtr &context, std::size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  auto ans = std::make_shared<Region>();
  ans->context = context->shared_from_this();
  // TODO(haowen): deleter_context is always null with above constructor,
  // we need add another constructor of Region to allow the caller
  // to provide deleter_context.
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

}  // namespace k2
