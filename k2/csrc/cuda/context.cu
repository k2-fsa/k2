// k2/csrc/cuda/context.cu

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

// WARNING(fangjun): this is a naive implementation to test the build system
#include <cstdlib>

#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/error.h"

static constexpr size_t kAlignment = 64;

namespace k2 {

// TODO(haowen): most of implementations below should be updated later.
class CpuContext : public Context {
 public:
  ContextPtr GetCpuContext() override { return nullptr; }
  ContextPtr GetPinnedContext() override { return nullptr; }
  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      int ret = posix_memalign(&p, kAlignment, bytes);
      assert(ret == 0);
    }
    *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

  void Deallocate(void *data, void *deleter_context) override { free(data); }
};

class CudaContext : public Context {
 public:
  CudaContext(int gpu_id) : gpu_id_(gpu_id) {
    if (gpu_id_ != -1) {
      CheckCudaError(cudaSetDevice(gpu_id_));
    }
    // TODO(haowen): choose one from available GPUs if gpu_id == -1?
    // and handle GPU ids from multiple machines.
    CheckCudaError(cudaStreamCreate(&stream_));
  }
  ContextPtr GetCpuContext() override { return nullptr; }
  ContextPtr GetPinnedContext() override { return nullptr; }
  DeviceType GetDeviceType() const override { return kCuda; }
  int GetDeviceId() const override { return gpu_id_; }

  void *Allocate(size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      CheckCudaError(cudaMalloc(&p, bytes));
    }
    *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void *deleter_context) override {
    cudaFree(data);
  }

  cudaStream_t GetCudaStream() const override { return stream_; }

  void Sync() const override { CheckCudaError(cudaStreamSynchronize(stream_)); }

  ~CudaContext() { CheckCudaError(cudaStreamDestroy(stream_)); }

 private:
  int gpu_id_;
  cudaStream_t stream_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int gpu_id /*= -1*/) {
  return std::make_shared<CudaContext>(gpu_id);
}

RegionPtr NewRegion(ContextPtr &context, size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  std::shared_ptr<Region> ans = std::make_shared<Region>();
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
