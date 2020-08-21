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
      // check the return code
    }
    *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context & /*other*/) const override { return true; }

  void Deallocate(void *data, void *deleter_context) override { free(data); }
};

class CudaContext : public Context {
 public:
  CudaContext() { CheckCudaError(cudaStreamCreate(&stream_)); }
  ContextPtr GetCpuContext() override { return nullptr; }
  ContextPtr GetPinnedContext() override { return nullptr; }
  DeviceType GetDeviceType() const override { return kCuda; }

  void *Allocate(size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      CheckCudaError(cudaMalloc(&p, bytes));
    }
    *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context & /*other*/) const override {
    // TODO: change this
    return true;
  }

  void Deallocate(void *data, void *deleter_context) override {
    cudaFree(data);
  }

  cudaStream_t GetCudaStream() const override { return stream_; }

  ~CudaContext() { CheckCudaError(cudaStreamDestroy(stream_)); }

 private:
  cudaStream_t stream_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int gpu_id /*= -1*/) {
  // TODO: select a gpu
  return std::make_shared<CudaContext>();
}

}  // namespace k2
