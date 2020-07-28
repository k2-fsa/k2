// k2/csrc/cuda/context.cu

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

// WARNING(fangjun): this is a naive implementation to test the build system
#include "k2/csrc/cuda/context.h"

#include <cstdlib>

static constexpr size_t kAlignment = 64;

namespace k2 {

class CpuContext : public Context {
 public:
  ContextPtr Duplicate() override { return nullptr; }

  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(size_t bytes) override {
    void *p = nullptr;
    if (bytes) {
      int ret = posix_memalign(&p, kAlignment, bytes);
      // check the return code
    }
    return p;
  }

  bool IsSame(const Context & /*other*/) const override { return true; }

  void Deallocate(void *data) override { free(data); }
};

class CudaContext : public Context {
 public:
  ContextPtr Duplicate() override { return nullptr; }

  DeviceType GetDeviceType() const override { return kCuda; }

  void *Allocate(size_t bytes) override {
    void *p = nullptr;
    if (bytes) {
      cudaError_t ret = cudaMalloc(&p, bytes);
      // check the return code
    }
    return p;
  }

  bool IsSame(const Context & /*other*/) const override {
    // TODO: change this
    return true;
  }

  void Deallocate(void *data) override { cudaFree(data); }
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int gpu_id /*= -1*/) {
  // TODO: select a gpu
  return std::make_shared<CudaContext>();
}

}  // namespace k2
