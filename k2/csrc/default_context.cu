/**
 * @brief
 * default_context
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cstdlib>
#include <mutex>  // NOLINT

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

static constexpr std::size_t kAlignment = 64;

// TODO(haowen): most of implementations below should be updated later.
class CpuContext : public Context {
 public:
  CpuContext() = default;
  ContextPtr GetCpuContext() override { return shared_from_this(); }
  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      int32_t ret = posix_memalign(&p, kAlignment, bytes);
      K2_CHECK_EQ(ret, 0);
    }
    if (deleter_context != nullptr) *deleter_context = nullptr;
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
  ContextPtr GetCpuContext() override { return k2::GetCpuContext(); }
  DeviceType GetDeviceType() const override { return kCuda; }
  int32_t GetDeviceId() const override { return gpu_id_; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      auto ret = cudaMalloc(&p, bytes);
      K2_CHECK_CUDA_ERROR(ret);
    }
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
    auto ret = cudaFree(data);
    K2_CHECK_CUDA_ERROR(ret);
  }

  cudaStream_t GetCudaStream() const override {
    return g_stream_override.OverrideStream(stream_);
  }

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
  static std::once_flag has_cuda_init_flag;
  static bool has_cuda = false;
  std::call_once(has_cuda_init_flag, []() {
    int n = 0;
    auto ret = cudaGetDeviceCount(&n);
    if (ret == cudaSuccess && n > 0)
      has_cuda = true;
    else
      K2_LOG(WARNING) << "CUDA is not available. Return a CPU context.";
  });

  if (has_cuda) return std::make_shared<CudaContext>(gpu_id);

  return GetCpuContext();
}

}  // namespace k2
