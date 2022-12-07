/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT

#include "k2/csrc/context.h"
#include "k2/csrc/cub.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/log.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

static constexpr std::size_t kAlignment = 64;

class CpuContext : public Context {
 public:
  CpuContext() = default;
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

  void CopyDataTo(size_t num_bytes, const void *src, ContextPtr dst_context,
                  void *dst) override {
    DeviceType device_type = dst_context->GetDeviceType();
    switch (device_type) {
      case kCpu:
        memcpy(dst, src, num_bytes);
        break;
      case kCuda: {
        // CPU -> CUDA
        DeviceGuard guard(dst_context);
        ContextPtr pinned_context = GetPinnedContext();
        auto region = NewRegion(pinned_context, num_bytes);
        memcpy(region->data, src, num_bytes);
        pinned_context->CopyDataTo(num_bytes, region->data, dst_context, dst);
        break;
      }
      default:
        K2_LOG(FATAL) << "Unsupported device type: " << device_type;
        break;
    }
  }
};

class CudaContext : public Context {
 public:
  explicit CudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
#ifdef K2_WITH_CUDA
    if (gpu_id != -1) {
      auto ret = cudaSetDevice(gpu_id_);
      K2_CHECK_CUDA_ERROR(ret);
    } else {
      int current_gpu_id;
      auto ret = cudaGetDevice(&current_gpu_id);
      K2_CHECK_CUDA_ERROR(ret);
      gpu_id_ = current_gpu_id;
    }
    allocator_ = new cub::CachingDeviceAllocator();
#else
    K2_LOG(FATAL) << "Unreachable code.";
#endif
  }
  DeviceType GetDeviceType() const override { return kCuda; }
  int32_t GetDeviceId() const override { return gpu_id_; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
#ifdef K2_WITH_CUDA
    DeviceGuard guard(gpu_id_);
    auto ret = allocator_->DeviceAllocate(&p, bytes);  // the default stream is 0
    K2_CHECK_CUDA_ERROR(ret);
    if (deleter_context != nullptr) *deleter_context = nullptr;
#endif
    return p;
  }

  void CopyDataTo(size_t num_bytes, const void *src, ContextPtr dst_context,
                  void *dst) override{
    DeviceType device_type = dst_context->GetDeviceType();
    switch (device_type) {
      case kCpu: {
        cudaError_t ret =
            cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToHost);
        K2_CHECK_CUDA_ERROR(ret);
        break;
      }
      case kCuda: {
        cudaError_t ret =
            cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDeviceToDevice,
                            dst_context->GetCudaStream());
        K2_CHECK_CUDA_ERROR(ret);
        break;
      }
      default:
        K2_LOG(FATAL) << "Unsupported device type: " << device_type;
        break;
    }
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
#ifdef K2_WITH_CUDA
    DeviceGuard guard(gpu_id_);
    auto ret = allocator_->DeviceFree(data);
    K2_CHECK_CUDA_ERROR(ret);
#endif
  }

  cudaStream_t GetCudaStream() const override {
#ifdef K2_WITH_CUDA
    return g_stream_override.OverrideStream(0);
#else
    return kCudaStreamInvalid;
#endif
  }

  void Sync() const override {
    DeviceGuard guard(gpu_id_);
    auto ret = cudaStreamSynchronize(GetCudaStream());
    K2_CHECK_CUDA_ERROR(ret);
  }

  ~CudaContext() {
    delete allocator_;
  }

 private:
  int32_t gpu_id_;
  cub::CachingDeviceAllocator* allocator_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
#ifdef K2_WITH_CUDA
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

  DeviceGuard guard(gpu_id);
  if (has_cuda) return std::make_shared<CudaContext>(gpu_id);

  return GetCpuContext();
#else
  return GetCpuContext();
#endif
}

}  // namespace k2
