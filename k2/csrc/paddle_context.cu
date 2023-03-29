/**
 * Copyright      2023  PaddlePaddle         (authors: Hui Zhang)
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

#include <memory>
#include <mutex>  // NOLINT

// #ifdef K2_WITH_CUDA
// #include "c10/cuda/CUDACachingAllocator.h"
// #include "c10/cuda/CUDAFunctions.h"
// #include "torch/cuda.h"
// #endif

#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/log.h"
#include "k2/csrc/paddle_context.h"

namespace phi = paddle::phi;

namespace k2 {

bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static std::once_flag has_cuda_init_flag;
static bool has_cuda = false;
static void InitHasCuda() {
#ifdef K2_WITH_CUDA
  if (torch::cuda::is_available())
    has_cuda = true;
  else
    K2_LOG(WARNING) << "CUDA is not available. Return a CPU context.";
#else
  K2_LOG(WARNING) << "k2 was not compiled with CUDA. Return a CPU context.";
#endif
}

class PaddleCpuContext : public Context {
 public:
  PaddleCpuContext() {
    // allocator_ = torch::GetAllocator(torch::kCPU);
    allocator_ = paddle::GetAllocator(phi::Place(phi::AllocationType::CPU));
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    int64_t max_bytes = internal::MaxCpuMemAllocate();
    if (max_bytes != -1) K2_CHECK_LE(static_cast<int64_t>(bytes), max_bytes);

    void *p = allocator_->raw_allocate(bytes);
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  void Deallocate(void *data, void *deleter_context) override {
    if (deleter_context != nullptr) {
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

 private:
  // torch::Allocator *allocator_;  // NOT owned here
  phi::Allocator *allocator_; // NOT owned here
};

class PaddleCudaContext : public Context {
 public:
  explicit PaddleCudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
#ifdef K2_WITH_CUDA
    K2_CHECK_GE(gpu_id, 0);
    K2_CHECK_LT(gpu_id, c10::cuda::device_count());

    c10::cuda::set_device(gpu_id);

    // The internals of `lazyInitCUDA` are executed only once
    // so it is fine to invoke lazyInitCUDA() multiple times.
    // The call will be inlined since it is defined in the header
    // aten/src/ATen/Context.h
    // at::globalContext().lazyInitCUDA();

    // allocator_ = c10::cuda::CUDACachingAllocator::get();
    allocator_ = phi::GetAllocator(phi::Place(phi::AllocationType::GPU, gpu_id));

    K2_CHECK(allocator_->raw_deleter() != nullptr);
#else
    K2_LOG(FATAL) << "Unreachable code.";
#endif
  }

  DeviceType GetDeviceType() const override { return kCuda; }

  int32_t GetDeviceId() const override { return gpu_id_; }

  cudaStream_t GetCudaStream() const override {
#ifdef K2_WITH_CUDA
    return g_stream_override.OverrideStream(
        c10::cuda::getCurrentCUDAStream(gpu_id_));
#else
    return cudaStream_t{};
#endif
  }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    // NOTE(fangjun): raw_allocate() returns a torch::DataPtr, which is
    // implicitly converted to a raw pointer. After this statement, the returned
    // torch::DataPtr object is freed. We could have saved torch::DataPtr's
    // deleter in `deleter_context`, but we use `deleter_context` already for
    // `ManagedTensor`. Therefore, we use forceUncachedAllocator() to choose
    // its deleter.
    //
    //
    // CAUTION: Update this if PyTorch changes its implementation.
    DeviceGuard guard(gpu_id_);
    void *p = allocator_->raw_allocate(bytes);
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  void Deallocate(void *data, void *deleter_context) override {
    DeviceGuard guard(gpu_id_);
    if (deleter_context != nullptr) {
      // a non-empty `deleter_context` indicates that
      // the memory is passed from a `torch::Tensor`
      delete reinterpret_cast<ManagedTensor *>(deleter_context);
    } else {
      // NOTE: See the comment in `Allocate`
      if (forceUncachedAllocator()) {
        K2_CHECK_CUDA_ERROR(cudaFree(data));
      } else {
        allocator_->raw_deallocate(data);
      }
    }
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Sync() const override {
    DeviceGuard guard(gpu_id_);
    auto ret = cudaStreamSynchronize(GetCudaStream());
    K2_CHECK_CUDA_ERROR(ret);
  }

  void CopyDataTo(size_t num_bytes, const void *src, ContextPtr dst_context,
                  void *dst) override {
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

 private:
  torch::Allocator *allocator_;  // NOT owned here
  int32_t gpu_id_;
};

ContextPtr GetCpuContext() { return std::make_shared<PaddleCpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  std::call_once(has_cuda_init_flag, InitHasCuda);

  if (has_cuda) {
#ifdef K2_WITH_CUDA
    if (gpu_id < 0) gpu_id = c10::cuda::current_device();
    DeviceGuard guard(gpu_id);
    return std::make_shared<PaddleCudaContext>(gpu_id);
#else
    K2_LOG(FATAL) << "Unreachable code.";
    return nullptr;
#endif
  }

  return GetCpuContext();
}

RegionPtr NewRegion(torch::Tensor tensor) {
  auto ans = std::make_shared<Region>();
  if (tensor.device().type() == torch::kCPU) {
    ans->context = GetCpuContext();
  } else if (tensor.is_cuda()) {
    ans->context = GetCudaContext(tensor.device().index());
  } else {
    K2_LOG(FATAL) << "Unsupported device: " << tensor.device()
                  << "\nOnly CPU and CUDA are supported";
  }

  // NOTE: the tensor is passed from Python and we have
  // to retain it to avoid potential segmentation fault.
  //
  // It will be freed in `Context::Deallocate`.
  auto *managed_tensor = new ManagedTensor(tensor);
  ans->data = tensor.data_ptr();
  ans->deleter_context = managed_tensor;
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR > 5)
  // nbytes() is available only for torch > 1.5
  // see https://github.com/pytorch/pytorch/pull/37028
  ans->num_bytes = tensor.storage().nbytes();
#else
  // capacity() is available only for torch <= 1.5.0
  ans->num_bytes = tensor.storage().capacity();
#endif
  ans->bytes_used = ans->num_bytes;
  return ans;
}

}  // namespace k2
