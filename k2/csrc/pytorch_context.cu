/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

#include <map>
#include <memory>
#include <mutex>  // NOLINT

#ifdef K2_WITH_CUDA
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "torch/cuda.h"
#endif

#ifdef K2_WITH_MPS
#include "torch/mps.h"
#endif

#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/log.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

#ifdef K2_WITH_MPS
// Global registry mapping MPS data pointers to their base tensors.
// When k2 allocates MPS memory via PytorchMpsContext::Allocate(), the
// resulting tensor is registered here so that CPU→MPS copies can use
// PyTorch's Metal-safe copy_() instead of a raw memcpy.
// A raw memcpy to a Metal buffer bypasses PyTorch's hazard tracking and
// causes subsequent Metal operations (e.g. .to('cpu')) to crash.
static std::mutex g_mps_registry_mutex;
static std::map<void *, torch::Tensor> g_mps_registry;

torch::Tensor MpsRegistryView(const void *ptr, int64_t n,
                               torch::ScalarType dtype) {
  std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
  const char *p = reinterpret_cast<const char *>(ptr);
  // Use upper_bound for O(log n) range lookup: find the first entry whose
  // base pointer is strictly greater than ptr, then step back one to get
  // the candidate allocation that may contain ptr.
  auto it = g_mps_registry.upper_bound(const_cast<void *>(ptr));
  if (it != g_mps_registry.begin()) {
    --it;
    const char *base = reinterpret_cast<const char *>(it->first);
    int64_t base_bytes = it->second.nbytes();
    if (p >= base && p < base + base_bytes) {
      ptrdiff_t byte_off = p - base;
      int64_t elem_size = torch::elementSize(dtype);
      K2_CHECK_EQ(byte_off % elem_size, 0)
          << "Unaligned MPS pointer: byte_off=" << byte_off
          << " elem_size=" << elem_size;
      return it->second.view(dtype).narrow(0, byte_off / elem_size, n);
    }
  }
  K2_LOG(FATAL) << "MPS pointer " << ptr
                << " not found in registry — was it allocated by "
                   "PytorchMpsContext::Allocate()?";
  return {};  // unreachable
}
#endif
// CAUTION: This is a workaround to free the CUDA memory
// correctly if `PYTORCH_NO_CUDA_MEMORY_CACHING` is set.
//
// We don't use the implementation from PyTorch since
// this function is not exported.
//
// Why do we need this function?
//
// From
// https://github.com/pytorch/pytorch/blob/d4045e9aa173b99d1135b4a64473a0fb630758d9/c10/core/Allocator.h#L154
//
// > If this returns a non nullptr, it means that allocate()
// > is guaranteed to return a unique_ptr with this deleter attached;
// > it means the rawAllocate and rawDeallocate APIs are safe to use.
// > This function MUST always return the same BoundDeleter.
//
// The comment says if `raw_allocate()` returns a non-nullptr, we can
// always use `raw_deallocate()`. However, this is not the case for
// CUDACachingAllocator.
//
// See
// https://github.com/pytorch/pytorch/blob/d4045e9aa173b99d1135b4a64473a0fb630758d9/c10/cuda/CUDACachingAllocator.cpp#L1190
//
// `CudaCachingAllocator::allocate()` returns a pointer associated with
// different deleters depending on whether the environment variable
// `PYTORCH_NO_CUDA_MEMORY_CACHING` is set. Thus, we have to be careful
// in choosing the deleter during the deallocation.
// Otherwise, you will be SAD.
//
// The environment variable is most useful for cuda-memcheck. It should
// never be set when cuda-memcheck is not used.
//
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

class PytorchCpuContext : public Context {
 public:
  PytorchCpuContext() {
    allocator_ = torch::GetAllocator(torch::kCPU);
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
#ifdef K2_WITH_MPS
      case kMps: {
        // CPU -> MPS: a raw memcpy bypasses PyTorch's Metal hazard tracking,
        // causing subsequent Metal operations (e.g. .to('cpu')) to crash.
        // Use PyTorch's Metal-safe copy_() via the global registry instead.
        torch::Tensor dst_base;
        bool found = false;
        {
          std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
          auto it = g_mps_registry.find(dst);
          if (it != g_mps_registry.end()) {
            dst_base = it->second;
            found = true;
          }
        }
        if (found) {
          // Wrap the CPU source in a from_blob tensor (safe for CPU).
          auto src_cpu = torch::from_blob(
              const_cast<void *>(src), {static_cast<int64_t>(num_bytes)},
              torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU));
          // Compute byte offset of dst within dst_base (0 for base pointers).
          int64_t byte_off = static_cast<int64_t>(
              static_cast<const char *>(dst) -
              static_cast<const char *>(dst_base.data_ptr()));
          // Use Metal copy_() to write into the MPS buffer.
          dst_base.narrow(0, byte_off, static_cast<int64_t>(num_bytes))
              .copy_(src_cpu);
        } else {
          // Fallback for MPS regions not created by Allocate() (e.g. wrapped
          // from a Python tensor via NewRegion(torch::Tensor)).
          memcpy(dst, src, num_bytes);
        }
        break;
      }
#endif
      default:
        K2_LOG(FATAL) << "Unsupported device type: " << device_type;
        break;
    }
  }

 private:
  torch::Allocator *allocator_;  // NOT owned here
};

class PytorchCudaContext : public Context {
 public:
  explicit PytorchCudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
#ifdef K2_WITH_CUDA
    K2_CHECK_GE(gpu_id, 0);
    K2_CHECK_LT(gpu_id, c10::cuda::device_count());

    c10::cuda::set_device(gpu_id);

    // The internals of `lazyInitCUDA` are executed only once
    // so it is fine to invoke lazyInitCUDA() multiple times.
    // The call will be inlined since it is defined in the header
    // aten/src/ATen/Context.h
#if K2_TORCH_VERSION_MAJOR > 2 || \
    (K2_TORCH_VERSION_MAJOR == 2 && K2_TORCH_VERSION_MINOR >= 6)
    at::globalContext().lazyInitDevice(torch::kCUDA);
#else
    at::globalContext().lazyInitCUDA();
#endif

    allocator_ = c10::cuda::CUDACachingAllocator::get();
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

#ifdef K2_WITH_MPS
class PytorchMpsContext : public Context {
 public:
  PytorchMpsContext() {}

  DeviceType GetDeviceType() const override { return kMps; }

  // Apple Silicon has a single MPS device.
  int32_t GetDeviceId() const override { return 0; }

  // MPS uses Metal command queues, not CUDA streams; return the sentinel so
  // that Eval() in eval.h routes MPS through the CPU sequential loop.
  cudaStream_t GetCudaStream() const override { return kCudaStreamInvalid; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    // Allocate via torch::Tensor so PyTorch's MPS memory manager owns the
    // Metal buffer.  The tensor is kept alive via ManagedTensor stored in
    // deleter_context; Deallocate() drops it to release the buffer.
    auto tensor = torch::empty({static_cast<int64_t>(bytes)},
                               torch::TensorOptions()
                                   .dtype(torch::kByte)
                                   .device(torch::kMPS));
    void *p = tensor.data_ptr();
    if (deleter_context != nullptr) {
      *deleter_context = new ManagedTensor(tensor);
      // Register in the global registry so that CPU→MPS copies (in
      // PytorchCpuContext::CopyDataTo) can use Metal-safe copy_() to write
      // into this buffer rather than a raw memcpy.
      std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
      g_mps_registry[p] = tensor;
    } else {
      // Caller opted out of tracking; memory will be released when the
      // tensor goes out of scope here.  This should not happen in practice
      // because k2 always passes a non-null deleter_context via NewRegion().
      K2_LOG(FATAL) << "PytorchMpsContext::Allocate called with null "
                       "deleter_context — MPS memory would be immediately "
                       "freed. This is a k2 bug.";
    }
    return p;
  }

  void Deallocate(void *data, void *deleter_context) override {
    if (deleter_context != nullptr) {
      // Unregister from the global registry before freeing the tensor.
      {
        std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
        g_mps_registry.erase(data);
      }
      // deleter_context holds a ManagedTensor; dropping it releases the MPS
      // buffer back to PyTorch's allocator.
      delete reinterpret_cast<ManagedTensor *>(deleter_context);
    } else {
      // Should not happen: every Allocate() stores a ManagedTensor.
      K2_LOG(FATAL) << "PytorchMpsContext::Deallocate called with null "
                       "deleter_context — cannot free MPS memory.";
    }
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kMps;
  }

  void Sync() const override { torch::mps::synchronize(); }

  void CopyDataTo(size_t num_bytes, const void *src, ContextPtr dst_context,
                  void *dst) override {
    DeviceType device_type = dst_context->GetDeviceType();
    switch (device_type) {
      case kCpu: {
        // MPS -> CPU: PyTorch's MPS backend stores tensor data in Metal
        // buffers that are NOT at data_ptr() on the CPU side.  We must use
        // Metal-aware copy_() rather than a raw memcpy.
        torch::Tensor src_base;
        {
          std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
          auto it = g_mps_registry.find(const_cast<void *>(src));
          if (it != g_mps_registry.end()) src_base = it->second;
        }
        if (src_base.defined()) {
          int64_t byte_off = static_cast<int64_t>(
              static_cast<const char *>(src) -
              static_cast<const char *>(src_base.data_ptr()));
          auto src_view =
              src_base.narrow(0, byte_off, static_cast<int64_t>(num_bytes));
          // Create a CPU tensor view of dst and copy via Metal.
          auto dst_cpu = torch::from_blob(
              dst, {static_cast<int64_t>(num_bytes)},
              torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU));
          dst_cpu.copy_(src_view);
        } else {
          // Fallback for regions not created by Allocate() (e.g. wrapped from
          // a Python tensor).  Data may be stale if copy_() was used for
          // the CPU->MPS write.
          torch::mps::synchronize();
          memcpy(dst, src, num_bytes);
        }
        break;
      }
      case kMps: {
        // MPS -> MPS: use Metal copy_() to stay within Metal's hazard
        // tracking system.
        torch::Tensor src_base, dst_base;
        {
          std::lock_guard<std::mutex> lock(g_mps_registry_mutex);
          auto sit = g_mps_registry.find(const_cast<void *>(src));
          if (sit != g_mps_registry.end()) src_base = sit->second;
          auto dit = g_mps_registry.find(dst);
          if (dit != g_mps_registry.end()) dst_base = dit->second;
        }
        if (src_base.defined() && dst_base.defined()) {
          int64_t src_off = static_cast<int64_t>(
              static_cast<const char *>(src) -
              static_cast<const char *>(src_base.data_ptr()));
          int64_t dst_off = static_cast<int64_t>(
              static_cast<const char *>(dst) -
              static_cast<const char *>(dst_base.data_ptr()));
          auto src_view =
              src_base.narrow(0, src_off, static_cast<int64_t>(num_bytes));
          dst_base.narrow(0, dst_off, static_cast<int64_t>(num_bytes))
              .copy_(src_view);
        } else {
          // Fallback: memcpy for regions not in the registry.
          memcpy(dst, src, num_bytes);
        }
        break;
      }
      default:
        K2_LOG(FATAL) << "Unsupported device type: " << device_type;
        break;
    }
  }
};
#endif  // K2_WITH_MPS

ContextPtr GetCpuContext() { return std::make_shared<PytorchCpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  std::call_once(has_cuda_init_flag, InitHasCuda);

  if (has_cuda) {
#ifdef K2_WITH_CUDA
    if (gpu_id < 0) gpu_id = c10::cuda::current_device();
    DeviceGuard guard(gpu_id);
    return std::make_shared<PytorchCudaContext>(gpu_id);
#else
    K2_LOG(FATAL) << "Unreachable code.";
    return nullptr;
#endif
  }

  return GetCpuContext();
}

ContextPtr GetMpsContext() {
#ifdef K2_WITH_MPS
  if (torch::mps::is_available()) {
    // Trigger lazy MPS backend initialization so the allocator is registered
    // before PytorchMpsContext tries to fetch it.
    torch::empty({0}, torch::TensorOptions().device(torch::kMPS));
    return std::make_shared<PytorchMpsContext>();
  }
  K2_LOG(WARNING) << "MPS is not available. Falling back to CPU context.";
#else
  K2_LOG(WARNING) << "k2 was not compiled with MPS support. "
                     "Falling back to CPU context.";
#endif
  return GetCpuContext();
}

RegionPtr NewRegion(torch::Tensor tensor) {
  auto ans = std::make_shared<Region>();
  if (tensor.device().type() == torch::kCPU) {
    ans->context = GetCpuContext();
  } else if (tensor.is_cuda()) {
    ans->context = GetCudaContext(tensor.device().index());
#ifdef K2_WITH_MPS
  } else if (tensor.device().type() == torch::kMPS) {
    ans->context = GetMpsContext();
#endif
  } else {
    K2_LOG(FATAL) << "Unsupported device: " << tensor.device()
                  << "\nOnly CPU, CUDA, and MPS are supported";
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
