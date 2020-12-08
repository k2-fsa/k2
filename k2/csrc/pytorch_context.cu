/**
 * @brief
 * pytorch_context
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <memory>
#include <mutex>  // NOLINT

#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

static std::once_flag has_cuda_init_flag;
static bool has_cuda = false;
static void InitHasCuda() {
  if (torch::cuda::is_available())
    has_cuda = true;
  else
    K2_LOG(WARNING) << "CUDA is not available. Return a CPU context.";
}

class PytorchCpuContext : public Context {
 public:
  PytorchCpuContext() {
    allocator_ = torch::GetAllocator(torch::kCPU);
    K2_CHECK(allocator_->raw_deleter() != nullptr);
  }

  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
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

  DeviceType GetDeviceType() const override { return kCuda; }

  int32_t GetDeviceId() const override { return gpu_id_; }

  cudaStream_t GetCudaStream() const override {
    return g_stream_override.OverrideStream(
        c10::cuda::getCurrentCUDAStream(gpu_id_));
  }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
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
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Sync() const override {
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

ContextPtr GetCpuContext() { return std::make_shared<PytorchCpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  std::call_once(has_cuda_init_flag, InitHasCuda);

  if (has_cuda) {
    if (gpu_id < 0) gpu_id = c10::cuda::current_device();
    return std::make_shared<PytorchCudaContext>(gpu_id);
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
  ans->num_bytes = tensor.storage().nbytes();
  ans->bytes_used = ans->num_bytes;
  return ans;
}

}  // namespace k2
