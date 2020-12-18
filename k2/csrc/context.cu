/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (author: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"
#include "k2/csrc/eval.h"

namespace k2 {

void CudaStreamOverride::Push(cudaStream_t stream) {
  if (stream == static_cast<cudaStream_t>(0x0)) return;
  stack_.push_back(stream);
  stream_override_ = stream;
}

void CudaStreamOverride::Pop(cudaStream_t stream) {
  if (stream == static_cast<cudaStream_t>(0x0)) return;
  K2_DCHECK(!stack_.empty());
  K2_DCHECK_EQ(stack_.back(), stream);
  stack_.pop_back();
  if (stack_.empty())
    stream_override_ = static_cast<cudaStream_t>(0x0);
  else
    stream_override_ = stack_.back();
}

RegionPtr NewRegion(ContextPtr context, std::size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  auto ans = std::make_shared<Region>();
  ans->context = context;
  // TODO(haowen): deleter_context is always null with above constructor,
  // we need add another constructor of Region to allow the caller
  // to provide deleter_context.
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

ParallelRunnerActive::ParallelRunnerActive(ContextPtr c) : c_(c) {
  if (c_->GetDeviceType() == kCuda) {
    auto ret = cudaEventCreate(&event_);
    K2_CHECK_CUDA_ERROR(ret);
    // record event on `c_->GetCudaStream` and will be waited on `NewStream`
    ret = cudaEventRecord(event_, c_->GetCudaStream());
    K2_CHECK_CUDA_ERROR(ret);
  }
}
cudaStream_t ParallelRunnerActive::NewStream(
    std::size_t num_work_items /*=0*/) {
  DeviceType d = c_->GetDeviceType();
  if (d == kCpu) {
    return kCudaStreamInvalid;
  } else {
    K2_CHECK_EQ(d, kCuda);
    if (num_work_items < 10000) return static_cast<cudaStream_t>(0x0);
    cudaStream_t stream;
    auto ret = cudaStreamCreate(&stream);
    K2_CHECK_CUDA_ERROR(ret);
    streams_.push_back(stream);

    ret = cudaStreamWaitEvent(stream, event_, 0);
    K2_CHECK_CUDA_ERROR(ret);
    return stream;
  }
}

void ParallelRunnerActive::Finish() {
  if (c_.get() == nullptr) return;
  if (c_->GetDeviceType() == kCuda) {
    for (std::size_t i = 0; i != streams_.size(); ++i) {
      // create and record event on `stream_[i]`, and wait on c_->GetCudaStream
      cudaEvent_t event;
      auto ret = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      K2_CHECK_CUDA_ERROR(ret);
      ret = cudaEventRecord(event, streams_[i]);
      K2_CHECK_CUDA_ERROR(ret);
      ret = cudaStreamWaitEvent(c_->GetCudaStream(), event, 0);
      K2_CHECK_CUDA_ERROR(ret);
      ret = cudaEventDestroy(event);
      K2_CHECK_CUDA_ERROR(ret);
      ret = cudaStreamDestroy(streams_[i]);
      K2_CHECK_CUDA_ERROR(ret);
    }
    // destroy event_
    auto ret = cudaEventDestroy(event_);
    K2_CHECK_CUDA_ERROR(ret);
  }
  c_ = nullptr;
}

void GetBlockSizesForLambda2(int32_t m, int32_t n, dim3 *block_dim,
                             dim3 *grid_dim, Lambda2KernelType *kernel_type) {
  // Note: 'n' is the 'inner-loop' one, the one which is supposed to vary the
  // fastest.
  int32_t n_block_size = (n <= 256 ? n : 256);
  int32_t m_block_size = 1;
  while (m_block_size * n_block_size < 256)
    m_block_size *= 4;  // limit for the product is 1024; we don't go beyond
                        // 512.  (128 * 4 = 512).
  *block_dim = dim3(n_block_size, m_block_size, 1);
  int32_t n_grid_size = NumBlocks(n, n_block_size),
          m_grid_size = NumBlocks(m, m_block_size);
  if (n_grid_size < 65536 && m_grid_size < 65536) {
    *grid_dim = dim3(n_grid_size, m_grid_size, 1);
    *kernel_type = Lambda2KernelType::Simple;
  } else if (n_grid_size < 65536) {
    // only m is problematic.
    *grid_dim = dim3(n_grid_size, 32768, NumBlocks(m_grid_size, 32768));
    *kernel_type = Lambda2KernelType::UseZForM;
  } else {
    // we know n is problematic.
    if (m_grid_size > 65536) {
      K2_LOG(FATAL) << "Grid too large for Eval2(): m=" << m << ", n=" << n;
    }
    // only n is problematic.
    *grid_dim = dim3(32768, m_grid_size, NumBlocks(n_grid_size, 32768));
    *kernel_type = Lambda2KernelType::UseZForN;
  }
}


void Semaphore::Signal(ContextPtr c) {
  DeviceType device_type = c->GetDeviceType();
  if (device_type_ == kUnk)
    device_type_ = device_type;
  else
    K2_CHECK_EQ(device_type, device_type_)
        << "Semaphore must always be used with the same device type.";
  if (device_type == kCuda) {
    cudaEvent_t event;
    cudaError_t e = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    K2_CHECK_CUDA_ERROR(e) << "Error creating event";
    // Note: this stream is subject to being overridden by With(stream..), see
    // class With.
    cudaStream_t stream = c->GetCudaStream();
    e = cudaEventRecord(event, stream);
    K2_CHECK_CUDA_ERROR(e) << "Error recording event.";
    std::lock_guard<std::mutex> lock(events_mutex_);
    events_.push_back(event);
  }
  semaphore_.release();
}

void Semaphore::Wait(ContextPtr c) {
  DeviceType device_type = c->GetDeviceType();
  if (device_type_ == kUnk)
    device_type_ = device_type;
  else
    K2_CHECK_EQ(device_type, device_type_)
        << "Semaphore must always be used with the same device type.";
  semaphore_.acquire();
  if (device_type == kCuda) {
    cudaEvent_t event;
    {
      std::lock_guard<std::mutex> lock(events_mutex_);
      K2_CHECK(!events_.empty());  // would be code bug.
      event = events_.front();
      events_.pop_front();
    }
    int flags = 0;
    cudaError_t e = cudaStreamWaitEvent(c->GetCudaStream(), event,
                                        flags);
    K2_CHECK_CUDA_ERROR(e) << "Error waiting on event.";
  }
}

}  // namespace k2
