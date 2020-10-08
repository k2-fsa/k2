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

namespace k2 {

RegionPtr NewRegion(ContextPtr &context, std::size_t num_bytes) {
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

ParallelRunner::ParallelRunner(ContextPtr c) : c_(c) {
  if (c_->GetDeviceType() == kCuda) {
    auto ret = cudaEventCreate(&event_);
    K2_CHECK_CUDA_ERROR(ret);
    // record event on `c_->GetCudaStream` and will be waited on `NewStream`
    ret = cudaEventRecord(event_, c_->GetCudaStream());
    K2_CHECK_CUDA_ERROR(ret);
  }
}
cudaStream_t ParallelRunner::NewStream() {
  DeviceType d = c_->GetDeviceType();
  if (d == kCpu) {
    return kCudaStreamInvalid;
  } else {
    K2_CHECK_EQ(d, kCuda);
    cudaStream_t stream;
    auto ret = cudaStreamCreate(&stream);
    K2_CHECK_CUDA_ERROR(ret);
    streams_.push_back(stream);

    ret = cudaStreamWaitEvent(stream, event_, 0);
    K2_CHECK_CUDA_ERROR(ret);
    return stream;
  }
}

ParallelRunner::~ParallelRunner() {
  if (c_->GetDeviceType() == kCuda) {
    for (std::size_t i = 0; i != streams_.size(); ++i) {
      // create and record event on `stream_[i]`, and wait on c_->GetCudaStream
      cudaEvent_t event;
      auto ret = cudaEventCreate(&event);
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
}

}  // namespace k2
