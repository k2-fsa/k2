/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#include <chrono>  // NOLINT
#include <memory>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/timer.h"

namespace k2 {

class TimerImpl {
 public:
  TimerImpl() = default;
  virtual ~TimerImpl() = default;
  virtual void Reset() = 0;
  // Return time in seconds
  virtual double Elapsed() = 0;
};

// modified from https://github.com/kaldi-asr/kaldi/blob/master/src/base/timer.h
class CpuTimerImpl : public TimerImpl {
 public:
  CpuTimerImpl() { Reset(); }

  using high_resolution_clock = std::chrono::high_resolution_clock;

  void Reset() override { begin_ = high_resolution_clock::now(); }

  // Return time in seconds
  double Elapsed() override {
    auto end = high_resolution_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin_);
    return diff.count() / 1000000.0;
  }

 private:
  high_resolution_clock::time_point begin_;
};

class CudaTimerImpl : public TimerImpl {
 public:
  explicit CudaTimerImpl(cudaStream_t stream) : stream_(stream) {
    K2_CUDA_SAFE_CALL(cudaEventCreate(&time_start_));
    K2_CUDA_SAFE_CALL(cudaEventCreate(&time_end_));
    Reset();
  }

  ~CudaTimerImpl() override {
    K2_CUDA_SAFE_CALL(cudaEventDestroy(time_start_));
    K2_CUDA_SAFE_CALL(cudaEventDestroy(time_end_));
  }

  void Reset() override {
    K2_CUDA_SAFE_CALL(cudaEventRecord(time_start_, stream_));
  }

  // Return time in seconds
  double Elapsed() override {
    K2_CUDA_SAFE_CALL(cudaEventRecord(time_end_, stream_));
    K2_CUDA_SAFE_CALL(cudaEventSynchronize(time_end_));

    float ms_elapsed = 0;
    K2_CUDA_SAFE_CALL(
        cudaEventElapsedTime(&ms_elapsed, time_start_, time_end_));
    return ms_elapsed / 1000;
  }

 private:
  cudaEvent_t time_start_;
  cudaEvent_t time_end_;
  cudaStream_t stream_;
};

Timer::Timer(ContextPtr context) {
  switch (context->GetDeviceType()) {
    case kCpu:
      timer_impl_ = std::make_unique<CpuTimerImpl>();
      break;
    case kCuda:
      timer_impl_ = std::make_unique<CudaTimerImpl>(context->GetCudaStream());
      break;
    default:
      K2_LOG(FATAL) << "Unsupported device type: " << context->GetDeviceType();
      break;
  }
}

Timer::~Timer() = default;

void Timer::Reset() const { timer_impl_->Reset(); }

// Return time in seconds
double Timer::Elapsed() const { return timer_impl_->Elapsed(); }

}  // namespace k2
