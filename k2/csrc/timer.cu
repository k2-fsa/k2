
/**
 * @brief
 * timer
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <sys/time.h>

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

  void Reset() override { gettimeofday(&time_start_, nullptr); }

  // Return time in seconds
  double Elapsed() override {
    struct timeval time_end;
    gettimeofday(&time_end, nullptr);
    double t1, t2;
    t1 = static_cast<double>(time_start_.tv_sec) +
         static_cast<double>(time_start_.tv_usec) / (1000 * 1000);
    t2 = static_cast<double>(time_end.tv_sec) +
         static_cast<double>(time_end.tv_usec) / (1000 * 1000);
    return t2 - t1;
  }

 private:
  struct timeval time_start_;
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

    float ms_elapsed;
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
