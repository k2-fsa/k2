// k2/csrc/cuda/timer.h

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_TIMER_H_
#define K2_CSRC_CUDA_TIMER_H_

#include "k2/csrc/cuda/error.h"

namespace k2 {
class Timer {
 public:
  Timer() {
    CheckCudaError(cudaEventCreate(&time_start_));
    CheckCudaError(cudaEventCreate(&time_end_));
    Reset();
  }

  ~Timer() {
    CheckCudaError(cudaEventDestroy(time_start_));
    CheckCudaError(cudaEventDestroy(time_end_));
  }

  void Reset() { CheckCudaError(cudaEventRecord(time_start_, 0)); }

  double Elapsed() {
    CheckCudaError(cudaEventRecord(time_end_, 0));
    CheckCudaError(cudaEventSynchronize(time_end_));

    float ms_elapsed;
    CheckCudaError(cudaEventElapsedTime(&ms_elapsed, time_start_, time_end_));
    return ms_elapsed / 1e3;
  }

 private:
  cudaEvent_t time_start_;
  cudaEvent_t time_end_;
};

}  // namespace k2

#endif  // K2_CSRC_CUDA_TIMER_H_
