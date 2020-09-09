/**
 * @brief
 * timer
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_TIMER_H_
#define K2_CSRC_TIMER_H_

#include "k2/csrc/log.cuh"

namespace k2 {
class Timer {
 public:
  Timer() {
    cudaEventCreate(&time_start_);
    cudaEventCreate(&time_end_);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());
    Reset();
  }

  ~Timer() {
    cudaEventDestroy(time_start_);
    cudaEventDestroy(time_end_);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());
  }

  void Reset() {
    cudaEventRecord(time_start_, 0);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());
  }

  double Elapsed() {
    cudaEventRecord(time_end_, 0);
    cudaEventSynchronize(time_end_);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());

    float ms_elapsed;
    cudaEventElapsedTime(&ms_elapsed, time_start_, time_end_);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());
    return ms_elapsed / 1e3;
  }

 private:
  cudaEvent_t time_start_;
  cudaEvent_t time_end_;
};

}  // namespace k2

#endif  // K2_CSRC_TIMER_H_
