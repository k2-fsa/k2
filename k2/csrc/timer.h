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

#include "k2/csrc/log.h"

namespace k2 {

class Timer {
 public:
  Timer() {
    static std::once_flag has_cuda_init_flag;
    static bool has_cuda = false;
    std::call_once(has_cuda_init_flag, []() {
      int n = 0;
      auto ret = cudaGetDeviceCount(&n);

      if (ret == cudaSuccess && n > 0)
        has_cuda = true;
      else
        K2_LOG(WARNING) << "CUDA is not available. Disable timer.";
    });
    has_cuda_ = has_cuda;

    if (has_cuda_) {
      K2_CUDA_SAFE_CALL(cudaEventCreate(&time_start_));
      K2_CUDA_SAFE_CALL(cudaEventCreate(&time_end_));
      Reset();
    }
  }

  ~Timer() {
    if (has_cuda_) {
      K2_CUDA_SAFE_CALL(cudaEventDestroy(time_start_));
      K2_CUDA_SAFE_CALL(cudaEventDestroy(time_end_));
    }
  }

  void Reset() {
    if (has_cuda_) K2_CUDA_SAFE_CALL(cudaEventRecord(time_start_, 0));
  }

  double Elapsed() {
    if (has_cuda_) {
      K2_CUDA_SAFE_CALL(cudaEventRecord(time_end_, 0));
      K2_CUDA_SAFE_CALL(cudaEventSynchronize(time_end_));

      float ms_elapsed;
      K2_CUDA_SAFE_CALL(
          cudaEventElapsedTime(&ms_elapsed, time_start_, time_end_));
      return ms_elapsed / 1e3;
    } else {
      return 1;
    }
  }

 private:
  cudaEvent_t time_start_;
  cudaEvent_t time_end_;
  bool has_cuda_;
};

}  // namespace k2

#endif  // K2_CSRC_TIMER_H_
