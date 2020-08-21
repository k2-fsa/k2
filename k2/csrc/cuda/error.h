// k2/csrc/cuda/error.h

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ERROR_H_
#define K2_CSRC_CUDA_ERROR_H_

#include <string>

#include "glog/logging.h"

namespace k2 {
inline void CheckCudaError(cudaError_t status,
                           const std::string &message = "") {
#if !defined(NDEBUG)
  if (status != cudaSuccess) {
    LOG(FATAL) << message
               << "\nCUDA Runtime Error: " << cudaGetErrorString(status);
  }
#endif
}
}  // namespace k2

#endif  // K2_CSRC_CUDA_ERROR_H_
