/**
 * @brief
 * nvtx.h
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#ifndef K2_CSRC_NVTX_H_
#define K2_CSRC_NVTX_H_

#include "nvToolsExt.h"

namespace k2 {

class NvtxRange {
 public:
  explicit NvtxRange(const char *name) {
#ifdef K2_ENABLE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~NvtxRange() {
#ifdef K2_ENABLE_NVTX
    nvtxRangePop();
#endif
  }
};

#define _K2_CONCAT(a, b) a##b
#define K2_CONCAT(a, b) _K2_CONCAT(a, b)

#ifdef __COUNTER__
#define K2_UNIQUE_VARIABLE_NAME(name) K2_CONCAT(name, __COUNTER__)
#else
#define K2_UNIQUE_VARIABLE_NAME(name) K2_CONCAT(name, __LINE__)
#endif

#ifdef K2_ENABLE_NVTX
#define NVTX_RANGE(name) k2::NvtxRange K2_UNIQUE_VARIABLE_NAME(k2_nvtx_)(name)
#else
#define NVTX_RANGE(name)
#endif

}  // namespace k2

#endif  // K2_CSRC_NVTX_H_
