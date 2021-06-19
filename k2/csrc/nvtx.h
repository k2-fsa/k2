/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
#ifndef K2_CSRC_NVTX_H_
#define K2_CSRC_NVTX_H_

#ifdef K2_ENABLE_NVTX
#include "nvToolsExt.h"
#endif

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
