/**
 * Copyright      2021  xiaomi Corporation (authors: Fangjun Kuang)
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
 *
 */

#ifndef K2_CSRC_CUB_H_
#define K2_CSRC_CUB_H_

#ifdef K2_WITH_CUDA
#ifdef K2_USE_NVTX3
#include <nvtx3/nvToolsExt.h>
#else
#include "nvToolsExt.h"
#endif

#include "cub/cub.cuh"  // NOLINT
#endif

#endif  // K2_CSRC_CUB_H_
