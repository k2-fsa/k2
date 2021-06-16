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

// See
// https://github.com/k2-fsa/k2/issues/698
// and
// https://github.com/pytorch/pytorch/issues/54245#issuecomment-805707551
// for why we need the following two macros
//
// NOTE: We define the following two macros so
// that k2 and PyTorch use a different copy
// of CUB.

#define CUB_NS_PREFIX namespace k2 {
#define CUB_NS_POSTFIX }

#ifdef K2_WITH_CUDA
#include "cub/cub.cuh"  // NOLINT
#endif

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#endif  // K2_CSRC_CUB_H_
