/**
 * k2/csrc/cudpp/cudpp_hip.cu
 *
 * Copyright (c) 2026  Advanced Micro Devices, Inc. (authors: Jeff Daily <jeff.daily@amd.com>)
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

// HIP replacement for k2/csrc/cudpp/cudpp.cu. The vendored CUDPP segmented scan
// is warp-synchronous (WARP_SIZE=32, two 32-lane warps per block) and so is a
// wave64 hazard on CDNA. SegmentedExclusiveSum is exactly a per-segment
// exclusive prefix sum, which hipCUB provides natively via
// DeviceScan::ExclusiveScanByKey. We turn the head-flags array into a monotone
// segment key (inclusive scan of the flags: the key increments by one at each
// segment start, so the per-key scan resets exactly at segment boundaries) and
// run an exclusive Sum scan keyed on it. Identical semantics, no warp
// arithmetic, correct on wave32 and wave64.

#include <hipcub/hipcub.hpp>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/cudpp/cudpp.h"
#include "k2/csrc/log.h"

namespace k2 {

template <typename T>
void SegmentedExclusiveSum(ContextPtr context, const T *d_in,
                           int32_t num_elements, const uint32_t *d_iflags,
                           T *d_out) {
  if (num_elements <= 0) return;
  hipStream_t stream = context->GetCudaStream();

  // keys[i] = inclusive sum of flags[0..i]; the key changes (by +1) exactly at
  // each segment start, which is where ExclusiveScanByKey must reset.
  Array1<uint32_t> keys(context, num_elements);
  uint32_t *keys_data = keys.Data();

  size_t temp_bytes = 0;
  K2_CHECK_EQ(hipcub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_iflags,
                                               keys_data, num_elements, stream),
              hipSuccess);
  {
    Array1<int8_t> d_temp(context, static_cast<int32_t>(temp_bytes));
    K2_CHECK_EQ(
        hipcub::DeviceScan::InclusiveSum(d_temp.Data(), temp_bytes, d_iflags,
                                         keys_data, num_elements, stream),
        hipSuccess);
  }

  temp_bytes = 0;
  K2_CHECK_EQ(
      hipcub::DeviceScan::ExclusiveScanByKey(
          nullptr, temp_bytes, keys_data, d_in, d_out, hipcub::Sum(), T(0),
          num_elements, hipcub::Equality(), stream),
      hipSuccess);
  {
    Array1<int8_t> d_temp(context, static_cast<int32_t>(temp_bytes));
    K2_CHECK_EQ(
        hipcub::DeviceScan::ExclusiveScanByKey(
            d_temp.Data(), temp_bytes, keys_data, d_in, d_out, hipcub::Sum(),
            T(0), num_elements, hipcub::Equality(), stream),
        hipSuccess);
  }
}

template void SegmentedExclusiveSum<int32_t>(ContextPtr context,
                                             const int32_t *d_in,
                                             int32_t num_elements,
                                             const uint32_t *d_iflags,
                                             int32_t *d_out);

template void SegmentedExclusiveSum<float>(ContextPtr context,
                                           const float *d_in,
                                           int32_t num_elements,
                                           const uint32_t *d_iflags,
                                           float *d_out);

template void SegmentedExclusiveSum<double>(ContextPtr context,
                                            const double *d_in,
                                            int32_t num_elements,
                                            const uint32_t *d_iflags,
                                            double *d_out);

}  // namespace k2
