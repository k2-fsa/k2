/**
 * k2/csrc/cudpp/cudpp.cu
 *
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

// This file is modified from CUDPP. The code style is kept
// largely the same with CUDPP, which is different from what k2 is using.
//
// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Revision: 3505 $
//  $Date: 2007-07-06 09:26:06 -0700 (Fri, 06 Jul 2007) $
//  -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <memory>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/cudpp/cudpp_util.h"
#include "k2/csrc/cudpp/segmented_scan_cta.h"
#include "k2/csrc/cudpp/segmented_scan_kernel.h"
#include "k2/csrc/cudpp/vector_kernel.h"
#include "k2/csrc/log.h"

namespace k2 {

struct SegmentedScanPlan {
 public:
  explicit SegmentedScanPlan(int32_t num_elements,
                             int32_t element_size_in_bytes, ContextPtr c)
      : num_elements(num_elements),
        element_size_in_bytes(element_size_in_bytes) {
    Allocate(c);
  }

  void Allocate(ContextPtr c) {
    int32_t numElts = num_elements;

    int32_t level = 0;

    do {
      int32_t numBlocks =
          max(1, (int32_t)ceil((double)numElts /
                               (SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
      if (numBlocks > 1) {
        level++;
      }
      numElts = numBlocks;
    } while (numElts > 1);

    block_sums.reset(new int8_t *[level]);
    block_flags.reset(new uint32_t *[level]);
    block_indexes.reset(new uint32_t *[level]);

    numElts = num_elements;

    level = 0;

    do {
      int32_t numBlocks =
          max(1, (int32_t)ceil((double)numElts /
                               (SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
      if (numBlocks > 1) {
        buf_i8.push_back(Array1<int8_t>(c, numBlocks * element_size_in_bytes));
        block_sums[level] = buf_i8.back().Data();

        buf_ui32.push_back(Array1<uint32_t>(c, numBlocks));
        block_flags[level] = buf_ui32.back().Data();

        buf_ui32.push_back(Array1<uint32_t>(c, numBlocks));
        block_indexes[level] = buf_ui32.back().Data();

        level++;
      }
      numElts = numBlocks;
    } while (numElts > 1);
  }

  // Intermediate block sums array
  std::unique_ptr<int8_t *[]> block_sums;

  // Intermediate block flags array
  std::unique_ptr<uint32_t *[]> block_flags;

  // Intermediate block indexes array
  std::unique_ptr<uint32_t *[]> block_indexes;

  int32_t num_elements;
  int32_t element_size_in_bytes;

  std::vector<Array1<int8_t>> buf_i8;
  std::vector<Array1<uint32_t>> buf_ui32;
};

template <typename T, class Op, bool isBackward, bool isExclusive,
          bool doShiftFlagsLeft>
static void SegmentedScanArrayRecursive(
    ContextPtr context, T *d_out, const T *d_idata, const uint32_t *d_iflags,
    T **d_blockSums, uint32_t **block_flags, uint32_t **block_indexes,
    int num_elements, int level, bool sm12OrBetterHw) {
  int32_t numBlocks =
      max(1, (int32_t)std::ceil((double)num_elements /
                                (SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

  // This is the number of elements per block that the
  // CTA level API is aware of
  uint32_t numEltsPerBlock = SCAN_CTA_SIZE * 2;

  // Space to store flags - we need two sets. One gets modified and the
  // other doesn't
  uint32_t flagSpace = numEltsPerBlock * sizeof(uint32_t);

  // Space to store indexes
  uint32_t idxSpace = numEltsPerBlock * sizeof(uint32_t);

  // Total shared memory space
  uint32_t sharedMemSize = sizeof(T) * (numEltsPerBlock) + idxSpace + flagSpace;

  // setup execution parameters
  dim3 grid(max(1, numBlocks), 1, 1);
  dim3 threads(SCAN_CTA_SIZE, 1, 1);

  bool fullBlock =
      (num_elements == (numBlocks * SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE));

  uint32_t traitsCode = 0;
  if (numBlocks > 1) traitsCode |= 1;
  if (fullBlock) traitsCode |= 2;
  if (sm12OrBetterHw) traitsCode |= 4;

  cudaStream_t stream = context->GetCudaStream();

  switch (traitsCode) {
    case 0:  // single block, single row, non-full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, false, false, false>>
          <<<grid, threads, sharedMemSize, stream>>>(d_out, d_idata, d_iflags,
                                                     num_elements, 0, 0, 0));
      break;
    case 1:  // multi block, single row, non-full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, false, true, false>>
          <<<grid, threads, sharedMemSize, stream>>>(
              d_out, d_idata, d_iflags, num_elements, d_blockSums[level],
              block_flags[level], block_indexes[level]));
      break;
    case 2:  // single block, single row, full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, true, false, false>>
          <<<grid, threads, sharedMemSize, stream>>>(d_out, d_idata, d_iflags,
                                                     num_elements, 0, 0, 0));
      break;
    case 3:  // multi block, single row, full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, true, true, false>>
          <<<grid, threads, sharedMemSize, stream>>>(
              d_out, d_idata, d_iflags, num_elements, d_blockSums[level],
              block_flags[level], block_indexes[level]));
      break;
    case 4:  // single block, single row, non-full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, false, false, true>>
          <<<grid, threads, sharedMemSize, stream>>>(d_out, d_idata, d_iflags,
                                                     num_elements, 0, 0, 0));
      break;
    case 5:  // multi block, single row, non-full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, false, true, true>>
          <<<grid, threads, sharedMemSize, stream>>>(
              d_out, d_idata, d_iflags, num_elements, d_blockSums[level],
              block_flags[level], block_indexes[level]));
      break;
    case 6:  // single block, single row, full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, true, false, true>>
          <<<grid, threads, sharedMemSize, stream>>>(d_out, d_idata, d_iflags,
                                                     num_elements, 0, 0, 0));
      break;
    case 7:  // multi block, single row, full last block
      K2_CUDA_SAFE_CALL(
          segmentedScan4<
              T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                     doShiftFlagsLeft, true, true, true>>
          <<<grid, threads, sharedMemSize, stream>>>(
              d_out, d_idata, d_iflags, num_elements, d_blockSums[level],
              block_flags[level], block_indexes[level]));
      break;
  }

  if (numBlocks > 1) {
    // After scanning all the sub-blocks, we are mostly done. But
    // now we need to take all of the last values of the
    // sub-blocks and segment scan those. This will give us a new value
    // that must be sdded to the first segment of each block to get
    // the final results.
    SegmentedScanArrayRecursive<T, Op, isBackward, false, false>(
        context, (T *)d_blockSums[level], (const T *)d_blockSums[level],
        block_flags[level], (T **)d_blockSums, block_flags, block_indexes,
        numBlocks, level + 1, sm12OrBetterHw);

    if (isBackward) {
      if (fullBlock)
        K2_CUDA_SAFE_CALL(vectorSegmentedAddUniformToRight4<T, Op, true>
                          <<<grid, threads, 0, stream>>>(
                              d_out, d_blockSums[level], block_indexes[level],
                              num_elements, 0, 0));
      else
        K2_CUDA_SAFE_CALL(vectorSegmentedAddUniformToRight4<T, Op, false>
                          <<<grid, threads, 0, stream>>>(
                              d_out, d_blockSums[level], block_indexes[level],
                              num_elements, 0, 0));
    } else {
      if (fullBlock)
        K2_CUDA_SAFE_CALL(vectorSegmentedAddUniform4<T, Op, true>
                          <<<grid, threads, 0, stream>>>(
                              d_out, d_blockSums[level], block_indexes[level],
                              num_elements, 0, 0));
      else
        K2_CUDA_SAFE_CALL(vectorSegmentedAddUniform4<T, Op, false>
                          <<<grid, threads, 0, stream>>>(
                              d_out, d_blockSums[level], block_indexes[level],
                              num_elements, 0, 0));
    }
  }
}

template <typename T>
void SegmentedExclusiveSum(ContextPtr context, const T *d_in,
                           int32_t num_elements, const uint32_t *d_iflags,
                           T *d_out) {
  SegmentedScanPlan plan(num_elements, sizeof(T), context);

  SegmentedScanArrayRecursive<T, OperatorAdd<T>, false /*isBackward*/,
                              true /*isExclusive*/, false /*isBackward*/>(
      context, d_out, d_in, d_iflags,
      reinterpret_cast<T **>(plan.block_sums.get()), plan.block_flags.get(),
      plan.block_indexes.get(), num_elements, 0, true /*sm12OrBetterHw*/);
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
