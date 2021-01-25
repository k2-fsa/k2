/**
 * k2/csrc/cudpp/cudpp.cu
 *
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <cmath>

#define SEGSCAN_ELTS_PER_THREAD 8
#define SCAN_CTA_SIZE 128
#define LOG_SCAN_CTA_SIZE 7
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#include "k2/csrc/cudpp/cudpp_util.h"
#include "k2/csrc/cudpp/segmented_scan_cta.h"
#include "k2/csrc/cudpp/segmented_scan_kernel.h"
#include "k2/csrc/cudpp/vector_kernel.h"
#include "k2/csrc/log.h"

namespace k2 {

struct CUDPPSegmentedScanPlan {
 public:
  explicit CUDPPSegmentedScanPlan(size_t numElements,
                                  size_t element_size_in_bytes)
      : num_elements(numElements),
        element_size_in_bytes(element_size_in_bytes) {
    Allocate();
  }

  ~CUDPPSegmentedScanPlan() { Deallocate(); }

  void Allocate() {
    size_t numElts = num_elements;

    size_t level = 0;

    do {
      size_t numBlocks = std::max(
          (size_t)1,
          (size_t)std::ceil((double)numElts /
                            ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
      if (numBlocks > 1) {
        level++;
      }
      numElts = numBlocks;
    } while (numElts > 1);

    m_blockSums = (void **)malloc(level * sizeof(void *));

    m_blockFlags = (unsigned int **)malloc(level * sizeof(unsigned int *));
    m_blockIndices = (unsigned int **)malloc(level * sizeof(unsigned int *));

    m_numLevelsAllocated = level;

    numElts = num_elements;

    level = 0;

    do {
      size_t numBlocks = std::max(
          (size_t)1,
          (size_t)std::ceil((double)numElts /
                            ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
      if (numBlocks > 1) {
        K2_CUDA_SAFE_CALL(cudaMalloc((void **)&(m_blockSums[level]),
                                     numBlocks * element_size_in_bytes));

        K2_CUDA_SAFE_CALL(cudaMalloc((void **)&(m_blockFlags[level]),
                                     numBlocks * sizeof(unsigned int)));

        K2_CUDA_SAFE_CALL(cudaMalloc((void **)&(m_blockIndices[level]),
                                     numBlocks * sizeof(unsigned int)));
        level++;
      }
      numElts = numBlocks;
    } while (numElts > 1);
  }

  void Deallocate() {
    for (unsigned int i = 0; i < m_numLevelsAllocated; ++i) {
      K2_CUDA_SAFE_CALL(cudaFree(m_blockSums[i]));
      K2_CUDA_SAFE_CALL(cudaFree(m_blockFlags[i]));
      K2_CUDA_SAFE_CALL(cudaFree(m_blockIndices[i]));
    }

    free((void **)m_blockSums);
    free((void **)m_blockFlags);
    free((void **)m_blockIndices);

    m_blockSums = nullptr;
    m_blockFlags = nullptr;
    m_blockIndices = nullptr;
    m_numLevelsAllocated = 0;
  }

  // Intermediate block sums array
  void **m_blockSums = nullptr;
  // Intermediate block flags array
  unsigned int **m_blockFlags = nullptr;

  // Intermediate block indices array
  unsigned int **m_blockIndices = nullptr;

  // Number of levels allocaed (in _scanBlockSums)
  size_t m_numLevelsAllocated = 0;

  size_t num_elements;
  size_t element_size_in_bytes;
};

template <typename T, class Op, bool isBackward, bool isExclusive,
          bool doShiftFlagsLeft>
void segmentedScanArrayRecursive(T *d_out, const T *d_idata,
                                 const unsigned int *d_iflags, T **d_blockSums,
                                 unsigned int **d_blockFlags,
                                 unsigned int **d_blockIndices, int numElements,
                                 int level, bool sm12OrBetterHw) {
  unsigned int numBlocks =
      max(1, (int)ceil((double)numElements /
                       ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

  // This is the number of elements per block that the
  // CTA level API is aware of
  unsigned int numEltsPerBlock = SCAN_CTA_SIZE * 2;

  // Space to store flags - we need two sets. One gets modified and the
  // other doesn't
  unsigned int flagSpace = numEltsPerBlock * sizeof(unsigned int);

  // Space to store indices
  unsigned int idxSpace = numEltsPerBlock * sizeof(unsigned int);

  // Total shared memory space
  unsigned int sharedMemSize =
      sizeof(T) * (numEltsPerBlock) + idxSpace + flagSpace;

  // setup execution parameters
  dim3 grid(max(1, numBlocks), 1, 1);
  dim3 threads(SCAN_CTA_SIZE, 1, 1);

  bool fullBlock =
      (numElements == (numBlocks * SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE));

  unsigned int traitsCode = 0;
  if (numBlocks > 1) traitsCode |= 1;
  if (fullBlock) traitsCode |= 2;
  if (sm12OrBetterHw) traitsCode |= 4;

  switch (traitsCode) {
    case 0:  // single block, single row, non-full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, false, false, false>>
          <<<grid, threads, sharedMemSize>>>(d_out, d_idata, d_iflags,
                                             numElements, 0, 0, 0);
      break;
    case 1:  // multi block, single row, non-full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, false, true, false>>
          <<<grid, threads, sharedMemSize>>>(
              d_out, d_idata, d_iflags, numElements, d_blockSums[level],
              d_blockFlags[level], d_blockIndices[level]);
      break;
    case 2:  // single block, single row, full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, true, false, false>>
          <<<grid, threads, sharedMemSize>>>(d_out, d_idata, d_iflags,
                                             numElements, 0, 0, 0);
      break;
    case 3:  // multi block, single row, full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, true, true, false>>
          <<<grid, threads, sharedMemSize>>>(
              d_out, d_idata, d_iflags, numElements, d_blockSums[level],
              d_blockFlags[level], d_blockIndices[level]);
      break;
    case 4:  // single block, single row, non-full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, false, false, true>>
          <<<grid, threads, sharedMemSize>>>(d_out, d_idata, d_iflags,
                                             numElements, 0, 0, 0);
      break;
    case 5:  // multi block, single row, non-full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, false, true, true>>
          <<<grid, threads, sharedMemSize>>>(
              d_out, d_idata, d_iflags, numElements, d_blockSums[level],
              d_blockFlags[level], d_blockIndices[level]);
      break;
    case 6:  // single block, single row, full last block
      segmentedScan4<T,
                     SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                         doShiftFlagsLeft, true, false, true>>
          <<<grid, threads, sharedMemSize>>>(d_out, d_idata, d_iflags,
                                             numElements, 0, 0, 0);
      break;
    case 7:  // multi block, single row, full last block
      segmentedScan4<T, SegmentedScanTraits<T, Op, isBackward, isExclusive,
                                            doShiftFlagsLeft, true, true, true>>
          <<<grid, threads, sharedMemSize>>>(
              d_out, d_idata, d_iflags, numElements, d_blockSums[level],
              d_blockFlags[level], d_blockIndices[level]);
      break;
  }

  if (numBlocks > 1) {
    // After scanning all the sub-blocks, we are mostly done. But
    // now we need to take all of the last values of the
    // sub-blocks and segment scan those. This will give us a new value
    // that must be sdded to the first segment of each block to get
    // the final results.
    segmentedScanArrayRecursive<T, Op, isBackward, false, false>(
        (T *)d_blockSums[level], (const T *)d_blockSums[level],
        d_blockFlags[level], (T **)d_blockSums, d_blockFlags, d_blockIndices,
        numBlocks, level + 1, sm12OrBetterHw);

    if (isBackward) {
      if (fullBlock)
        vectorSegmentedAddUniformToRight4<T, Op, true>
            <<<grid, threads>>>(d_out, d_blockSums[level],
                                d_blockIndices[level], numElements, 0, 0);
      else
        vectorSegmentedAddUniformToRight4<T, Op, false>
            <<<grid, threads>>>(d_out, d_blockSums[level],
                                d_blockIndices[level], numElements, 0, 0);
    } else {
      if (fullBlock)
        vectorSegmentedAddUniform4<T, Op, true>
            <<<grid, threads>>>(d_out, d_blockSums[level],
                                d_blockIndices[level], numElements, 0, 0);
      else
        vectorSegmentedAddUniform4<T, Op, false>
            <<<grid, threads>>>(d_out, d_blockSums[level],
                                d_blockIndices[level], numElements, 0, 0);
    }
  }
}

void cudppSegmentedScan(void *d_out, const void *d_in,
                        const unsigned int *d_iflags, size_t numElements) {
  // TODO(fangjun): support other types
  CUDPPSegmentedScanPlan plan(numElements, sizeof(int32_t));

  using T = int32_t;
  segmentedScanArrayRecursive<T, OperatorAdd<T>, false /*isBackward*/,
                              true /*isExclusive*/, false /*isBackward*/>(
      (T *)d_out, (const T *)d_in, d_iflags, (T **)plan.m_blockSums,
      plan.m_blockFlags, plan.m_blockIndices, numElements, 0,
      true /*sm12OrBetterHw*/);
}

}  // namespace k2
