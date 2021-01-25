// This file is modified from
// cudpp/src/cudpp/kernel/vector_kernel.cuh
//
// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 5636 $
//  $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

#ifndef K2_CSRC_CUDPP_VECTOR_KERNEL_H_
#define K2_CSRC_CUDPP_VECTOR_KERNEL_H_

#include "k2/csrc/cudpp/cudpp_util.h"

/** @brief Add a uniform value to data elements of an array (vec4 version)
 *
 * This function reads one value per CTA from \a d_uniforms into shared
 * memory and adds that value to values "owned" by the CTA in \a d_vector.
 * The uniform value is added to only those values "owned" by the CTA which
 * have an index less than d_maxIndex. If d_maxIndex for that CTA is UINT_MAX
 * it adds the uniform to all values "owned" by the CTA.
 * Each thread adds the uniform value to eight values in \a d_vector.
 *
 * @param[out] d_vector The d_vector whose values will have the uniform added
 * @param[in] d_uniforms The array of uniform values (one per CTA)
 * @param[in] d_maxIndices The array of maximum indices (one per CTA). This is
 *            index upto which the uniform would be added. If this is UINT_MAX
 *            the uniform is added to all elements of the CTA. This index is
 *            1-based.
 * @param[in] numElements The number of elements in \a d_vector to process
 * @param[in] blockOffset an optional offset to the beginning of this block's
 * data.
 * @param[in] baseIndex an optional offset to the beginning of the array
 * within \a d_vector.
 */
template <class T, class Oper, bool isLastBlockFull>
__global__ void vectorSegmentedAddUniform4(T *d_vector, const T *d_uniforms,
                                           const unsigned int *d_maxIndices,
                                           unsigned int numElements,
                                           int blockOffset, int baseIndex) {
  __shared__ T uni[2];

  unsigned int blockAddress =
      blockIdx.x + __mul24(gridDim.x, blockIdx.y) + blockOffset;

  // Get this block's uniform value from the uniform array in device memory
  // We store it in shared memory so that the hardware's shared memory
  // broadcast capability can be used to share among all threads in each warp
  // in a single cycle

  // instantiate operator functor
  Oper op;

  if (threadIdx.x == 0) {
    if (blockAddress > 0)
      uni[0] = d_uniforms[blockAddress - 1];
    else
      uni[0] = op.identity();

    // Tacit assumption that T is four-byte wide
    *((unsigned int *)(uni + 1)) = d_maxIndices[blockAddress];
  }

  // Compute this thread's output address
  int width = __mul24(gridDim.x, (blockDim.x << 1));

  unsigned int address = baseIndex + __mul24(width, blockIdx.y) + threadIdx.x +
                         __mul24(blockIdx.x, (blockDim.x << 3));

  __syncthreads();

  unsigned int maxIndex = *((unsigned int *)(uni + 1));

  bool isLastBlock = (blockIdx.x == (gridDim.x - 1));

  if (maxIndex < UINT_MAX) {
    // Since maxIndex is a 1 based index
    --maxIndex;
    bool leftLess = address < maxIndex;
    bool rightLess = (address + 7 * blockDim.x) < maxIndex;

    if (leftLess) {
      if (rightLess) {
        for (unsigned int i = 0; i < 8; ++i)
          d_vector[address + i * blockDim.x] =
              op(d_vector[address + i * blockDim.x], uni[0]);
      } else {
        for (unsigned int i = 0; i < 8; ++i) {
          if (address < maxIndex)
            d_vector[address] = op(d_vector[address], uni[0]);

          address += blockDim.x;
        }
      }
    }
  } else {
    if (!isLastBlockFull && isLastBlock) {
      for (unsigned int i = 0; i < 8; ++i) {
        if (address < numElements)
          d_vector[address] = op(d_vector[address], uni[0]);

        address += blockDim.x;
      }
    } else {
      for (unsigned int i = 0; i < 8; ++i) {
        d_vector[address] = op(d_vector[address], uni[0]);

        address += blockDim.x;
      }
    }
  }
}

/** @brief Add a uniform value to data elements of an array (vec4 version)
 *
 * This function reads one value per CTA from \a d_uniforms into shared
 * memory and adds that value to values "owned" by the CTA in \a d_vector.
 * The uniform value is added to only those values "owned" by the CTA which
 * have an index greater than d_minIndex. If d_minIndex for that CTA is 0
 * it adds the uniform to all values "owned" by the CTA.
 * Each thread adds the uniform value to eight values in \a d_vector.
 *
 * @param[out] d_vector The d_vector whose values will have the uniform added
 * @param[in] d_uniforms The array of uniform values (one per CTA)
 * @param[in] d_minIndices The array of minimum indices (one per CTA). The
 *            uniform is added to the right of this index (that is, to every
 * index that is greater than this index). If this is 0, the uniform is added to
 * all elements of the CTA. This index is 1-based to prevent overloading of what
 * 0 means. In our case it means absence of a flag. But if the first element of
 * a CTA has flag the index will also be 0. Hence we use 1-based indices so the
 * index is 1 in the latter case.
 * @param[in] numElements The number of elements in \a d_vector to process
 * @param[in] blockOffset an optional offset to the beginning of this block's
 * data.
 * @param[in] baseIndex an optional offset to the beginning of the array
 * within \a d_vector.
 *
 */
template <class T, class Oper, bool isLastBlockFull>
__global__ void vectorSegmentedAddUniformToRight4(
    T *d_vector, const T *d_uniforms, const unsigned int *d_minIndices,
    unsigned int numElements, int blockOffset, int baseIndex) {
  __shared__ T uni[2];

  unsigned int blockAddress =
      blockIdx.x + __mul24(gridDim.x, blockIdx.y) + blockOffset;

  // instantiate operator functor
  Oper op;

  // Get this block's uniform value from the uniform array in device memory
  // We store it in shared memory so that the hardware's shared memory
  // broadcast capability can be used to share among all threads in each warp
  // in a single cycle

  if (threadIdx.x == 0) {
    // FIXME - blockAddress test here is incompatible with how it is calculated
    // above
    if (blockAddress < (gridDim.x - 1))
      uni[0] = d_uniforms[blockAddress + 1];
    else
      uni[0] = op.identity();

    // Tacit assumption that T is four-byte wide
    *((unsigned int *)(uni + 1)) = d_minIndices[blockAddress];
  }

  // Compute this thread's output address
  int width = __mul24(gridDim.x, (blockDim.x << 1));

  unsigned int address = baseIndex + __mul24(width, blockIdx.y) + threadIdx.x +
                         __mul24(blockIdx.x, (blockDim.x << 3));

  __syncthreads();

  unsigned int minIndex = *((unsigned int *)(uni + 1));

  bool isLastBlock = (blockIdx.x == (gridDim.x - 1));

  if (minIndex > 0) {
    // Since minIndex is a 1 based index
    --minIndex;
    bool leftInRange = address > minIndex;
    bool rightInRange = (address + 7 * blockDim.x) > minIndex;

    if (rightInRange) {
      if (leftInRange) {
        if (!isLastBlockFull && isLastBlock) {
          for (unsigned int i = 0; i < 8; ++i) {
            if (address < numElements)
              d_vector[address] = op(d_vector[address], uni[0]);
            address += blockDim.x;
          }
        } else {
          for (unsigned int i = 0; i < 8; ++i) {
            d_vector[address] = op(d_vector[address], uni[0]);
            address += blockDim.x;
          }
        }
      } else {
        if (!isLastBlockFull && isLastBlock) {
          for (unsigned int i = 0; i < 8; ++i) {
            if (address > minIndex && address < numElements)
              d_vector[address] = op(d_vector[address], uni[0]);

            address += blockDim.x;
          }
        } else {
          for (unsigned int i = 0; i < 8; ++i) {
            if (address > minIndex)
              d_vector[address] = op(d_vector[address], uni[0]);

            address += blockDim.x;
          }
        }
      }
    }
  } else {
    if (!isLastBlockFull && isLastBlock) {
      for (unsigned int i = 0; i < 8; ++i) {
        if (address < numElements)
          d_vector[address] = op(d_vector[address], uni[0]);

        address += blockDim.x;
      }
    } else {
      for (unsigned int i = 0; i < 8; ++i) {
        d_vector[address] = op(d_vector[address], uni[0]);

        address += blockDim.x;
      }
    }
  }
}
#endif  // K2_CSRC_CUDPP_VECTOR_KERNEL_H_
