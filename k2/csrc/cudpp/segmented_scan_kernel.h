// This file is from
// cudpp/src/cudpp/kernel/segmented_scan_kernel.cuh
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

#ifndef K2_CSRC_CUDPP_SEGMENTED_SCAN_KERNEL_H_
#define K2_CSRC_CUDPP_SEGMENTED_SCAN_KERNEL_H_

#include "k2/csrc/cudpp/segmented_scan_cta.h"
#include "k2/csrc/cudpp/cudpp_util.h"

namespace k2 {

/**
* @brief Main segmented scan kernel
*
* This __global__ device function performs one level of a multiblock
* segmented scan on an one-dimensioned array in \a d_idata, returning the
* result in \a d_odata (which may point to the same array).
*
* This function performs one level of a recursive, multiblock scan.  At the
* app level, this function is called by cudppSegmentedScan and used in combination
* with either vectorSegmentedAddUniform4() (forward) or
* vectorSegmentedAddUniformToRight4() (backward) to produce a complete segmented scan.
*
* Template parameter \a T is the datatype of the array to be scanned.
* Template parameter \a traits is the SegmentedScanTraits struct containing
* compile-time options for the segmented scan, such as whether it is forward
* or backward, inclusive or exclusive, etc.
*
* @param[out] d_odata The output (scanned) array
* @param[in] d_idata The input array to be scanned
* @param[in] d_iflags The input array of flags
* @param[out] d_blockSums The array of per-block sums
* @param[out] d_blockFlags The array of per-block OR-reduction of flags
* @param[out] d_blockIndices The array of per-block min-reduction of indices
* @param[in] numElements The number of elements to scan
*/
template <class T, class traits>
__global__
void segmentedScan4(T                  *d_odata,
                    const T            *d_idata,
                    const unsigned int *d_iflags,
                    unsigned int       numElements,
                    T                  *d_blockSums=0,
                    unsigned int       *d_blockFlags=0,
                    unsigned int       *d_blockIndices=0
                    )
{
    SharedMemory<T> smem;
    T* temp = smem.getPointer();

    int ai, bi, aiDev, biDev;

    // Chop up the shared memory into 4 contiguous spaces - the first
    // for the data, the second for the indices, the third for the
    // read-only version of the flags and the last for the read-write
    // version of the flags
    unsigned int* indices = (unsigned int *)(temp + 2*blockDim.x);
    unsigned int* flags   = (unsigned int *)(indices + 2*blockDim.x);

    T threadScan0[4];
    T threadScan1[4];
    unsigned int threadFlag = 0;

    int devOffset = blockIdx.x * (2 * blockDim.x);

    // load data into shared memory
    loadForSegmentedScanSharedChunkFromMem4<T, traits>(
        temp, threadScan0, threadScan1, threadFlag,
        flags, indices, d_idata, d_iflags,
        numElements, devOffset, ai, bi, aiDev, biDev);

    segmentedScanCTA<T, traits>(
        temp, flags, indices,
        d_blockSums, d_blockFlags, d_blockIndices);

    // write results to device memory
    storeForSegmentedScanSharedChunkToMem4<T, traits>(
        d_odata, threadScan0, threadScan1, threadFlag,
        temp, numElements,  devOffset, ai, bi, aiDev, biDev);
}

}  // namespace k2

#endif  // K2_CSRC_CUDPP_SEGMENTED_SCAN_KERNEL_H_
