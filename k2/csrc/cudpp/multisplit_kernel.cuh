// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp_multisplit.h"
#include <cudpp_globals.h>
#include <cudpp_util.h>
#include "sharedmem.h"

typedef unsigned int uint;

#ifdef WIN32

typedef _int32 int32_t;
typedef unsigned _int32 uint32_t;
typedef _int64 int64_t;
typedef unsigned _int64 uint64_t;
typedef unsigned _int64 uint64;

#else

#include <stdint.h>
typedef unsigned long long int uint64;

#endif

/**
 * @file
 * multisplit_kernel.cu
 *   
 * @brief CUDPP kernel-level multisplit routines
 */

/** \addtogroup cudpp_kernel
 * @{
 */

/** @name Multisplit Functions
 * @{
 */

//==========================================
template<class T>
__global__ void markBins_general(uint* d_mark, uint* d_elements,
    uint numElements, uint numBuckets, T bucketMapper) {

  unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int offset = blockDim.x * gridDim.x;
  unsigned int logBuckets = ceil(log2((float) numBuckets));

  for (int i = myId; i < numElements; i += offset) {
    unsigned int myVal = d_elements[i];
    unsigned int myBucket = bucketMapper(myVal);
    d_mark[i] = myBucket;
  }
}
//===========================================
__global__ void packingKeyValuePairs(uint64* packed, uint* input_key,
    uint* input_value, uint numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint myKey = input_key[tid];
  uint myValue = input_value[tid];
  // putting the key as the more significant 32 bits.
  uint64 output = (static_cast<uint64>(myKey) << 32)
      + static_cast<uint>(myValue);
  packed[tid] = output;
}
//===========================================
__global__ void unpackingKeyValuePairs(uint64* packed, uint* out_key,
    uint* out_value, uint numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint64 myPacked = packed[tid];
  out_value[tid] = static_cast<uint>(myPacked & 0x00000000FFFFFFFF);
  out_key[tid] = static_cast<uint>(myPacked >> 32);
}
//=========================================================================
// Compaction based on Warp-level Multisplit
//=========================================================================
#ifndef __MULTISPLIT_COMPACTION_CUH_
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void histogram_pre_scan_compaction(key_t* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  // The new warp-level histogram computing kernel:
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[(NUM_W * DEPTH) << 1];

  if(blockIdx.x == (gridDim.x - 1)) // last block, potentially may try to read invalid inputs
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < (2 * NUM_W * DEPTH))
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();


    // === computing the histogram:
    key_t   myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      bool valid_input = false;

      // == reading the input only if valid:
      if((global_index + laneId) < numElements)
      {
        myInput[kk] = input[global_index + laneId];
        valid_input = true;
      }
      // masking out those threads which read an invalid key
      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // computing histogram
      uint32_t rx_buffer = __ballot_sync(0xFFFFFFFF, bucket_identifier(myInput[kk]));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      binCounter[kk] = __popc(myHisto & mask);

      // === storing the results into the shared memory
      if(laneId < 2)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
    }
    __syncthreads();

    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (2 * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
  else // all other blocks, that we are sure they are certainly processing valid inputs:
  {
    // === computing the histogram:
    key_t   myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);

      myInput[kk] = input[global_index + laneId];

      uint32_t rx_buffer = __ballot_sync(0xFFFFFFFF, bucket_identifier(myInput[kk]));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      binCounter[kk] = __popc(myHisto);

      // === storing the results into the shared memory
      if(laneId < 2)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
      __syncthreads();
    }
    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (2 * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
}
//==================================
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void split_post_scan_compaction(key_t* key_input, uint32_t* warpOffsets, key_t* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[2 * NUM_W * DEPTH + 32 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_t* keys_ms_smem = &warp_offsets_smem[2 * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // memory hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < 2*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  key_t   myInput[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myBucket = (bucket_identifier(myInput[kk]))?1u:0u;
      }

      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // Computing the histogram and local indices:
      uint32_t rx_buffer = __ballot_sync(0xFFFFFFFF, myBucket);
      uint32_t myMask  = 0xFFFFFFFF & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      scan_histo[kk] = binCounter[kk];
      uint32_t n = __shfl_sync(0xffffffff, scan_histo[kk], 0, 2);
      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_t myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * bucket_identifier(myNewKey) + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], bucket_identifier(myNewKey), 32);

      if(valid_input)
        key_output[finalIndex] = myNewKey;
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];

      // Computing the histogram and local indices:

      uint32_t myBucket = bucket_identifier(myInput[kk]);
      uint32_t rx_buffer = __ballot_sync(0xffffffff, myBucket);
      uint32_t myMask  = 0xFFFFFFFF  & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);

      scan_histo[kk] = binCounter[kk];

      uint32_t n = __shfl_sync(0xffffffff, scan_histo[kk], 0, 2);

      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_t myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];

      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
    }
  }
}
//=====================================================
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t, typename value_t>
__global__ void split_post_scan_pairs_compaction(key_t* key_input, value_t* value_input, uint32_t* warpOffsets, key_t* key_output, value_t* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[2 * NUM_W * DEPTH + 64 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_t* keys_ms_smem = &warp_offsets_smem[2 * NUM_W * DEPTH];
  value_t* values_ms_smem = &keys_ms_smem[32 * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // memory hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < 2*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  key_t   myInput[DEPTH];
  value_t   myValue[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myValue[kk] = value_input[global_index + laneId];

        myBucket = (bucket_identifier(myInput[kk]))?1u:0u;
      }

      uint32_t mask = __ballot_sync(0xffffffff, valid_input);

      // Computing the histogram and local indices:
      uint32_t rx_buffer = __ballot_sync(0xffffffff, myBucket);
      uint32_t myMask  = 0xFFFFFFFF & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      scan_histo[kk] = binCounter[kk];
      uint32_t n = __shfl_sync(0xffffffff, scan_histo[kk], 0, 2);
      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_t myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      value_t myNewValue = (valid_input)?values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * bucket_identifier(myNewKey) + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], bucket_identifier(myNewKey), 32);

      if(valid_input){
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];
      myValue[kk] = value_input[global_index + laneId];

      // Computing the histogram and local indices:

      uint32_t myBucket = bucket_identifier(myInput[kk]);
      uint32_t rx_buffer = __ballot_sync(0xffffffff, myBucket);
      uint32_t myMask  = 0xFFFFFFFF  & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);

      scan_histo[kk] = binCounter[kk];

      uint32_t n = __shfl_sync(0xffffffff, scan_histo[kk], 0, 2);

      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_t myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      value_t myNewValue = values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];

      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}

#endif
//=========================================================================
// Warp-level Multisplit GPU Kernels
//=========================================================================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_WMS_prescan(key_type* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH];

  if(blockIdx.x == (gridDim.x - 1)) // last block, potentially may try to read invalid inputs
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < NUM_B * NUM_W * DEPTH)
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if((global_index + laneId) < numElements)
      {
        myInput[kk] = input[global_index + laneId];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
    }
    __syncthreads();

    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (NUM_B * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
  else // all other blocks, that we are sure they are certainly processing valid inputs:
  {
    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      uint32_t myBucket = 0;

      myInput[kk] = input[global_index + laneId];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);

      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
      __syncthreads();
    }
    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (NUM_B * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
}
//==============================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_WMS_postscan(key_type* key_input, uint32_t* warpOffsets, key_type* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH + 32 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_type* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  uint32_t  myInput[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);
      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      // Computing the histogram and local indices:
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      // warp-wide inclusive scan
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up_sync(0xFFFFFFFF, scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      // making it exclusive scan.
      scan_histo[kk] -= binCounter[kk];

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_type myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);

      if(valid_input)
        key_output[finalIndex] = myNewKey;
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];
      uint32_t myBucket = bucket_identifier(myInput[kk]);

      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      // Computing the histogram and local indices:
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);
      // Inclusive scan:
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up_sync(0xFFFFFFFF, scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      //making it exclusive scan.
      scan_histo[kk] -= binCounter[kk];

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
    }
  }
}
//=============================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type, typename value_type>
__global__ void multisplit_WMS_pairs_postscan(key_type* key_input, value_type* value_input, uint32_t* warpOffsets, key_type* key_output, value_type* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH + 64 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_type* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];
  value_type* values_ms_smem = &warp_offsets_smem[32 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  uint32_t  myInput[DEPTH];
  uint32_t  myValue[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x - 1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myValue[kk] = value_input[global_index + laneId];
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t mask = __ballot_sync(0xffffffff, valid_input);
      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      // Computing the histogram and local indices:
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xffffffff, bit & 0x01);
        myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);

      // Warp-wide Inclusive scan
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up_sync(0xffffffff, scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }
    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_type myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      value_type myNewValue = (valid_input)?values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t myNewBucket = bucket_identifier(myNewKey);

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);

      if(valid_input){
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket;
      myInput[kk] = key_input[global_index + laneId];
      myValue[kk] = value_input[global_index + laneId];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      // Computing the histogram and local indices:
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xffffffff, bit & 0x01);
        myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      // warp-wide inclusive scan:
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up_sync(0xffffffff, scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl_sync(0xffffffff, scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      value_type myNewValue = values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;

      finalIndex -= __shfl_sync(0xffffffff, scan_histo[kk], myNewBucket, 32);

      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}
//=========================================================================
// Block-level Multisplit GPU Kernels
//=========================================================================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_prescan(key_type* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH];

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < NUM_B * NUM_W * DEPTH)
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if(global_index < numElements)
      {
        myInput[kk] = input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * NUM_W * NUM_B + warpId * NUM_B + laneId] = binCounter[kk];
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
    #pragma unroll
    for(int i = 1; i <= LOG_W; i++)
    {
      // Performing reduction over all elements:
      #pragma unroll
      for(int kk = 0; kk<DEPTH; kk++)
      {
        if((warpId & ((1<<i)-1)) == 0)
        {
          if(laneId < NUM_B){
            scratchPad[laneId + warpId * NUM_B + kk * NUM_W * NUM_B] += scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + kk * NUM_W * NUM_B];
          }
        }
      }
      __syncthreads();
    }

    // === storing the results back to the global memory with different hierarchy
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < DEPTH * NUM_B)
    {
      uint32_t bucket_src = tid % NUM_B;
      uint32_t roll_src   = tid / NUM_B;

      // global memory hierarchy: bucket -> block -> roll
      bin[bucket_src * gridDim.x * DEPTH + blockIdx.x * DEPTH + roll_src] = scratchPad[bucket_src + roll_src * NUM_W * NUM_B];
      tid += blockDim.x;
    }
  }
  else // all other blocks
  {
    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;

    // == reading the input only if valid:
      myInput[kk] = input[global_index];
      myBucket = bucket_identifier(myInput[kk]);
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // == computing warp-wide histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * NUM_W * NUM_B + warpId * NUM_B + laneId] = binCounter[kk];
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
    #pragma unroll
    for(int i = 1; i <= LOG_W; i++)
    {
      // Performing reduction over all elements:
      #pragma unroll
      for(int kk = 0; kk<DEPTH; kk++)
      {
        if((warpId & ((1<<i)-1)) == 0)
        {
          if(laneId < NUM_B){
            scratchPad[laneId + warpId * NUM_B + kk * NUM_W * NUM_B] += scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + kk * NUM_W * NUM_B];
          }
        }
      }
      __syncthreads();
    }

    // Global memory hierarchy: Bucket -> block -> roll
    if(NUM_W >= DEPTH)
    {
      if((laneId < NUM_B) && (warpId < DEPTH))
        bin[laneId * gridDim.x * DEPTH + blockIdx.x * DEPTH + warpId] = scratchPad[laneId + warpId * NUM_B * NUM_W];
    }
  }
}
//==================================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_postscan(key_type* key_input, uint32_t* blockOffsets, key_type* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  __shared__ uint32_t scratchPad[2 * NUM_B * DEPTH + 32 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];
  uint32_t* block_offsets_smem = scratchPad;
  uint32_t* warp_offsets_smem = &block_offsets_smem[NUM_B * DEPTH];
  uint32_t* scan_histo_smem = &warp_offsets_smem[NUM_B * DEPTH];
  key_type* keys_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH];

  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  uint32_t  myInput[DEPTH];
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_temp[DEPTH];

  // === Loading block offset results from global memory:
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < NUM_B * DEPTH)
  {
    uint32_t bucket_src = tid % NUM_B;
    uint32_t roll_src = tid / NUM_B;
    block_offsets_smem[bucket_src + roll_src * NUM_B] = blockOffsets[roll_src + blockIdx.x * DEPTH + bucket_src * DEPTH * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      uint32_t valid_input = false;
      if(global_index < numElements)
      {
        myInput[kk] = key_input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);

      // storing the results in smem in order to be scanned:
      // smem hierarchy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl_sync(0xffffffff, scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;
      myLocalBlockIndex = (valid_input)?myLocalBlockIndex:blockDim.x;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan:
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up_sync(0xffffffff, block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myInput[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements)?true:false;

      key_type myNewKey = (valid_input)?keys_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = 0;
      if(valid_input) {
        finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
        key_output[finalIndex] = myNewKey;
      }
    }
  }
  else // all others
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      myInput[kk] = key_input[global_index];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl_sync(0xffffffff, scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up_sync(0xffffffff, block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
      key_output[finalIndex] = myNewKey;
    }
  }
}
//===============================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type, typename value_type>
__global__ void multisplit_BMS_pairs_postscan(key_type* key_input, value_type* value_input, uint32_t* blockOffsets, key_type* key_output, value_type* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  __shared__ uint32_t scratchPad[2 * NUM_B * DEPTH + 64 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];
  uint32_t* block_offsets_smem = scratchPad;
  uint32_t* warp_offsets_smem = &block_offsets_smem[NUM_B * DEPTH];
  uint32_t* scan_histo_smem = &warp_offsets_smem[NUM_B * DEPTH];
  key_type* keys_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH];
  value_type* values_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH + 32 * NUM_W * DEPTH];

  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  uint32_t  myInput[DEPTH];
  uint32_t  myValue[DEPTH];
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_temp[DEPTH];

  // === Loading block offset results from global memory:
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < NUM_B * DEPTH)
  {
    uint32_t bucket_src = tid % NUM_B;
    uint32_t roll_src = tid / NUM_B;
    block_offsets_smem[bucket_src + roll_src * NUM_B] = blockOffsets[roll_src + blockIdx.x * DEPTH + bucket_src * DEPTH * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      uint32_t valid_input = false;
      if(global_index < numElements)
      {
        myInput[kk] = key_input[global_index];
        myValue[kk] = value_input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot_sync(0xFFFFFFFF, valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl_sync(0xffffffff, scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;
      myLocalBlockIndex = (valid_input)?myLocalBlockIndex:blockDim.x;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up_sync(0xffffffff, block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myInput[kk];
      values_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myValue[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements)?true:false;

      key_type myNewKey = (valid_input)?keys_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      value_type myNewValue = (valid_input)?values_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = 0;
      if(valid_input) {
        finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else // all others
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      myInput[kk] = key_input[global_index];
      myValue[kk] = value_input[global_index];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot_sync(0xFFFFFFFF, bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl_sync(0xffffffff, scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up_sync(0xFFFFFFFF, block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput[kk];
      values_ms_smem[myNewBlockIndex + kk * blockDim.x] = myValue[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      value_type myNewValue = values_ms_smem[threadIdx.x + kk * blockDim.x];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}

/** @} */// end Multisplit functions
/** @} */// end cudpp_kernel
