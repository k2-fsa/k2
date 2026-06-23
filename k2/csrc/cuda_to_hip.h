/**
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

// The single CUDA->HIP compat header for the ROCm port. It is force-included on
// every HIP translation unit (CMAKE_HIP_FLAGS -include .../cuda_to_hip.h), so
// the aliases below precede all other includes regardless of file order. It is
// the only file that knows about HIP; everywhere else keeps the CUDA spelling.
//
// On the NVIDIA build this header is never compiled (it is only force-included
// under K2_WITH_HIP), so the CUDA path is byte-for-byte unchanged.

#ifndef K2_CSRC_CUDA_TO_HIP_H_
#define K2_CSRC_CUDA_TO_HIP_H_

#if defined(K2_WITH_HIP)

// Pull in the libc host declarations BEFORE <hip/hip_runtime.h> so that inside
// a .cu compiled as HIP a host-side memcpy/memset resolves to the libc host
// overload rather than HIP's __device__ overload (gpuRIR lesson).
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>  // NOLINT(build/include_order)

// Note on device-vs-host dispatch under clang/HIP: k2 keys two different things
// on __CUDA_ARCH__: (a) the host/device DECORATOR K2_CUDA_HOSTDEV, and (b)
// intra-function `#ifdef __CUDA_ARCH__` device-intrinsic-vs-host dispatch.
// Under clang/HIP a __host__ __device__ function is preprocessed ONCE in the
// host pass (where __CUDA_ARCH__ is absent and a `#define` of it does not
// take), so we canNOT emulate (a) by defining __CUDA_ARCH__ (cudaKDTree
// lesson). Instead the decorator is unconditionally __host__ __device__ on
// HIP (log.h) and the (b) dispatch sites use K2_DEVICE_CODE (defined in
// macros.h), which keys on
// __HIP_DEVICE_COMPILE__ -- correct per-pass inside a __host__ __device__ body.

// ---------------------------------------------------------------------------
// Runtime types and enums
// ---------------------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaError hipError
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorAssert hipErrorAssert
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorInitializationError hipErrorNotInitialized

#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t

#define cudaDeviceProp hipDeviceProp_t

#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaEventDisableTiming hipEventDisableTiming

// ---------------------------------------------------------------------------
// Runtime API
// ---------------------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync

#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion

#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent

#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// ---------------------------------------------------------------------------
// CUB: k2 uses the CUDA spelling cub::. hipcub puts its API in the (inline,
// version-tagged, hidden-visibility) hipcub namespace, so aliasing cub ->
// hipcub lets every existing cub::DeviceScan/DeviceReduce/... call resolve to
// hipcub unchanged. (hipcub ignores CUB_WRAPPED_NAMESPACE; the inline namespace
// plus hidden visibility already prevent a clash with torch's bundled hipcub.)
// ---------------------------------------------------------------------------
#define cub hipcub

// ---------------------------------------------------------------------------
// cuRAND device API (rand.cu): 1:1 with the hipRAND device API.
// ---------------------------------------------------------------------------
#define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#define curand_init hiprand_init
#define curand_uniform4 hiprand_uniform4
#define curand_uniform2_double hiprand_uniform2_double
#define curand4 hiprand4

#endif  // defined(K2_WITH_HIP)

#endif  // K2_CSRC_CUDA_TO_HIP_H_
