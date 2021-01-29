// This file is modified from cudpp/src/cudpp/cudpp_util.h
// and cudpp/src/cudpp/sharedmem.h
//
// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

#ifndef K2_CSRC_CUDPP_CUDPP_UTIL_H_
#define K2_CSRC_CUDPP_CUDPP_UTIL_H_

#include <cfloat>

namespace k2 {

constexpr int32_t SEGSCAN_ELTS_PER_THREAD = 8;
constexpr int32_t SCAN_CTA_SIZE = 128;
constexpr int32_t LOG_SCAN_CTA_SIZE = 7;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t LOG_WARP_SIZE = 5;

template <typename T, int N>
struct typeToVector {
  typedef T Result;
};

template <>
struct typeToVector<int, 4> {
  typedef int4 Result;
};

template <>
struct typeToVector<float, 4> {
  typedef float4 Result;
};

template <>
struct typeToVector<double, 4> {
  typedef double4 Result;
};

template <>
struct typeToVector<int, 3> {
  typedef int3 Result;
};

template <>
struct typeToVector<float, 3> {
  typedef float3 Result;
};

template <>
struct typeToVector<int, 2> {
  typedef int2 Result;
};

template <>
struct typeToVector<float, 2> {
  typedef float2 Result;
};

template <typename T>
class OperatorAdd {
 public:
  __device__ T operator()(const T a, const T b) { return a + b; }
  __device__ T identity() { return (T)0; }
};

template <typename T>
class OperatorMin {
 public:
  __device__ T operator()(const T a, const T b) const { return min(a, b); }

  // no implementation - only specializations allowed
  __device__ T identity() const;
};

template <>
__device__ inline int OperatorMin<int>::identity() const {
  return INT_MAX;
}

template <>
__device__ inline float OperatorMin<float>::identity() const {
  return FLT_MAX;
}

template <>
__device__ inline double OperatorMin<double>::identity() const {
  return DBL_MAX;
}

template <typename T>
class OperatorMax {
 public:
  __device__ T operator()(const T a, const T b) const { return max(a, b); }

  // no implementation - only specializations allowed
  __device__ T identity() const;
};

template <>
__device__ inline int OperatorMax<int>::identity() const {
  return INT_MIN;
}

template <>
__device__ inline float OperatorMax<float>::identity() const {
  return -FLT_MAX;
}

template <>
__device__ inline double OperatorMax<double>::identity() const {
  return -DBL_MAX;
}

template <typename T>
struct SharedMemory {
  __device__ T *getPointer() {
    // Ensure that we won't compile any un-specialized types
    extern __device__ void Error_UnsupportedType();
    Error_UnsupportedType();
    return (T *)0;
  }
};

template <>
struct SharedMemory<int> {
  __device__ int *getPointer() {
    extern __shared__ int s_int[];
    return s_int;
  }
};

template <>
struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

}  // namespace k2

#endif  // K2_CSRC_CUDPP_CUDPP_UTIL_H_
