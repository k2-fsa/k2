
#ifndef K2_CSRC_CUDPP_CUDPP_UTIL_H_
#define K2_CSRC_CUDPP_CUDPP_UTIL_H_
#include <cfloat>

namespace k2 {

/** @brief Utility template struct for generating small vector types from scalar
 * types
 *
 * Given a base scalar type (\c int, \c float, etc.) and a vector length (1
 * through 4) as template parameters, this struct defines a vector type (\c
 * float3, \c int4, etc.) of the specified length and base type.  For example:
 * \code
 * template <class T>
 * __device__ void myKernel(T *data)
 * {
 *     typeToVector<T,4>::Result myVec4;             // create a vec4 of type T
 *     myVec4 = (typeToVector<T,4>::Result*)data[0]; // load first element of
 * data as a vec4
 * }
 * \endcode
 *
 * This functionality is implemented using template specialization.  Currently
 * specializations for int, float, and unsigned int vectors of lengths 2-4 are
 * defined.  Note that this results in types being generated at compile time --
 * there is no runtime cost.  typeToVector is used by the optimized scan \c
 * __device__ functions in scan_cta.cu.
 */
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
