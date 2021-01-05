// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * cudpp_util.h
 *
 * @brief C++ utility functions and classes used internally to cuDPP
 */

#ifndef __CUDPP_UTIL_H__
#define __CUDPP_UTIL_H__

#ifdef WIN32
#include <windows.h>
#endif

#include <cuda.h>
#include <cudpp.h>
#include <limits.h>
#include <float.h>

#if (CUDA_VERSION >= 3000)
#define LAUNCH_BOUNDS(x) __launch_bounds__((x))
#define LAUNCH_BOUNDS_MINBLOCKs(x, y) __launch_bounds__((x),(y))
#else
#define LAUNCH_BOUNDS(x)
#define LAUNCH_BOUNDS_MINBLOCKS(x, y)
#endif

#ifndef _SafeDeleteArray
#define _SafeDeleteArray(x) { if(x) { delete [](x); (x)=0; } }
#endif

/** @brief Determine if \a n is a power of two.
  * @param n Value to be checked to see if it is a power of two
  * @returns True if \a n is a power of two, false otherwise
  */
inline bool
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

/** @brief Determine if an integer \a n is a multiple of an integer \a f.
  * @param n Multiple
  * @param f Factor
  * @returns True if \a n is a multiple of \a f, false otherwise
  */
inline bool
isMultiple(int n, int f)
{
    if (isPowerOfTwo(f))
        return ((n&(f-1))==0);
    else
        return (n%f==0);
}

/** @brief Compute the smallest power of two larger than \a x.
  * @param x Input value
  * @returns The smallest power f two larger than \a x
  */
inline unsigned int
ceilPow2( unsigned int x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/** @brief Compute the largest power of two smaller than or equal to \a x.
  * @param x Input value
  * @returns The largest power of two smaller than or equal to \a x.
  */
inline unsigned int
floorPow2(unsigned int x)
{
    return ceilPow2(x) >> 1;
}


/** @brief Compute the base 2 logarithm of a power-of-2 integer\a x.
  * Taken from: http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
  * @param x Input value
  * @returns The log base 2 of \a x.
  */
inline unsigned int
logBase2Pow2(unsigned int x)
{
    static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0,
                                     0xFF00FF00, 0xFFFF0000};
    unsigned int r = (x & b[0]) != 0;
    for (unsigned int i = 4; i > 0; i--) { r |= ((x & b[i]) != 0) << i; }
    return r;
}


/** @brief Returns the maximum value for type \a T.
 * @returns Maximum value for type \a T.
 *
 * Implemented using template specialization on \a T.
 */
template <class T>
__host__ __device__ inline T getMax() { return 0; }
/** @brief Returns the minimum value for type \a T.
 * @returns Minimum value for type \a T.
 *
 * Implemented using template specialization on \a T.
 */
template <class T>
__host__ __device__ inline T getMin() { return 0; }
// type specializations for the above
// getMax
template <> __host__ __device__ inline int getMax() { return INT_MAX; }
template <> __host__ __device__ inline unsigned int getMax() { return INT_MAX; }
template <> __host__ __device__ inline float getMax() { return FLT_MAX; }
template <> __host__ __device__ inline double getMax() { return DBL_MAX; }
template <> __host__ __device__ inline char getMax() { return (char)INT_MAX; }
template <> __host__ __device__ inline unsigned char getMax() { return (unsigned char)INT_MAX; }
template <> __host__ __device__ inline long long getMax() { return LLONG_MAX; }
template <> __host__ __device__ inline unsigned long long getMax() { return ULLONG_MAX; }
// getMin
template <> __host__ __device__ inline int getMin() { return INT_MIN; }
template <> __host__ __device__ inline unsigned int getMin() { return 0; }
template <> __host__ __device__ inline float getMin() { return -FLT_MAX; }
template <> __host__ __device__ inline double getMin() { return -DBL_MAX; }
template <> __host__ __device__ inline char getMin() { return (char)INT_MIN; }
template <> __host__ __device__ inline unsigned char getMin() { return (unsigned char)0; }
template <> __host__ __device__ inline long long getMin() { return LLONG_MIN; }
template <> __host__ __device__ inline unsigned long long getMin() { return 0; }

/** @brief Returns the maximum of three values.
  * @param a First value.
  * @param b Second value.
  * @param c Third value.
  * @returns The maximum of \a a, \a b and \a c.
  */
template<class T>
inline int max3(T a, T b, T c)
{
    return (a > b) ? ((a > c)? a : c) : ((b > c) ? b : c);
}

/** @brief Utility template struct for generating small vector types from scalar types
  *
  * Given a base scalar type (\c int, \c float, etc.) and a vector length (1 through 4) as
  * template parameters, this struct defines a vector type (\c float3, \c int4, etc.) of the
  * specified length and base type.  For example:
  * \code
  * template <class T>
  * __device__ void myKernel(T *data)
  * {
  *     typeToVector<T,4>::Result myVec4;             // create a vec4 of type T
  *     myVec4 = (typeToVector<T,4>::Result*)data[0]; // load first element of data as a vec4
  * }
  * \endcode
  *
  * This functionality is implemented using template specialization.  Currently specializations
  * for int, float, and unsigned int vectors of lengths 2-4 are defined.  Note that this results
  * in types being generated at compile time -- there is no runtime cost.  typeToVector is used by
  * the optimized scan \c __device__ functions in scan_cta.cu.
  */
template <typename T, int N>
struct typeToVector
{
    typedef T Result;
};

template<>
struct typeToVector<char, 4>
{
    typedef char4 Result;
};
template<>
struct typeToVector<unsigned char, 4>
{
    typedef uchar4 Result;
};
template<>
struct typeToVector<short, 4>
{
    typedef short4 Result;
};
template<>
struct typeToVector<unsigned short, 4>
{
    typedef ushort4 Result;
};
template<>
struct typeToVector<int, 4>
{
    typedef int4 Result;
};
template<>
struct typeToVector<unsigned int, 4>
{
    typedef uint4 Result;
};
template<>
struct typeToVector<float, 4>
{
    typedef float4 Result;
};
template<>
struct typeToVector<double, 4>
{
    typedef double4 Result;
};
template<>
struct typeToVector<long long, 4>
{
    typedef longlong4 Result;
};
template<>
struct typeToVector<unsigned long long, 4>
{
    typedef ulonglong4 Result;
};
template<>
struct typeToVector<char, 3>
{
    typedef char3 Result;
};
template<>
struct typeToVector<unsigned char, 3>
{
    typedef uchar3 Result;
};
template<>
struct typeToVector<short, 3>
{
    typedef short3 Result;
};
template<>
struct typeToVector<unsigned short, 3>
{
    typedef ushort3 Result;
};
template<>
struct typeToVector<int, 3>
{
    typedef int3 Result;
};
template<>
struct typeToVector<unsigned int, 3>
{
    typedef uint3 Result;
};
template<>
struct typeToVector<float, 3>
{
    typedef float3 Result;
};
template<>
struct typeToVector<long long, 3>
{
    typedef longlong3 Result;
};
template<>
struct typeToVector<unsigned long long, 3>
{
    typedef ulonglong3 Result;
};
template<>
struct typeToVector<char, 2>
{
    typedef char2 Result;
};
template<>
struct typeToVector<unsigned char, 2>
{
    typedef uchar2 Result;
};
template<>
struct typeToVector<short, 2>
{
    typedef short2 Result;
};
template<>
struct typeToVector<unsigned short, 2>
{
    typedef ushort2 Result;
};
template<>
struct typeToVector<int, 2>
{
    typedef int2 Result;
};
template<>
struct typeToVector<unsigned int, 2>
{
    typedef uint2 Result;
};
template<>
struct typeToVector<float, 2>
{
    typedef float2 Result;
};
template<>
struct typeToVector<long long, 2>
{
    typedef longlong2 Result;
};
template<>
struct typeToVector<unsigned long long, 2>
{
    typedef ulonglong2 Result;
};
template <typename T>
class OperatorAdd
{
public:
    __device__ T operator()(const T a, const T b) { return a + b; }
    __device__ T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    __device__ T operator()(const T a, const T b) { return a * b; }
    __device__ T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    __device__ T operator() (const T a, const T b) const { return max(a, b); }
    __device__ T identity() const; // no implementation - only specializations allowed
};

template <>
__device__ inline char OperatorMax<char>::identity() const { return CHAR_MIN; }
template <>
__device__ inline unsigned char OperatorMax<unsigned char>::identity() const { return 0; }
template <>
__device__ inline short OperatorMax<short>::identity() const { return SHRT_MIN; }
template <>
__device__ inline unsigned short OperatorMax<unsigned short>::identity() const { return 0; }
template <>
__device__ inline int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
__device__ inline unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
__device__ inline float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <>
__device__ inline double OperatorMax<double>::identity() const { return -DBL_MAX; }
template <>
__device__ inline long long OperatorMax<long long>::identity() const { return LLONG_MIN; }
template <>
__device__ inline unsigned long long OperatorMax<unsigned long long>::identity() const { return 0; }

template <typename T>
class OperatorMin
{
public:
    __device__ T operator() (const T a, const T b) const { return min(a, b); }
    __device__ T identity() const; // no implementation - only specializations allowed
};

template <>
__device__ inline char OperatorMin<char>::identity() const { return CHAR_MAX; }
template <>
__device__ inline unsigned char OperatorMin<unsigned char>::identity() const { return UCHAR_MAX; }
template <>
__device__ inline short OperatorMin<short>::identity() const { return SHRT_MAX; }
template <>
__device__ inline unsigned short OperatorMin<unsigned short>::identity() const { return USHRT_MAX; }
template <>
__device__ inline int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
__device__ inline unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
__device__ inline float OperatorMin<float>::identity() const { return FLT_MAX; }
template <>
__device__ inline double OperatorMin<double>::identity() const { return DBL_MAX; }
template <>
__device__ inline long long OperatorMin<long long>::identity() const { return LLONG_MAX; }
template <>
__device__ inline unsigned long long OperatorMin<unsigned long long>::identity() const { return ULLONG_MAX; }


class LSBBucketMapper {
public:
  LSBBucketMapper(unsigned int numBuckets) {
    lsbBitMask = 0xFFFFFFFF >>
        (32 - (unsigned int) ceil(log2((float)numBuckets)));
    this->numBuckets = numBuckets;
  }

  __device__ __inline__ unsigned int operator()(unsigned int element) {
    return (element & lsbBitMask) % numBuckets;
  }

private:
  unsigned int numBuckets;
  unsigned int lsbBitMask;
};

class MSBBucketMapper {
public:
  MSBBucketMapper(unsigned int numBuckets) {
    msbShift = 32 - ceil(log2((float)numBuckets));
    this->numBuckets = numBuckets;
  }

  __device__ __inline__ unsigned int operator()(unsigned int element) {
    return (element >> msbShift) % numBuckets;
  }

private:
  unsigned int numBuckets;
  unsigned int msbShift;
};

class OrderedCyclicBucketMapper {
public:
  OrderedCyclicBucketMapper(unsigned int elements, unsigned int buckets) {
    numElements = elements;
    numBuckets = buckets;
    elementsPerBucket = (elements + buckets - 1) / buckets;
  }

  __device__ __inline__ unsigned int operator()(unsigned int element) {
    return (element % numElements) / elementsPerBucket;
  }

private:
  unsigned int numBuckets;
  unsigned int numElements;
  unsigned int elementsPerBucket;
};

class CustomBucketMapper {
public:
  CustomBucketMapper(BucketMappingFunc bucketMappingFunc) {
    bucketMapper = bucketMappingFunc;
  }

  __device__ unsigned int operator()(unsigned int element) {
    return (*bucketMapper)(element);
  }

private:
  BucketMappingFunc bucketMapper;
};

#endif // __CUDPP_UTIL_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
