// k2/csrc/cuda/debug.h

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_DEBUG_H_
#define K2_CSRC_CUDA_DEBUG_H_

/**
 * Include multiple cuda headers to make host compiler preprocessor happy.
 *
 * @todo
 *  Find a way to avoid this and make .h/.cc with cuda code
 *  could be parsed by host compiler (specially, GNU-gcc).
 *  (May assgin to nvcc to take control through change cmake
 *  compiler and options. Then, it need cmake-3.18 `FindCUDAToolkit`
 *  or other config helpers to make things easy.)
 */
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <stdio.h>
#include <assert.h>

namespace k2 {

/**
 * @brief A static assertion
 *
 * @param[in] exp the compile-time boolean expression that must be true
 * @param[in] msg an error message if exp is false
 *
 * @note `static_assert` is supported by both of host and device.
 *
 * @code{.cpp}
 * K2_STATIC_ASSERT(DEFINED_SHAPE % DEFINED_X == 0);
 * @endcode
 */
#define K2_STATIC_ASSERT(exp, msg) static_assert(exp, msg)

/**
 * @brief Check if the expression is true.
 *
 * @details Implemented by `assert`, which is supported by host and device.
 *
 * @param[in] exp the boolean expression that should be true
 *
 * @code{.cpp}
 * K2_ASSERT(1 == 1);
 * @endcode
 */
#define K2_ASSERT(exp) assert(exp)

/**
 * @brief Check if two arguments are equal.
 *
 * @details Implemented by `assert`, which is supported by host and device.
 *
 * @param[in] a left argument to compare
 * @param[in] b right argument to compare
 *
 * @code{.cpp}
 * K2_CHECK_EQ(1, 1);
 * @endcode
 */
#define K2_CHECK_EQ(a, b) assert( a == b )

/**
 * @fn
 *  __host__ __device__ __forceinline__ cudaError_t
 *  _K2CudaDebug(cudaError_t error,
 *               const char *filename,
 *               int line)
 *
 * @brief This is a error checking function, with context information.
 *        It's not designed to called by users, but inner macros.
 *        It's made used by other macros.
 *
 * @param[in] error         an enum type indicating CUDA errors.
 * @param[in] filename      the source filename that the error comes from.
 * @param[in] line          the code line that the error happened.
 * @return                  Pass the input CUDA error.
 *
 * @code{.cpp}
 * _K2CudaDebug(cudaGetLastError(), __FILE__, __LINE__);
 * @endcode
 */
__host__ __device__ __forceinline__ cudaError_t _K2CudaDebug(
    cudaError_t error,
    const char *filename,
    int line) {
  (void)filename;
  (void)line;
  if (cudaSuccess != error) {
  #ifndef __CUDA_ARCH__
    fprintf(stderr, "CUDA error ID=%d, NAME=%s, [%s, %d]: %s\n",
            error, cudaGetErrorName(error),
            filename, line,
            cudaGetErrorString(error));
    fflush(stderr);
  #elif __CUDA_ARCH__ >= 200
    printf("CUDA error ID=%d, NAME=%s, "
           "[block (%d,%d,%d) thread (%d,%d,%d), %s, %d]: %s\n",
           error, cudaGetErrorName(error),
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           filename, line,
           cudaGetErrorString(error));
  #endif
  }
  return error;
}

/**
 * @def K2_CUDA_CHECK_ERROR(cudaError)
 *
 * @brief Macro for checking cuda error.
 *
 * @details
 *  If error is not cudaSuccess, print the error message and exit with the
 *  error enum value. Otherwise, it does nothing.
 *
 * @param[in] e an enum type indicating CUDA errors.
 *
 * @code{.cpp}
 * K2_CUDA_CHECK_ERROR(cudaGetLastError());
 * @endcode
 */
#define K2_CUDA_CHECK_ERROR(e)                                    \
  if (::k2::_K2CudaDebug((cudaError_t)(e), __FILE__, __LINE__)) { \
    exit(e);                                                      \
  }

/**
 * @def K2_CUDA_SAFE_CALL([cuda_runtime_api|kernel])
 *
 * @brief Macro for checking "cuda standard runtime api"
 *        or "kernels" return status.
 *
 * @details
 *  - If api return status is not cudaSuccess, print and exit.
 *  - The `cudaDeviceSynchronize` only happens when `NDEBUG` is not defined.
 *
 * @note
 *  Kernel launches do not return any error code, thus checking should after it.
 *  To wait kernel to finish, cudaDeviceSynchronize is called between.
 *
 * @code{.cpp}
 * K2_CUDA_SAFE_CALL(cudaRuntimeApi());
 * K2_CUDA_SAFE_CALL(kernel_func<<<...>>>());
 * @endcode
 */
#ifndef NDEBUG
  #define K2_CUDA_SAFE_CALL(...)               \
    do {                                       \
      (__VA_ARGS__);                           \
      cudaDeviceSynchronize();                 \
      K2_CUDA_CHECK_ERROR(cudaGetLastError()); \
    } while (0)
#else
  #define K2_CUDA_SAFE_CALL(...)               \
    do {                                       \
      (__VA_ARGS__);                           \
      K2_CUDA_CHECK_ERROR(cudaGetLastError()); \
    } while (0)
#endif

/**
 * @def K2_DLOG
 *
 * @brief Log macro for printf statements.
 *
 * @details
 *  `printf` is supported by both host and device. This Log is for debugging,
 *  the error msg is printed to the stderr. The log msg always get printed,
 *  regardless of macro `NDEBUG`. Thus it should only be used for debugging.
 *
 * @code{.cpp}
 * K2_DLOG("Value is %d, string is %s ..", i, str);
 * @endcode
 */
#ifndef __CUDA_ARCH__
  #define K2_DLOG(format, ...) printf(format,__VA_ARGS__)
#elif __CUDA_ARCH__ >= 200
  #define K2_DLOG(format, ...)                                            \
    printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.x,  \
           blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, \
           __VA_ARGS__)
#endif

/**
 * @brief An more expensive asserts only checked if `K2_PARANOID` defined.
 *
 * @details
 * If triggered, these info get shown: "file, line, [blockIdx, threadIdx],
 * formated message, and the standard asserts info".
 *
 * @param[in]           exp     the expression expected to be true
 * @param[in]           format  an error message if exp is false
 * @param[in] \optional ...     the optional arguments for printf format.
 *
 * @code{.cpp}
 * K2_PARANOID_ASSERT(a >= b, "a must be greater than b, "
 *     "but now a = %d, b = %d", a, b);
 *
 * K2_PARANOID_ASSERT(a >= b, "a must be greater than b");
 * @endcode
 */
#ifdef K2_PARANOID
  #define K2_PARANOID_ASSERT(exp, format, ...)                           \
    do {                                                                 \
      if (exp)                                                           \
        (void)0;                                                         \
      else {                                                             \
        K2_DLOG(" [%s, %d] " format, __FILE__, __LINE__, ##__VA_ARGS__); \
        assert(exp);                                                     \
      }                                                                  \
    } while (0)
#else
  #define K2_PARANOID_ASSERT(exp, format, ...) ((void)0)
#endif

}  // namespace k2

#endif  // K2_CSRC_CUDA_DEBUG_H_
