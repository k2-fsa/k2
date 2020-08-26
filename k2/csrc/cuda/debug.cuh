// k2/csrc/cuda/errors.h

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ERRORS_H_
#define K2_CSRC_CUDA_ERRORS_H_

#include <cstdio>

#include "k2/csrc/cuda/arch.h"

namespace k2 {

// In debug mode and include this header,
// define K2_MAKE_ERROR_CHECK to turn on error checking
#if (!defined(NDEBUG) && !defined(K2_MAKE_ERROR_CHECK))
  #define K2_MAKE_ERROR_CHECK
#endif

/**
 * \brief A static assertion
 * \param exp the compile-time boolean expression that must be true
 * \param msg an error message if exp is false
 *
 * `static_assert` is supported by both of host and device.
 */
#ifndef K2_STATIC_ASSERT
  #define K2_STATIC_ASSERT(exp, msg) static_assert(exp, msg)
#endif

/**
 * \brief This is a error checking function, with context information.
 * If K2_MAKE_ERROR_CHECK is defined and error is not cudaSuccess,
 * the corresponding error message is printed to:
 *   - host: stderr
 *   - device: stdout in device
 * along with the supplied source context.
 * It's used to make other macros convenient.
 *
 * \return The CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t K2_CUDA_DEBUG(
    cudaError_t error,
    const char *filename,
    int line,
    bool abort = true) {
#ifdef K2_MAKE_ERROR_CHECK
  (void)filename;
  (void)line;
  if (cudaSuccess != error) {
  #if (K2_PTX_ARCH == 0)
    fprintf(stderr, "CUDA error ID=%d, NAME=%s, [%s, %d]: %s\n",
            error, cudaGetErrorName(error),
            filename, line,
            cudaGetErrorString(error));
    fflush(stderr);
  #elif (K2_PTX_ARCH >= 200)
    printf("CUDA error ID=%d, NAME=%s, [block (%d,%d,%d) thread (%d,%d,%d), %s, %d]: %s\n",
           error, cudaGetErrorName(error),
           blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
           filename, line,
           cudaGetErrorString(error));
  #endif
    if (abort) {
      exit(EXIT_FAILURE);
    }
  }
#endif
  return error;
}

/**
 * \brief Macro for checking cuda error.
 * If not cudaSuccess and K2_MAKE_ERROR_CHECK, print and exit.
 */
#ifndef K2_CUDA_CHECK_ERROR
  #define K2_CUDA_CHECK_ERROR(e) \
    k2::K2_CUDA_DEBUG((cudaError_t) (e), __FILE__, __LINE__, true)
#endif

/**
 * \brief Macro for checking cuda standard runtime api return status.
 * If api return status is not cudaSuccess, print and exit.
 * Thus it's actually could be an alias of K2_CUDA_CHECK_ERROR.
 *
 * \note:
 * All runtime api return an error code, but for an asynchronous api,
 * the error code only reports errors that occur on the host
 * prior to executing the task, typically related to parameter validation.
 * The device kernel error would unfortunately be left for next runtime api call.
 * To check cuda runtime async api, as cuda kernel launches are asynchronous also,
 * the same macro for kernel could be used.
 * See `K2_CUDA_KERNEL_SAFE_CALL`.
 *
 * \usage:
 * K2_CUDA_API_SAFE_CALL(cuda_runtime_api())
 */
#ifndef K2_CUDA_API_SAFE_CALL
  #define K2_CUDA_API_SAFE_CALL(c) K2_CUDA_CHECK_ERROR(c)
#endif

/**
 * \brief Macro for checking cuda standard runtime api return status.
 * If api return status is not cudaSuccess, print and exit.
 * Thus it's actually an alias of K2_CUDA_CHECK_ERROR.
 *
 * \note:
 * Kernel launches do not return any error code, thus checking should after it.
 * To wait kernel to finish, cudaDeviceSynchronize is called between.
 * And to avoid pre-launch error disturbing, check error before kernel launch is
 * also needed (but not necessary if one promise each error is take cared properly)
 *
 * \usage:
 * K2_CUDA_KERNEL_SAFE_CALL(kernel_foo<<<>>>())
 */
#ifndef K2_CUDA_KERNEL_SAFE_CALL
  #define K2_CUDA_KERNEL_SAFE_CALL(c)                 \
    do {                                              \
      K2_CUDA_CHECK_ERROR(cudaGetLastError());        \
      K2_CUDA_API_SAFE_CALL(c);                       \
      K2_CUDA_API_SAFE_CALL(cudaDeviceSynchronize()); \
      K2_CUDA_CHECK_ERROR(cudaGetLastError());        \
    } while (0)
#endif

/**
 * \brief Log macro for printf statements.
 * `printf` is supported by both host and device.
 * This Log is for debugging, the error msg is printed to the stderr.
 *
 * Refer to:
 * https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/util_debug.cuh#L109
 */
#if !defined(K2_DLOG)
  #if !(defined(__clang__) && defined(__CUDA__))
    #if (K2_PTX_ARCH == 0)
      #define K2_DLOG(format, ...) printf(format,__VA_ARGS__)
    #elif (K2_PTX_ARCH >= 200)
      #define K2_DLOG(format, ...)                                            \
        printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.z,  \
               blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, \
               __VA_ARGS__)
#endif
  #else
    // A hack to implement the variadic printf as clang for other c-compilers,
    // and sielence the warning
    #pragma clang diagnostic ignored "-Wc++11-extensions"
    #pragma clang diagnostic ignored "-Wunnamed-type-template-args"
      template <class... Args>
      inline __host__ __device__ void va_printf(char const* format, Args const&... args) {
    #ifdef __CUDA_ARCH__
        printf(format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, args...);
    #else
        printf(format, args...);
    #endif
      }
    #ifndef __CUDA_ARCH__
      #define K2_DLOG(format, ...) va_printf(format,__VA_ARGS__)
    #else
      #define K2_DLOG(format, ...) va_printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, __VA_ARGS__)
    #endif
  #endif
#endif

}  // end namespace k2

#endif  // K2_CSRC_CUDA_ERRORS_H_
