// k2/csrc/cuda/debug.h

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_DEBUG_H_
#define K2_CSRC_CUDA_DEBUG_H_

/**
 * To make host compiler preprocessor happy.
 *
 * @todo
 *  Find a way to avoid this and make .h/.cc with cuda code
 *  could be parsed by host compiler (specially, GNU-gcc).
 */
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <cstdio>

#include "k2/csrc/cuda/arch.h"

namespace k2 {

/**
 * @brief If in debug, define the `K2_MAKE_ERROR_CHECK`
 *        to turn on error checking.
 */
#if (!defined(NDEBUG) && !defined(K2_MAKE_ERROR_CHECK))
  #define K2_MAKE_ERROR_CHECK
#endif

/**
 * @brief A static assertion
 *
 * @param[in] exp the compile-time boolean expression that must be true
 * @param[in] msg an error message if exp is false
 *
 * @note `static_assert` is supported by both of host and device.
 */
#ifndef K2_STATIC_ASSERT
  #define K2_STATIC_ASSERT(exp, msg) static_assert(exp, msg)
#endif

/**
 * @fn
 *  __host__ __device__ __forceinline__ cudaError_t
 *  K2_CUDA_DEBUG(cudaError_t error,
 *                const char *filename,
 *                int line)
 *
 *
 * @brief This is a error checking function, with context information.
 *
 * @details
 *  If K2_MAKE_ERROR_CHECK is defined and error is not cudaSuccess,
 *  the corresponding error message is printed to:
 *    - host: stderr
 *    - device: stdout in device
 *  along with the supplied source context.
 *  It's used to make other macros convenient.
 *
 * @param[in] error         an enum type indicating CUDA errors.
 * @param[in] filename      the source filename that the error comes from.
 * @param[in] line          the code line that the error happened.
 * @return                  Pass the input CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t K2_CUDA_DEBUG(
    cudaError_t error,
    const char *filename,
    int line) {
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
    printf("CUDA error ID=%d, NAME=%s, "
           "[block (%d,%d,%d) thread (%d,%d,%d), %s, %d]: %s\n",
           error, cudaGetErrorName(error),
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           filename, line,
           cudaGetErrorString(error));
  #endif
  }
#endif
  return error;
}

/**
 * @def K2_CUDA_CHECK_ERROR(cudaError)
 *
 * @brief Macro for checking cuda error.
 *
 * @details
 *  If `K2_MAKE_ERROR_CHECK` is defined and e is not cudaSuccess,
 *  print the error message and exit with the error enum value.
 *  Otherwise, it does nothing.
 *
 * @param[in] e an enum type indicating CUDA errors.
 *
 * @remark
 *  Device cannot let host terminate/exit the process. And calling a
 *  `__host__ function("exit")` from a `__host__ __device__ function`
 *  is not allowed. Thus, the exit(e) is put here other than inside of
 *  `K2_CUDA_DEBUG`.
 *
 * @todo
 *  Should the error code return by this macro?
 */
#ifndef K2_CUDA_CHECK_ERROR
  #define K2_CUDA_CHECK_ERROR(e)                                   \
    if (k2::K2_CUDA_DEBUG((cudaError_t)(e), __FILE__, __LINE__)) { \
      exit(e);                                                     \
    }
#endif

/**
 * @def K2_CUDA_API_SAFE_CALL(CUDA_RUNTIME_API)
 *
 * @brief Macro for checking cuda standard runtime api return status.
 *
 * @details
 *  If api return status is not cudaSuccess, print and exit.
 *  Thus it's actually could be an alias of K2_CUDA_CHECK_ERROR.
 *
 * @note
 *  All runtime api return an error code. But for an asynchronous api,
 *  the error code only reports errors that occur on the host
 *  prior to executing the task, typically related to parameter validation.
 *  The device kernel error would unfortunately be left for next runtime api
 *  call. To check the whole process of cuda runtime async api,
 *  as cuda kernel launches are asynchronous also,
 *  the same macro for kernel could be used. See
 *  `K2_CUDA_KERNEL_SAFE_CALL`.
 *
 * @code{.cpp}
 * K2_CUDA_API_SAFE_CALL(CUDA_RUNTIME_API())
 * @endcode
 */
#ifndef K2_CUDA_API_SAFE_CALL
  #define K2_CUDA_API_SAFE_CALL(c) \
    do {                           \
      cudaError_t e = (c);         \
      K2_CUDA_CHECK_ERROR(e);      \
    } while (0)
#endif

/**
 * @def K2_CUDA_KERNEL_SAFE_CALL([kernel|cuda_runtime_api])
 *
 * @brief Macro for checking cuda standard runtime api return status.
 *
 * @details
 *  If api return status is not cudaSuccess, print and exit.
 *  Thus it's actually an alias of K2_CUDA_CHECK_ERROR.
 *
 * @note
 *  Kernel launches do not return any error code, thus checking should after it.
 *  To wait kernel to finish, cudaDeviceSynchronize is called between.
 *  And to avoid pre-launch error disturbing, check error before kernel launch
 *  is also needed.
 *  (But not necessary if programmer promise each error is take cared properly.)
 *
 * @remark
 *  macro `__VA_ARGS__` is used to pass the kernel<<<...>>> as one argument,
 *  otherwise the compiler rise a error "passed 2 arguments, but takes just 1".
 *
 * @code{.cpp}
 * K2_CUDA_KERNEL_SAFE_CALL(kernel_func<<<...>>>())
 * @endcode
 */
#ifndef K2_CUDA_KERNEL_SAFE_CALL
  #define K2_CUDA_KERNEL_SAFE_CALL(...)               \
    do {                                              \
      K2_CUDA_CHECK_ERROR(cudaGetLastError());        \
      (__VA_ARGS__);                                  \
      K2_CUDA_API_SAFE_CALL(cudaDeviceSynchronize()); \
      K2_CUDA_CHECK_ERROR(cudaGetLastError());        \
    } while (0)
#endif

/**
 * @def K2_DLOG
 *
 * @brief Log macro for printf statements.
 *
 * @details `printf` is supported by both host and device.
 *          This Log is for debugging, the error msg is printed to the stderr.
 *          The log msg always get printed, regardless of macro
 *          `K2_MAKE_ERROR_CHECK`. Thus it should only be
 *          make used of in debugging.
 *
 * @note
 *  This code is refered from:
 *  https://github.com/NVlabs/cub/blob/ \
 *  c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/util_debug.cuh#L109
 */
#if !defined(K2_DLOG)
  #if !(defined(__clang__) && defined(__CUDA__))
    #if (K2_PTX_ARCH == 0)
      #define K2_DLOG(format, ...) printf(format,__VA_ARGS__)
    #elif (K2_PTX_ARCH >= 200)
      #define K2_DLOG(format, ...)                                            \
        printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.x,  \
               blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, \
               __VA_ARGS__)
#endif
  #else
    /**
     * A hack to implement the variadic printf for clang,
     * and sielence the warning
     */
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wc++11-extensions"
    #pragma clang diagnostic ignored "-Wunnamed-type-template-args"
      template <class... Args>
      inline __host__ __device__
      void va_printf(char const* format, Args const&... args) {
    #ifdef __CUDA_ARCH__
        printf(format, blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z, args...);
    #else
        printf(format, args...);
    #endif
      }
    #ifndef __CUDA_ARCH__
      #define K2_DLOG(format, ...) \
        va_printf(format,__VA_ARGS__)
    #else
      #define K2_DLOG(format, ...) \
        va_printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, __VA_ARGS__)
    #endif
    #pragma clang diagnostic pop
  #endif
#endif

}  // namespace k2

#endif  // K2_CSRC_CUDA_DEBUG_H_
