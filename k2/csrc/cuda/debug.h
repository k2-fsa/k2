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
 * @note
 *  `assert(exp)` in device, if exp == 0, the kernel excution is halted.
 *  If the program is run within a debugger, this triggers a breakpoint and the
 *  debugger can be used to inspect the current state of the device.
 *  Otherwise, each thread for which expression is equal to zero prints a
 *  message to stderr after synchronization with the host via
 *  `cudaDeviceSynchronize(),cudaStreamSynchronize(),cudaEventSynchronize()`.
 *  The format of this message is as follows:
 *
 *  @code
 *  <filename>:<line number>:<function>:
 *  block: [blockId.x,blockId.x,blockIdx.z],
 *  thread: [threadIdx.x,threadIdx.y,threadIdx.z]
 *  Assertion `<expression>` failed.
 *  @endcode
 *
 *  @code{.cpp}
 *  K2_ASSERT(1 == 1);
 *  @endcode
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
 * @note
 *  Assertions are for debugging purposes. They can affect performance
 *  and it is therefore recommended to disable them in production code.
 *  `assert` can be disabled at compile time by defining the `NDEBUG`
 *  preprocessor macro before including assert.h.
 *
 *  @code{.cpp}
 *  K2_CHECK_EQ(1, 1);
 *  @endcode
 */
#define K2_CHECK_EQ(a, b) assert( a == b )

/**
 * @fn
 *  __host__ __device__ __forceinline__ cudaError_t
 *  _K2CudaDebug(cudaError_t error,
 *               const char *filename,
 *               int line)
 *
 *
 * @brief This is a error checking function, with context information.
 *        It's not designed to called by users, but inner macros.
 *
 * @details
 *  If error is not cudaSuccess,
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
 *  If error is not cudaSuccess,
 *  print the error message and exit with the error enum value.
 *  Otherwise, it does nothing.
 *
 * @param[in] e an enum type indicating CUDA errors.
 *
 * @remark
 *  Device cannot terminate/exit the host process. And calling a
 *  `__host__ function("exit")` from a `__host__ __device__ function`
 *  is not allowed. Thus, the exit(e) is put here, rather than inside of
 *  `_K2CudaDebug`.
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
 * K2_CUDA_API_SAFE_CALL(CUDA_RUNTIME_API());
 * @endcode
 */
#define K2_CUDA_API_SAFE_CALL(c) \
  do {                           \
    cudaError_t e = (c);         \
    K2_CUDA_CHECK_ERROR(e);      \
  } while (0)

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
 *  otherwise the compiler raises a error "passed 2 arguments, but takes just 1".
 *
 * @code{.cpp}
 * K2_CUDA_KERNEL_SAFE_CALL(kernel_func<<<...>>>());
 * @endcode
 */
#ifndef NDEBUG
  #define K2_CUDA_KERNEL_SAFE_CALL(...)               \
    do {                                              \
      (__VA_ARGS__);                                  \
      K2_CUDA_API_SAFE_CALL(cudaDeviceSynchronize()); \
      K2_CUDA_CHECK_ERROR(cudaGetLastError());        \
    } while (0)
#else
  #define K2_CUDA_KERNEL_SAFE_CALL(...) (__VA_ARGS__)
#endif

/**
 * @def K2_DLOG
 *
 * @brief Log macro for printf statements.
 *
 * @details
 *  `printf` is supported by both host and device. This Log is for debugging,
 *  the error msg is printed to the stderr. The log msg always get printed,
 *  regardless of macro `NDEBUG`. Thus it should only be
 *  make used of in debugging.
 *
 * @note
 *  This code is refered from:
 *  https://github.com/NVlabs/cub/blob/ \
 *  c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/util_debug.cuh#L109
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
 * If triggered, the info shown includes: file, line, [blockIdx, threadIdx],
 * formated message, and the standard asserts info.
 *
 * @param[in]           exp     the compile-time boolean expression that must be true
 * @param[in]           format  an error message if exp is false
 * @param[in] \optional ...     the optional arguments for printf format.
 *
 * @note
 * `assert` is supported by both of host and device.
 *  - host: assert(false), raises a "SIGABRT" exit
 *  - device: assert(false), device put msg into stderr and halt this one
 *            thread, but the msg won't get printed util synchronization.
 *
 * @remark
 *  Assertions are for debugging purposes. They can affect performance
 *  and it is therefore recommended to disable them in production code.
 *  `assert` can be disabled at compile time by defining the `NDEBUG`
 *  preprocessor macro before including assert.h.
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
