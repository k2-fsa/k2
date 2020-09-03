// k2/csrc/cuda/log.h

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

// Glog-like logging functions for k2.

#ifndef K2_CSRC_CUDA_LOG_H_
#define K2_CSRC_CUDA_LOG_H_

#include <stdio.h>

#include <cstdint>

namespace k2 {

namespace internal {

enum class LogLevel {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kFatal = 4,
};

// They are used in LOG(xxx), so their names
// do not follow the google c++ code style
constexpr LogLevel DEBUG = LogLevel::kDebug;
constexpr LogLevel INFO = LogLevel::kInfo;
constexpr LogLevel WARNING = LogLevel::kWarning;
constexpr LogLevel ERROR = LogLevel::kError;
constexpr LogLevel FATAL = LogLevel::kFatal;

class Logger {
 public:
  explicit __host__ __device__ Logger(LogLevel level) : level_(level) {}

  __host__ __device__ ~Logger() {
    printf("\n");
    if (level_ == FATAL) {
#if defined(__CUDA_ARCH__)
      // abort() is not available for device code.
      __threadfence();
      asm("trap;");
#else
      abort();
#endif
    }
  }

  __host__ __device__ const Logger &operator<<(const char *s) const {
    printf("%s", s);
    return *this;
  }

  __host__ __device__ const Logger &operator<<(int32_t i) const {
    printf("%d", i);
    return *this;
  }

  __host__ __device__ const Logger &operator<<(uint32_t i) const {
    printf("%u", i);
    return *this;
  }

  __host__ __device__ const Logger &operator<<(double d) const {
    printf("%f", d);
    return *this;
  }

 private:
  LogLevel level_;
};

class Voidifier {
 public:
  __host__ __device__ void operator&(const Logger &) const {}
};

}  // namespace internal

}  // namespace k2

#define K2_KERNEL_DEBUG_STR                                                \
  "block: [" << blockIdx.x << "," << blockIdx.y << "," << blockIdx.z       \
             << "], thread: [" << threadIdx.x << "," << threadIdx.y << "," \
             << threadIdx.z << "]"

#define K2_FILE_DEBUG_STR __FILE__ << ":" << __func__ << ":" << __LINE__ << " "

#define K2_CHECK(x)                                                           \
  (x) ? (void)0                                                               \
      : k2::internal::Voidifier() & k2::internal::Logger(k2::internal::FATAL) \
                                        << K2_FILE_DEBUG_STR                  \
                                        << "Check failed: " << #x << " "

// WARNING: x and y are may be evaluated multiple times, but this happens only
// when the check fails. Since the program aborts if it fails, I don't think
// the extra evaluation of x and y matters.
#define _K2_CHECK_OP(x, y, op)                                               \
  ((x)op(y)) ? (void)0                                                       \
             : k2::internal::Voidifier() &                                   \
                   k2::internal::Logger(k2::internal::FATAL)                 \
                       << K2_FILE_DEBUG_STR << "Check failed: " << #x << " " \
                       << #op << " " << #y << " (" << (x) << " vs. " << (y)  \
                       << ") "

#define K2_CHECK_EQ(x, y) _K2_CHECK_OP(x, y, ==)
#define K2_CHECK_NE(x, y) _K2_CHECK_OP(x, y, !=)
#define K2_CHECK_LT(x, y) _K2_CHECK_OP(x, y, <)
#define K2_CHECK_LE(x, y) _K2_CHECK_OP(x, y, <=)
#define K2_CHECK_GT(x, y) _K2_CHECK_OP(x, y, >)
#define K2_CHECK_GE(x, y) _K2_CHECK_OP(x, y, >=)

#define K2_LOG(x)                       \
  k2::internal::Logger(k2::internal::x) \
      << "[" << #x << "] " << K2_FILE_DEBUG_STR

// TODO(fangjun): add K2_DCHECK, K2_DCHECK_EQ, ...

#endif  // K2_CSRC_CUDA_LOG_H_
