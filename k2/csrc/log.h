/**
 * @brief
 * log
 * Glog-like logging functions for k2.
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_LOG_H_
#define K2_CSRC_LOG_H_

#include <cstdint>
#include <cstdio>

namespace k2 {

namespace internal {

#if defined(NDEBUG)
constexpr bool kDisableDebug = true;
#else
constexpr bool kDisableDebug = false;
#endif

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
  __host__ __device__ Logger(const char *filename, const char *func_name,
                             uint32_t line_num, LogLevel level)
      : filename_(filename),
        func_name_(func_name),
        line_num_(line_num),
        level_(level) {
    switch (level) {
      case DEBUG:
        printf("[D] ");
        break;
      case INFO:
        printf("[I] ");
        break;
      case WARNING:
        printf("[W] ");
        break;
      case ERROR:
        printf("[E] ");
        break;
      case FATAL:
        printf("[F] ");
        break;
    }
    printf("%s:%s:%u ", filename, func_name, line_num);
#if defined(__CUDA_ARCH__)
    printf("block:[%u,%u,%u], thread: [%u,%u,%u] ", blockIdx.x, blockIdx.y,
           blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif
  }

  __host__ __device__ ~Logger() {
    printf("\n");
    if (level_ == FATAL) {
#if defined(__CUDA_ARCH__)
      // this is usually caused by one of the CHECK macros and the detailed
      // error messages should have already been printed by the macro, so we
      // use an arbitrary string here.
      __assert_fail("Some bad things happened", filename_, line_num_,
                    func_name_);
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

  template <typename T>
  const Logger &operator<<(const T &) const {
    return *this;
  }

 private:
  const char *filename_;
  const char *func_name_;
  uint32_t line_num_;
  LogLevel level_;
};

class Voidifier {
 public:
  __host__ __device__ void operator&(const Logger &) const {}
};

}  // namespace internal

}  // namespace k2

#define K2_CHECK(x)                                              \
  (x) ? (void)0                                                  \
      : ::k2::internal::Voidifier() &                            \
            ::k2::internal::Logger(__FILE__, __func__, __LINE__, \
                                   ::k2::internal::FATAL)        \
                << "Check failed: " << #x << " "

// WARNING: x and y may be evaluated multiple times, but this happens only
// when the check fails. Since the program aborts if it fails, we don't think
// the extra evaluation of x and y matters.
//
// CAUTION: we recommend the following use case:
//
//      auto x = Foo();
//      auto y = Bar();
//      K2_CHECK_EQ(x, y) << "Some message";
//
//  And please avoid
//
//      K2_CHECK_EQ(Foo(), Bar());
//
//  if `Foo()` or `Bar()` causes some side effects, e.g., changing some
//  local static variables or global variables.
#define _K2_CHECK_OP(x, y, op)                                              \
  ((x)op(y)) ? (void)0                                                      \
             : ::k2::internal::Voidifier() &                                \
                   ::k2::internal::Logger(__FILE__, __func__, __LINE__,     \
                                          ::k2::internal::FATAL)            \
                       << "Check failed: " << #x << " " << #op << " " << #y \
                       << " (" << (x) << " vs. " << (y) << ") "

#define K2_CHECK_EQ(x, y) _K2_CHECK_OP(x, y, ==)
#define K2_CHECK_NE(x, y) _K2_CHECK_OP(x, y, !=)
#define K2_CHECK_LT(x, y) _K2_CHECK_OP(x, y, <)
#define K2_CHECK_LE(x, y) _K2_CHECK_OP(x, y, <=)
#define K2_CHECK_GT(x, y) _K2_CHECK_OP(x, y, >)
#define K2_CHECK_GE(x, y) _K2_CHECK_OP(x, y, >=)

#define K2_LOG(x) \
  ::k2::internal::Logger(__FILE__, __func__, __LINE__, ::k2::internal::x)

#define K2_CHECK_CUDA_ERROR(x) \
  K2_CHECK_EQ(x, cudaSuccess) << " Error: " << cudaGetErrorString(x) << ". "

// ============================================================
//       For debug check
// ------------------------------------------------------------

#define K2_DCHECK(x) ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK(x)

#define K2_DCHECK_EQ(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_EQ(x, y)

#define K2_DCHECK_NE(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_NE(x, y)

#define K2_DCHECK_LT(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_LT(x, y)

#define K2_DCHECK_LE(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_LE(x, y)

#define K2_DCHECK_GT(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_GT(x, y)

#define K2_DCHECK_GE(x, y) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_GE(x, y)

#define K2_DLOG(x)                        \
  ::k2::internal::kDisableDebug ? (void)0 \
                                : ::k2::internal::Voidifier() & K2_LOG(x)

#define K2_DCHECK_CUDA_ERROR(x) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_CUDA_ERROR(x)

#endif  // K2_CSRC_LOG_H_
