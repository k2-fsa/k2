/**
 * @brief
 * log
 * Glog-like logging functions for k2.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Meixu Song)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 *
 * The following environment variables are related to logging:
 *
 *  K2_LOG_LEVEL
 *    - If it is not set, the default log level is INFO.
 *      That is, only messages logged with
 *          LOG(INFO), LOG(WARNING), LOG(ERROR) and LOG(FATAL)
 *      get printed.
 *    - Set it to "TRACE" to get all log message being printed
 *    - Set it to "FATAL" to print only FATAL messages
 */

#ifndef K2_CSRC_LOG_H_
#define K2_CSRC_LOG_H_

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>

#include "k2/csrc/macros.h"

#ifdef __CUDA_ARCH__
#define K2_CUDA_HOSTDEV __host__ __device__
#else
#define K2_CUDA_HOSTDEV
#endif

namespace k2 {

namespace internal {

#if defined(NDEBUG)
constexpr bool kDisableDebug = true;
#else
constexpr bool kDisableDebug = false;
#endif

enum class LogLevel {
  kTrace = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4,
  kFatal = 5,  // print message and abort the program
};

// They are used in K2_LOG(xxx), so their names
// do not follow the google c++ code style
//
// You can use them in the following way:
//
//  K2_LOG(TRACE) << "some message";
//  K2_LOG(DEBUG) << "some message";
//
constexpr LogLevel TRACE = LogLevel::kTrace;
constexpr LogLevel DEBUG = LogLevel::kDebug;
constexpr LogLevel INFO = LogLevel::kInfo;
constexpr LogLevel WARNING = LogLevel::kWarning;
constexpr LogLevel ERROR = LogLevel::kError;
constexpr LogLevel FATAL = LogLevel::kFatal;

std::string GetStackTrace();

/* Return the current log level.


   If the current log level is TRACE, then all logged messages are printed out.

   If the current log level is DEBUG, log messages with "TRACE" level are not
   shown and all other levels are printed out.

   Similarly, if the current log level is INFO, log message with "TRACE" and
   "DEBUG" are not shown and all other levels are printed out.

   If it is FATAL, then only FATAL messages are shown.
 */
K2_CUDA_HOSTDEV LogLevel GetCurrentLogLevel();

class Logger {
 public:
  K2_CUDA_HOSTDEV Logger(const char *filename, const char *func_name,
                         uint32_t line_num, LogLevel level)
      : filename_(filename),
        func_name_(func_name),
        line_num_(line_num),
        level_(level) {
    cur_level_ = GetCurrentLogLevel();
    switch (level) {
      case TRACE:
        if (cur_level_ <= TRACE) printf("[T] ");
        break;
      case DEBUG:
        if (cur_level_ <= DEBUG) printf("[D] ");
        break;
      case INFO:
        if (cur_level_ <= INFO) printf("[I] ");
        break;
      case WARNING:
        if (cur_level_ <= WARNING) printf("[W] ");
        break;
      case ERROR:
        if (cur_level_ <= ERROR) printf("[E] ");
        break;
      case FATAL:
        if (cur_level_ <= FATAL) printf("[F] ");
        break;
    }

    if (cur_level_ <= level_) {
      printf("%s:%s:%u ", filename, func_name, line_num);
#if defined(__CUDA_ARCH__)
      printf("block:[%u,%u,%u], thread: [%u,%u,%u] ", blockIdx.x, blockIdx.y,
             blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif
    }
  }

  K2_CUDA_HOSTDEV ~Logger() {
    printf("\n");
    if (level_ == FATAL) {
#if defined(__CUDA_ARCH__)
      // this is usually caused by one of the K2_CHECK macros and the detailed
      // error messages should have already been printed by the macro, so we
      // use an arbitrary string here.
      __assert_fail("Some bad things happened", filename_, line_num_,
                    func_name_);
#else
      std::string stack_trace = GetStackTrace();
      if (!stack_trace.empty()) {
        printf("\n\n%s\n", stack_trace.c_str());
      }
      fflush(nullptr);
      abort();
#endif
    }
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(int8_t i) const {
    if (cur_level_ <= level_) printf("%d", i);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(const char *s) const {
    if (cur_level_ <= level_) printf("%s", s);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(int32_t i) const {
    if (cur_level_ <= level_) printf("%d", i);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(uint32_t i) const {
    if (cur_level_ <= level_) printf("%u", i);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(uint64_t i) const {
    if (cur_level_ <= level_) printf("%llu", (long long unsigned int)i);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(int64_t i) const {
    if (cur_level_ <= level_) printf("%lli", (long long int)i);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(float f) const {
    if (cur_level_ <= level_) printf("%f", f);
    return *this;
  }

  K2_CUDA_HOSTDEV const Logger &operator<<(double d) const {
    if (cur_level_ <= level_) printf("%f", d);
    return *this;
  }

  template <typename T>
  const Logger &operator<<(const T &t) const {
    // require T overloads operator<<
    std::ostringstream os;
    os << t;
    return *this << os.str().c_str();
  }

  // specialization to fix compile error: `stringstream << nullptr` is ambiguous
  const Logger &operator<<(const std::nullptr_t &null) const {
    if (cur_level_ <= level_) *this << "(null)";
    return *this;
  }

 private:
  const char *filename_;
  const char *func_name_;
  uint32_t line_num_;
  LogLevel level_;
  LogLevel cur_level_;
};

class Voidifier {
 public:
  K2_CUDA_HOSTDEV void operator&(const Logger &)const {}
};

inline bool EnableCudaDeviceSync() {
  static std::once_flag init_flag;
  static bool enable_cuda_sync = false;
  std::call_once(init_flag, []() {
    enable_cuda_sync = (std::getenv("K2_SYNC_KERNELS") != nullptr);
  });
  return enable_cuda_sync;
}

inline bool DisableChecks() {
  // Currently this just disables the checks called in the constructor of
  // RaggedShape, which can otherwise dominate the time when in debug mode.
  static std::once_flag init_flag;
  static bool disable_checks = false;
  std::call_once(init_flag, []() {
    disable_checks = (std::getenv("K2_DISABLE_CHECKS") != nullptr);
  });
  return disable_checks;
}

inline K2_CUDA_HOSTDEV LogLevel GetCurrentLogLevel() {
#if defined(__CUDA_ARCH__)
  return DEBUG;
#else
  static LogLevel log_level = INFO;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    const char *env_log_level = std::getenv("K2_LOG_LEVEL");
    if (env_log_level == nullptr) return;
    std::string s = env_log_level;
    if (s == "TRACE")
      log_level = TRACE;
    else if (s == "DEBUG")
      log_level = DEBUG;
    else if (s == "INFO")
      log_level = INFO;
    else if (s == "WARNING")
      log_level = WARNING;
    else if (s == "ERROR")
      log_level = ERROR;
    else if (s == "FATAL")
      log_level = FATAL;
    else
      printf(
          "Unknown K2_LOG_LEVEL: %s"
          "\nSupported values are: "
          "TRACE, DEBUG, INFO, WARNING, ERROR, FATAL",
          s.c_str());
  });
  return log_level;
#endif
}

}  // namespace internal

}  // namespace k2

#define K2_STATIC_ASSERT(x) static_assert(x, "")

#define K2_CHECK(x)                                             \
  (x) ? (void)0                                                 \
      : ::k2::internal::Voidifier() &                           \
            ::k2::internal::Logger(__FILE__, K2_FUNC, __LINE__, \
                                   ::k2::internal::FATAL)       \
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
                   ::k2::internal::Logger(__FILE__, K2_FUNC, __LINE__,      \
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
  ::k2::internal::Logger(__FILE__, K2_FUNC, __LINE__, ::k2::internal::x)

// `x` would be error code returned from any cuda function call or kernel
// launch.
//
// Caution: don't do this:
//     K2_CHECK_CUDA_ERROR(cudaGetLastError())
// as it will call `cudaGetLastError` twice and reset the error status.
#define K2_CHECK_CUDA_ERROR(x) \
  K2_CHECK_EQ(x, cudaSuccess) << " Error: " << cudaGetErrorString(x) << ". "

// The parameter of `K2_CUDA_SAFE_CALL` should be cuda function call or kernel
// launch.
// Noted we would never call `cudaDeviceSynchronize` in release mode and
// user can even disable this call for debug mode by setting an environment
// variable `K2_SYNC_KERNELS` with any non-empty value, see
// function EnableCudaDeviceSync above.
#define K2_CUDA_SAFE_CALL(...)                \
  do {                                        \
    __VA_ARGS__;                              \
    if (!::k2::internal::kDisableDebug &&     \
        k2::internal::EnableCudaDeviceSync()) \
      cudaDeviceSynchronize();                \
    cudaError_t e = cudaGetLastError();       \
    K2_CHECK_CUDA_ERROR(e);                   \
  } while (0)

// ------------------------------------------------------------
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

// `x` would be error code returned from any cuda function call or kernel
// launch.
//
// CAUTION: don't do this:
//     auto error = cudaGetLastError();
//     K2_DCHECK_CUDA_ERROR(error);
// as you may reset the error status without checking it in release mode.
#define K2_DCHECK_CUDA_ERROR(x) \
  ::k2::internal::kDisableDebug ? (void)0 : K2_CHECK_CUDA_ERROR(x)

#endif  // K2_CSRC_LOG_H_
