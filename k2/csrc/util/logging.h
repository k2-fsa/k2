// k2/csrc/util/logging_is_google_glog.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Meixu Song)

// See ../../LICENSE for clarification regarding multiple authors

/**
 * This is the logging header that need to be included for:
 *  - cpp: logging
 *  - gpu: debugging
 *    - Note: `#ifdef NDEBUG`, only the error msg get print into stdout.
 *            As other msg is for debugging, which would be silence under NDEBUG.
 *
 * Usage:
 * - gpu:
 *  As the "debug.h" has doxygen docs. Please check there for each macro.
 * - cpu:
 *  Check bellow @code...@endcode.
 *  Or http://rpg.ifi.uzh.ch/docs/glog.html
 *
 * @code{.cpp}
 * Severity Level:
 * enum class LogLevel { kDebug = 0, kInfo, kWarning, kError, kFatal };
 *
 * constexpr LogLevel DEBUG = LogLevel::kDebug;
 * constexpr LogLevel INFO = LogLevel::kInfo;
 * constexpr LogLevel WARNING = LogLevel::kWarning;
 * constexpr LogLevel ERROR = LogLevel::kError;
 * constexpr LogLevel FATAL = LogLevel::kFatal;
 *
 * // config glog
 * //  The glog should be config by main() caller, which is not K2's job.
 * //  Thus user of K2 don't need care this init part.
 * //  BTW, glog has its default setting.
 * //  Here, a init helper class demo is given.
 * class GLogHelper {
 *  public:
 *   GLogHelper(char* settingStr) {
 *     google::InitGoogleLogging(settingStr);
 *     FLAGS_stderrthreshold=google::INFO; // or FLAGS_logtostderr = 0;
 *     FLAGS_colorlogtostderr=true;
 *     FLAGS_v = 3;
 *   }
 *
 *   ~GLogHelper() {
 *     google::ShutdownGoogleLogging();
 *   }
 *
 *   // helper funcs
 *   void set_log_dir_1(const char *dst_folder) {
 *     google::SetLogDestination(google::INFO,
 *                               (std::string(dst_folder) + "INFO").c_str());
 *     google::SetLogDestination(google::WARNING,
 *                               (std::string(dst_folder) + "WARNING").c_str());
 *     google::SetLogDestination(google::ERROR,
 *                               (std::string(dst_folder) + "ERROR").c_str());
 *   }
 *
 *   void set_log_dir_2(const char *dst_folder) {
 *     FLAGS_log_dir = dst_folder; // set google gflags if exists
 *   }
 * };
 * ...
 * // accept the cmd line setting as init args
 * GLogHelper gh(argv[0]);
 *
 *
 * // LOG()
 * LOG(DEBUG) << "This get print only config glog at DEBUG level."
 * LOG(INFO) << "...";
 * LOG(WARNING) << "...";
 * LOG(ERROR) << "...";
 * LOG(FATAL) << "When this get print, program exit(1) with stacktrace";
 * LOG_IF(INFO, month > 12) << "month shoud not > 12, month = ";
 *
 * // CHECK()
 * //  Just checking and logging, do not terminate programs if fails.
 * CHECK(fp->Write(x) == 0) << "Write failed!";
 * CHECK_NOTNULL(some_ptr);
 * CHECK_EQ, CHECK_NE, CHECK_LE, CHECK_LT, CHECK_GE, CHECK_GT
 * /// string checkings
 * CHECK_STREQ(s1, s2) << "This get print if s1 != s2";
 * CHECK_STRNE, CHECK_STRCASEEQ, CHECK_STRCASENE;
 * /// numerical precision checkings
 * CHECK_DOUBLE_EQ, CHECK_NEAR;
 *
 * // DLOG()
 * //  Only valid in debug mode (`NDEBUG` is not defined). This
 * //  avoids slowing down the program in the production environment due to
 * //  the large number of logs.
 * DLOG(INFO) << "...";
 * DLOG_IF(INFO, month > 12) << "month shoud not > 12, month = " << month;
 *
 * // PLOG()
 * //  As google perror log style, log with error discription
 * //  and `errno` (error code).
 * PLOG(LEVEL) << "";
 * PLOG_IF(LEVEL, expression) << "error msg";
 * PCHECK(write(1, NULL, 2) >= 0) << "Write NULL failed";
 * /// Result:
 * /// F0825 185142 test.cc:22] Check failed: write(1, NULL, 2) >= 0
 * /// Write NULL failed: Bad address [14]
 *
 *
 * // RAW_LOG
 * //  Thread-safe logs, require `#include <glog / raw_logging.h>`
 * RAW_LOG()
 * @endcode
 */

#ifndef K2_CSRC_UTIL_LOGGING_H_
#define K2_CSRC_UTIL_LOGGING_H_

#include "k2/csrc/util/glog_macros.h"

// Choose one logging implementation.
// All have the same common API of google/glog
#ifdef K2_USE_GLOG
#include "k2/csrc/util/logging_is_google_glog.h"
#else
#include "k2/csrc/util/logging_is_not_google_glog.h"
#endif

namespace k2 {

/**
 * @fn bool InitK2Logging(int *argc, char **argv)
 *
 * @brief Funtion that used for glog initialization.
 *
 * @details It could accept the cmd line or
 *
 * @note It's unexpected to call for user.
 */
bool InitK2Logging(int* argc, char** argv);

/**
 * @fn void UpdateLoggingLevelsFromFlags()
 *
 * @brief Funtion to update glog config from gflags.
 *
 * @note It's unexpected to call by k2 user.
 */
void UpdateLoggingLevelsFromFlags();

/**
 * @fn constexpr bool IsUsingGoogleLogging()
 *
 * @return bool that tells glog is presented or not
 */
constexpr bool IsUsingGoogleLogging() {
#ifdef K2_USE_GLOG
  return true;
#else
  return false;
#endif
}

}  // namespace k2
#endif  // K2_CSRC_UTIL_LOGGING_H_
