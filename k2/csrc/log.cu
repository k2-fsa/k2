/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Stack trace related stuff is from kaldi.
 * Refer to
 * https://github.com/kaldi-asr/kaldi/blob/master/src/base/kaldi-error.cc
 */

#include "k2/csrc/log.h"

#include <time.h>

#ifdef K2_HAVE_EXECINFO_H
#include <execinfo.h>  // To get stack trace in error messages.
#ifdef K2_HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // K2_HAVE_CXXABI_H
#endif  // K2_HAVE_EXECINFO_H

#include <stdlib.h>

#include <chrono>  // NOLINT
#include <ctime>
#include <string>

namespace k2 {

namespace internal {

std::string GetTimeStamp() {
  using namespace std::chrono;  // NOLINT
  auto now = system_clock::now();
  std::time_t time = system_clock::to_time_t(now);
  std::tm tm;
#ifndef _MSC_VER
  localtime_r(&time, &tm);
#else
  localtime_s(&tm, &time);
#endif
  char s[128];
  std::strftime(s, sizeof(s), "%F %T", &tm);
  int32_t ms =
      duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
  std::ostringstream os;
  os << s << "." << ms;
  return os.str();
}

static bool LocateSymbolRange(const std::string &trace_name, std::size_t *begin,
                              std::size_t *end) {
  // Find the first '_' with leading ' ' or '('.
  *begin = std::string::npos;
  for (std::size_t i = 1; i < trace_name.size(); ++i) {
    if (trace_name[i] != '_') {
      continue;
    }
    if (trace_name[i - 1] == ' ' || trace_name[i - 1] == '(') {
      *begin = i;
      break;
    }
  }
  if (*begin == std::string::npos) {
    return false;
  }
  *end = trace_name.find_first_of(" +", *begin);
  return *end != std::string::npos;
}

#ifdef K2_HAVE_EXECINFO_H
static std::string Demangle(const std::string &trace_name) {
#ifndef K2_HAVE_CXXABI_H
  return trace_name;
#else   // K2_HAVE_CXXABI_H
  // Try demangle the symbol. We are trying to support the following formats
  // produced by different platforms:
  //
  // Linux:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  //
  // Mac:
  //   0 server 0x000000010f67614d _ZNK5kaldi13MessageLogger10LogMessageEv + 813
  //
  // We want to extract the name e.g., '_ZN5kaldi13UnitTestErrorEv' and
  // demangle it info a readable name like kaldi::UnitTextError.
  std::size_t begin, end;
  if (!LocateSymbolRange(trace_name, &begin, &end)) {
    return trace_name;
  }
  std::string symbol = trace_name.substr(begin, end - begin);
  int status;
  char *demangled_name = abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
  if (status == 0 && demangled_name != nullptr) {
    symbol = demangled_name;
    free(demangled_name);
  }
  return trace_name.substr(0, begin) + symbol +
         trace_name.substr(end, std::string::npos);
#endif  // K2_HAVE_CXXABI_H
}
#endif  // K2_HAVE_EXECINFO_H

std::string GetStackTrace() {
  std::string ans;
#ifdef K2_HAVE_EXECINFO_H
  constexpr const std::size_t kMaxTraceSize = 50;
  constexpr const std::size_t kMaxTracePrint = 50;  // Must be even.
                                                    // Buffer for the trace.
  void *trace[kMaxTraceSize];
  // Get the trace.
  std::size_t size = backtrace(trace, kMaxTraceSize);
  // Get the trace symbols.
  char **trace_symbol = backtrace_symbols(trace, size);
  if (trace_symbol == nullptr) return ans;

  // Compose a human-readable backtrace string.
  ans += "[ Stack-Trace: ]\n";
  if (size <= kMaxTracePrint) {
    for (std::size_t i = 0; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
  } else {  // Print out first+last (e.g.) 5.
    for (std::size_t i = 0; i < kMaxTracePrint / 2; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    ans += ".\n.\n.\n";
    for (std::size_t i = size - kMaxTracePrint / 2; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    if (size == kMaxTraceSize)
      ans += ".\n.\n.\n";  // Stack was too long, probably a bug.
  }

  // We must free the array of pointers allocated by backtrace_symbols(),
  // but not the strings themselves.
  free(trace_symbol);
#endif  // K2_HAVE_EXECINFO_H
  return ans;
}

}  // namespace internal

}  // namespace k2
