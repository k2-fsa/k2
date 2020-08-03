//
// Created by songmeixu (songmeixu@outlook.com) on 2020/8/3.
// Copyright (c) 2020 Xiaomi Inc. All rights reserved.
//

#ifndef K2_UTIL_LOGGING_H_
#define K2_UTIL_LOGGING_H_

// Choose one logging implementation.
// All have the same common API of google/glog
#ifdef K2_USE_GLOG
#include "logging_is_google_glog.h"
#else
#include "logging_is_not_google_glog.h"
#endif

namespace k2 {

// Functions that we use for initialization.
bool InitK2Logging(int* argc, char** argv);
void UpdateLoggingLevelsFromFlags();

constexpr bool IsUsingGoogleLogging() {
#ifdef K2_USE_GLOG
  return true;
#else
  return false;
#endif
}

}  // namespace k2
#endif  // K2_UTIL_LOGGING_H_
