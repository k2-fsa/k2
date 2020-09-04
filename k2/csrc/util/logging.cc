// k2/csrc/util/logging.h

// Copyright (c) 2020 Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/util/logging.h"

#ifdef K2_USE_GLOG
// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too.
// (e.g. Torch).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
}  // namespace glog_internal_namespace_
}  // namespace google

namespace k2 {
bool InitK2Logging(int* argc, char** argv) {
  if (*argc == 0)
    return true;
#if !defined(_MSC_VER)
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
#endif
  {
    ::google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    // This is never defined on Windows
    ::google::InstallFailureSignalHandler();
#endif
  }
  UpdateLoggingLevelsFromFlags(); // read globle google gflags if exists.
  return true;
}

void UpdateLoggingLevelsFromFlags() {
  // set some default FLAGS/gflags for K2 here,
  // if that differ with glog default.
}
}  // namespace k2
#else  // !K2_USE_GLOG
namespace k2 {

bool InitK2Logging(int *argc, char **argv) {
  if (*argc == 0)
    return true;

  loguru::init(*argc, argv);
  return true;
}

void UpdateLoggingLevelsFromFlags() {}

}  // namespace k2
#endif  // !K2_USE_GLOG
