// k2/csrc/cuda/logging_test.cc

// Copyright (c) 2020 Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "k2/csrc/util/logging.h"

//@todo refine this gtest, add more examples, as you see, it's just the old way
//      you use glog, except the init, the init is the top caller job if he
//      doesn't like the default one. Actually, he may only be able do this
//      through glog, fortunately we support glog.

namespace k2 {

TEST(LoggingTest, Log) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  {
    LOG(INFO) << "INFO msg";
    ASSERT_DEATH(LOG(FATAL) << "FATAL makes program exit(1)", "");
  }

  // @todo the glog config works for loguru partly, to figure it out.
  // test "severity level",
  {
#ifdef K2_USE_GLOG
    LOG(INFO) << "file";
    // Most flags work immediately after updating values.
    FLAGS_logtostderr = 1;  // by default, glog output to stderr
    LOG(INFO) << "stderr";
    FLAGS_logtostderr = 0;
    // This won't change the log destination. If you want to set this
    // value, you should do this before google::InitGoogleLogging .
    FLAGS_log_dir = "/some/log/directory";
    LOG(INFO) << "the same file";
#else
    LOG(INFO) << "file";
    // Most flags work immediately after updating values.
    loguru::g_stderr_verbosity = 1;  // by default, glog output to stderr
    LOG(INFO) << "stderr";
    loguru::g_stderr_verbosity = 0;
    // you could do this before loguru::init().
    loguru::add_file("/tmp/loguru_test",
                     loguru::Truncate,
                     loguru::g_stderr_verbosity);
    LOG(INFO) << "the new file";
#endif
  }
}

}  // namespace k2
