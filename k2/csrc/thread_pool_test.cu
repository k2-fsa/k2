/**
 * @brief Unin test for thread pool.
 *
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <string>

#include "gtest/gtest.h"
#include "k2/csrc/thread_pool.h"

namespace k2 {

// return a string representation of the current thread ID.
static std::string GetThreadId() {
  std::ostringstream os;
  os << std::this_thread::get_id();
  return os.str();
}

TEST(ThreadPool, Test) {
  ThreadPool *p = GetThreadPool();

  for (int32_t i = 0; i != 16; ++i) {
    p->RunTask([i]() {
      printf("task %2d: %s\n", i, GetThreadId().c_str());
      usleep(100);
    });
  }

  p->WaitAllTasksFinished();
}

}  // namespace k2
