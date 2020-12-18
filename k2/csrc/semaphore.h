/**
 * @brief
 * semaphore
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <mutex>
#include <condition_variable>
#include "k2/csrc/log.h"

// caution: this contains class k2std::counting_semaphore, but not class k2::Semaphore
// which is in context.h.

// `k2std` is our replacement for `std` until we require C++20 that has
// `counting_semaphore` as part of the standard library.
namespace k2std {


// This is intended to implement a subset of the functionality of C++20's
// counting_semaphore (at the time of writing, we compile with C++14.)
class counting_semaphore {
public:
  counting_semaphore(int count = 0): count_(count) { }

  void release() {  // could also be 'signal'
    std::unique_lock<std::mutex> lock(mutex_);
    ++count_;
    cv_.notify_one();
  }
  void acquire() {  // could also be 'wait'
    std::unique_lock<std::mutex> lock(mutex_);
    while(count_ == 0)
      cv_.wait(lock);
    --count_;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
};


}
