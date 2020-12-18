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

#ifndef K2_CSRC_SEMAPHORE_H_
#define K2_CSRC_SEMAPHORE_H_

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
  explicit counting_semaphore(int count = 0): count_(count) { }

  void release() {  // could also be 'signal'
    std::lock_guard<std::mutex> lock(mutex_);
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


}  // namespace k2


#endif  // K2_CSRC_SEMAPHORE_H_
