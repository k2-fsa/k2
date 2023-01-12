/**
 * Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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
 */

#include <utility>

#include "k2/csrc/thread_pool.h"

namespace k2 {

static int32_t GetDefaultNumThreads() {
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 1;
  return num_threads;
}

ThreadPool::ThreadPool(int32_t num_threads)
    : threads_(num_threads > 0 ? num_threads : GetDefaultNumThreads()) {
  for (auto &thread : threads_)
    thread = std::thread([this]() { this->ProcessTasks(); });
}

ThreadPool::~ThreadPool() {
  WaitAllTasksFinished();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    keep_running_ = false;
    not_empty_cond_.notify_all();  // wake up all threads in the pool
  }
  for (auto &thread : threads_) thread.join();
}

void ThreadPool::WaitAllTasksFinished() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!finished_) {
    // wait for the `empty_cond_` condition.
    empty_cond_.wait(lock);
  }
}

void ThreadPool::ProcessTasks() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (keep_running_) {
    while (tasks_.empty() && keep_running_) {
      // Wait for the `not_empty_` condition.
      // SubmitTask() or the destructor will signal it.
      not_empty_cond_.wait(lock);
    }

    if (!keep_running_) break;

    {
      auto task = std::move(tasks_.front());
      tasks_.pop();
      if (!tasks_.empty()) not_empty_cond_.notify_one();
      ++running_counter_;
      lock.unlock();  // let other threads proceed
      task();
      // any resource associated with `task` is freed here
    }

    lock.lock();
    --running_counter_;
    if (tasks_.empty() && running_counter_ == 0) {
      finished_ = true;
      // if WaitAllTasksFinished() is waiting, wake it up
      empty_cond_.notify_one();
    }

    // the lock is still held here as we assume
    // that on entering the while() loop, the lock is locked by us
  }
}

ThreadPool *GetThreadPool() {
  static ThreadPool *pool = nullptr;
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    // TODO(fangjun): how to determine the number
    // of threads in the pool?
    //
    // It is never freed
    pool = new ThreadPool(2);
  });
  K2_CHECK_NE(pool, nullptr);
  return pool;
}

}  // namespace k2
