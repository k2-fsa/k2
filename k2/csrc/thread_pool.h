/**
 * @brief Thread pool.
 *
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#ifndef K2_CSRC_THREAD_POOL_H_
#define K2_CSRC_THREAD_POOL_H_

#include <condition_variable>  // NOLINT
#include <functional>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

#include "k2/csrc/log.h"

namespace k2 {

class ThreadPool {
 public:
  using TaskFunc = std::function<void()>;

  /* Create a pool with given number of threads.
   *
   * @param [in] num_threads  The number of threads in the pool. If it is <= 0,
   *                          the number of threads in the pool is set to
   *                          `std::thread::hardware_concurrency()`.
   */
  explicit ThreadPool(int32_t num_threads);

  // It has un-copyable members of type `std::mutex` and
  // `std::condition_variable` so this class is not copyable.

  // The destructor waits for all threads in the pool to exit.
  ~ThreadPool();

  // return the number of threads in the pool
  int32_t GetNumThreads() const {
    return static_cast<int32_t>(threads_.size());
  }

  /* Insert a task into the queue. It is executed immediately
   * if there are currently idle threads. Otherwise, it is saved
   * in the queue and waits for an idle thread to be free to pop it from
   * the queue and to execute it.
   *
   * @param [in] task   The task to be run. It should be convertible to
   *                    `std::function<void()>`.
   */
  template <typename Lambda>
  void SubmitTask(Lambda task) {
    K2_CHECK_GT(threads_.size(), 0u);
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.emplace(static_cast<TaskFunc>(task));
    not_empty_cond_.notify_one();
    finished_ = false;
  }

  /* The caller is blocked if `tasks_` is not empty.
   * If `tasks_` is empty, it returns immediately.
   */
  void WaitAllTasksFinished();

 private:
  /* Remove tasks from `task_` and process them.
   *
   * If `keep_running_` is false, the function will return
   * after processing the current task.
   *
   * If the task queue is empty, the caller is blocked.
   */
  void ProcessTasks();

 private:
  // new tasks are added to this queue
  std::queue<TaskFunc> tasks_;

  std::vector<std::thread> threads_;
  std::mutex mutex_;

  // When the last task is removed and processed from `tasks_`,
  // `empty_cond_` is signaled.
  std::condition_variable empty_cond_;

  // When a new task is added to `tasks_`,
  // `not_empty_cond_` is signaled.
  std::condition_variable not_empty_cond_;

  // Set it to false in the destructor to
  // ask threads to not process tasks in the
  // waiting queue
  bool keep_running_ = true;

  // Set it to true when the task queue is empty
  bool finished_ = true;
};

/* Get a pointer to global thread pool.
 *
 * The returned pointer is NOT owned by the caller and
 * it must not be freed by the caller.
 */
ThreadPool *GetThreadPool();

}  // namespace k2

#endif  // K2_CSRC_THREAD_POOL_H_
