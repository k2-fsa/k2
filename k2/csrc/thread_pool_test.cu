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

#include <algorithm>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <utility>

#include "gtest/gtest.h"
#include "k2/csrc/math.h"
#include "k2/csrc/thread_pool.h"

namespace k2 {

// return a string representation of the current thread ID.
static std::string GetThreadId() {
  std::ostringstream os;
  os << std::this_thread::get_id();
  return os.str();
}

TEST(ThreadPool, TestGetThreadPool) {
  ThreadPool *p = GetThreadPool();
  std::mutex mutex;
  std::unordered_set<std::string> ids;
  int32_t num_tasks = RandInt(10, 10000);
  std::vector<int32_t> data;

  for (int32_t i = 0; i != num_tasks; ++i) {
    p->SubmitTask([&mutex, &ids, &data, i]() {
      std::string id = GetThreadId();
      {
        std::lock_guard<std::mutex> lock(mutex);
        ids.insert(std::move(id));
        data.push_back(i);
      }
    });
  }

  p->WaitAllTasksFinished();
  EXPECT_LE(ids.size(), p->GetNumThreads());

  EXPECT_EQ(static_cast<int32_t>(data.size()), num_tasks);

  std::sort(data.begin(), data.end());
  for (int32_t i = 0; i != num_tasks; ++i) EXPECT_EQ(i, data[i]);
}

TEST(ThreadPool, TestThreadPool) {
  std::mutex mutex;
  std::unordered_set<std::string> ids;
  std::vector<int32_t> data;

  int32_t num_tasks = RandInt(20, 10000);
  int32_t num_threads = 10;
  {
    ThreadPool pool(num_threads);
    for (int32_t i = 0; i != num_tasks; ++i) {
      pool.SubmitTask([&mutex, &ids, &data, i]() {
        std::string id = GetThreadId();
        {
          std::lock_guard<std::mutex> lock(mutex);
          ids.insert(std::move(id));
          data.push_back(i);
        }
      });
    }
    // the destructor of pool will ensure
    // that all submitted tasks are finished
    // before exiting
  }

  EXPECT_LE(ids.size(), num_threads);

  EXPECT_EQ(static_cast<int32_t>(data.size()), num_tasks);

  std::sort(data.begin(), data.end());
  for (int32_t i = 0; i != num_tasks; ++i) EXPECT_EQ(i, data[i]);
}

}  // namespace k2
