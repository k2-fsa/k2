/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_TIMER_H_
#define K2_CSRC_TIMER_H_

#include <memory>

#include "k2/csrc/context.h"

namespace k2 {

class TimerImpl;

class Timer {
 public:
  explicit Timer(ContextPtr context);
  ~Timer();

  void Reset() const;

  // Return time in seconds
  double Elapsed() const;

 private:
  std::unique_ptr<TimerImpl> timer_impl_;
};

}  // namespace k2

#endif  // K2_CSRC_TIMER_H_
