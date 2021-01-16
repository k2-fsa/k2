/**
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
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
