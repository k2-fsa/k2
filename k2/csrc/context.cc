/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"

namespace k2 {

void BackgroundRunner::Background(std::function<void()> &f) {}

void BackgroundRunner::Wait() {}

}  // namespace k2
