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

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

void BackgroundRunner::Background(std::function<void()> &f) {}

void BackgroundRunner::Wait() {}

RegionPtr NewRegion(ContextPtr &context, std::size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  std::shared_ptr<Region> ans = std::make_shared<Region>();
  ans->context = context->shared_from_this();
  // TODO(haowen): deleter_context is always null with above constructor,
  // we need add another constructor of Region to allow the caller
  // to provide deleter_context.
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

}  // namespace k2
