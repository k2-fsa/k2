/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (author: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"

namespace k2 {

RegionPtr NewRegion(ContextPtr &context, std::size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  auto ans = std::make_shared<Region>();
  ans->context = context;
  // TODO(haowen): deleter_context is always null with above constructor,
  // we need add another constructor of Region to allow the caller
  // to provide deleter_context.
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

}  // namespace k2
