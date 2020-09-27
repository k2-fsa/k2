/**
 * @brief
 * util
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/util.h"

#include <stdlib.h>

#include "k2/csrc/log.h"

namespace k2host {

void *MemAlignedMalloc(std::size_t nbytes, std::size_t alignment) {
  void *p = nullptr;
#if defined(_MSC_VER)
  // windows
  p = _aligned_malloc(nbytes, alignment);
#else
  int32_t ret = posix_memalign(&p, alignment, nbytes);
  K2_CHECK_EQ(ret, 0);
#endif

  K2_CHECK_NE(p, nullptr);
  return p;
}

void MemFree(void *ptr) {
#if defined(_MSC_VER)
  // windows
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace k2host
