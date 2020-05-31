// k2/csrc/util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/util.h"

#include <stdlib.h>

#include "glog/logging.h"

namespace k2 {

void *MemAlign(size_t nbytes, size_t alignment) {
  void *p = nullptr;
#if defined(_MSC_VER)
  // windows
  p = _aligned_malloc(nbytes, alignment);
#else
  int ret = posix_memalign(&p, alignment, nbytes);
  CHECK_EQ(ret, 0);
#endif

  CHECK_NOTNULL(p);
  return p;
}

void MemFree(void *ptr) { free(ptr); }

}  // namespace k2
