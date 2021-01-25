/**
 * k2/csrc/cudpp_test.cu
 *
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/cudpp/cudpp.h"

namespace k2 {

TEST(CUDPP, SegmentedScan) {
  ContextPtr c = GetCudaContext();
  if (c->GetDeviceType() != kCuda) return;

  std::vector<int32_t> v = {1, 2, 3, 4, 5, 6, 7};
  std::vector<uint32_t> flags = {0, 0, 0, 0, 1, 0, 0};
  Array1<int32_t> din(c, v);
  Array1<uint32_t> dflags(c, flags);
  Array1<int32_t> dout(c, v.size(), 0);

  cudppSegmentedScan(din.Data(), din.Data(), dflags.Data(), v.size());
  K2_LOG(INFO) << din;
}

}  // namespace k2
