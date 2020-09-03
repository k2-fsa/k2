// k2/csrc/cuda/array_test.cu

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include <algorithm>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/tensor.h"

namespace k2 {
template <typename T, DeviceType d>
void TestArray1() {
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }
  // TODO
}

TEST(ArrayTest, Array1Test) {
  TestArray1<int32_t, kCpu>();
  TestArray1<int32_t, kCuda>();
  TestArray1<double, kCpu>();
  TestArray1<double, kCuda>();
}

}  // namespace k2
