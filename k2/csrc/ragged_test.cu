/**
 * @brief
 * ragged_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/array.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/ragged.h"

namespace k2 {
template <typename T, DeviceType d>
void TestTranspose() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // created with Array1(ContextPtr ctx, int32_t size), test Array1.Data()
    Array1<T> array(context, 5);
    ASSERT_EQ(array.Dim(), 5);
    std::vector<T> data(array.Dim());
    std::iota(data.begin(), data.end(), 0);
    T *array_data = array.Data();
    // copy data from CPU to CPU/GPU
    auto kind = GetMemoryCopyKind(*cpu, *array.Context());
    MemoryCopy(static_cast<void *>(array_data),
               static_cast<void *>(data.data()),
               array.Dim() * array.ElementSize(), kind);
    // copy data from CPU/GPU to CPU
    kind = GetMemoryCopyKind(*array.Context(), *cpu);
    std::vector<T> cpu_data(array.Dim());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<void *>(array_data),
               array.Dim() * array.ElementSize(), kind);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(cpu_data[i], i);
    }
  }

  {
    // test operator=(T t)
    Array1<T> array(context, 5);
    ASSERT_EQ(array.Dim(), 5);
    // operator=(T t)
    array = 2;
    // copy data from CPU/GPU to CPU
    T *array_data = array.Data();
    auto kind = GetMemoryCopyKind(*array.Context(), *cpu);
    std::vector<T> cpu_data(array.Dim());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<void *>(array_data),
               array.Dim() * array.ElementSize(), kind);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(cpu_data[i], 2);
    }
  }

  {
    // created with Array1(ContextPtr, int32_t size, T elem)
    Array1<T> array(context, 5, 2);
    ASSERT_EQ(array.Dim(), 5);
    // copy data from CPU/GPU to CPU
    T *array_data = array.Data();
    auto kind = GetMemoryCopyKind(*array.Context(), *cpu);
    std::vector<T> cpu_data(array.Dim());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<void *>(array_data),
               array.Dim() * array.ElementSize(), kind);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(cpu_data[i], 2);
    }
  }

  {
    // created with Array(ContextPtr, const std:vector<T>&)
    std::vector<T> data(5);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    ASSERT_EQ(array.Dim(), 5);
    // copy data from CPU/GPU to CPU
    T *array_data = array.Data();
    auto kind = GetMemoryCopyKind(*array.Context(), *cpu);
    std::vector<T> cpu_data(array.Dim());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<void *>(array_data),
               array.Dim() * array.ElementSize(), kind);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(cpu_data[i], data[i]);
    }
  }

  // TODO(haowen): add more tests
}

TEST(ArrayTest, Array1Test) {
  TestArray1<int32_t, kCpu>();
  TestArray1<int32_t, kCuda>();
  TestArray1<double, kCpu>();
  TestArray1<double, kCuda>();
}

}  // namespace k2
