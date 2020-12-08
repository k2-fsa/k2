/**
 * @brief Unittest for pinned context.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/test_utils.h"
#include "k2/csrc/timer.h"

namespace k2 {

TEST(PinnedContext, Cache) {
  if (GetCudaContext()->GetDeviceType() == kCpu) {
    // No CUDA capable devices are found, skip the test.
    return;
  }
  // NOTE: this test has to be run before all other
  // tests in this file since it assumes that the
  // memory pool of PinnedContext is empty at the
  // beginning.
  const int8_t *p = nullptr;
  const int8_t *q = nullptr;
  ContextPtr pinned = GetPinnedContext();

  {
    Array1<int8_t> a1(pinned, 100);
    p = a1.Data();
  }

  // at this point, a1 is freed but its
  // memory is not returned to the system
  // and can be reused in later allocations

  {
    Array1<int8_t> a2(pinned, 100);
    // a2 should reuse the memory occupied by a1
    EXPECT_EQ(p, a2.Data());
  }

  {
    // a3 cannot reuse the memory of a1 since it
    // requires 101 bytes, which is larger than 100
    Array1<int8_t> a3(pinned, 101);
    EXPECT_NE(p, a3.Data());
    q = a3.Data();
  }
  // at this point, the pool contains two blocks
  // of memory: 100 bytes and 101 bytes. The blocks
  // are sorted by size in increasing order.

  {
    Array1<int8_t> a4(pinned, 99);  // reuse the block with 100 bytes
    Array1<int8_t> a5(pinned, 1);   // reuse the block with 101 bytes
    EXPECT_EQ(p, a4.Data());
    EXPECT_EQ(q, a5.Data());
  }
}

static void PinnedContextSpeedTest() {
  ContextPtr cpu = GetCpuContext();
  ContextPtr cuda = GetCudaContext();
  ContextPtr pinned = GetPinnedContext();

  double elapsed_cpu = 0.;
  double elapsed_pinned = 0.;
  double total_bytes = 0.;

  Timer timer(cuda);

  int32_t bytes = (1 << 20) * 30;  // 30MB
  for (int32_t i = 0; i != 20; ++i) {
    int32_t num_bytes = bytes + i % 5;
    total_bytes += num_bytes;

    Array1<int8_t> cpu_array(cpu, num_bytes);
    Array1<int8_t> cuda_array1(cuda, num_bytes);
    Array1<int8_t> cuda_array2(cuda, num_bytes);
    Array1<int8_t> pinned_array(pinned, num_bytes);

    int8_t *cpu_array_data = cpu_array.Data();
    int8_t *pinned_array_data = pinned_array.Data();

    for (int32_t k = 0; k != num_bytes; ++k) {
      cpu_array_data[k] = static_cast<int8_t>(RandInt(-128, 127));
      pinned_array_data[k] = static_cast<int8_t>(RandInt(-128, 127));
    }

    if (i & 1) {
      timer.Reset();
      // CAUTION: it includes the time for allocating extra pinned memory
      // and extra copying from non-pinned to pinned memory.
      cpu->CopyDataTo(num_bytes, cpu_array.Data(), cuda, cuda_array1.Data());
      elapsed_cpu += timer.Elapsed();

      timer.Reset();
      pinned->CopyDataTo(num_bytes, pinned_array.Data(), cuda,
                         cuda_array2.Data());
      elapsed_pinned += timer.Elapsed();
    } else {
      timer.Reset();
      pinned->CopyDataTo(num_bytes, pinned_array.Data(), cuda,
                         cuda_array2.Data());
      elapsed_pinned += timer.Elapsed();

      timer.Reset();
      cpu->CopyDataTo(num_bytes, cpu_array.Data(), cuda, cuda_array1.Data());
      elapsed_cpu += timer.Elapsed();
    }

    CheckArrayData(cpu_array, cuda_array1);
    CheckArrayData(pinned_array, cuda_array2);
  }

  printf("non-pinned memory copy (host->device): : %.2f GB/s\n",
         (total_bytes / (1 << 30)) / elapsed_cpu);

  printf("pinned memory copy (host->device): : %.2f GB/s\n",
         (total_bytes / (1 << 30)) / elapsed_pinned);
}

TEST(PinnedContext, SpeedTest) { PinnedContextSpeedTest(); }

}  // namespace k2
