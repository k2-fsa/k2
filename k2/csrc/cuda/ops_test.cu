// k2/csrc/cuda/ops_test.cu

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include <cstdio>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/cuda/array.h"
#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/ops.h"
#include "k2/csrc/cuda/timer.h"

namespace k2 {

template <typename T>
void MatrixTanspose(int32_t num_rows, int32_t num_cols, const T *src, T *dest) {
  for (int32_t i = 0; i < num_rows; ++i) {
    for (int32_t j = 0; j < num_cols; ++j) {
      dest[j * num_rows + i] = src[i * num_cols + j];
    }
  }
}

template <typename T>
void GpuTransposeTest(int32_t num_rows, int32_t num_cols, int32_t num_reps = 1,
                      bool print_bandwidth = false) {
  int32_t num_elements = num_rows * num_cols;
  std::vector<T> host_src(num_elements);
  std::iota(host_src.begin(), host_src.end(), 0);
  std::vector<T> gold(num_elements);
  MatrixTanspose<T>(num_rows, num_cols, host_src.data(), gold.data());

  int32_t num_bytes = num_elements * sizeof(T);
  ContextPtr context = GetCudaContext();
  auto src_region = NewRegion(context, num_bytes);
  Array2<T> src(num_rows, num_cols, num_cols, 0, src_region);
  cudaMemcpy(src.Data(), static_cast<void *>(host_src.data()), num_bytes,
             cudaMemcpyHostToDevice);

  auto dest_region = NewRegion(context, num_bytes);
  Array2<T> dest(num_cols, num_rows, num_rows, 0, dest_region);

  // warm up in case that the first kernel launch takes longer time.
  Transpose<T>(context, src, &dest);

  Timer t;
  for (int32_t i = 0; i < num_reps; ++i) {
    Transpose<T>(context, src, &dest);
  }
  double elapsed = t.Elapsed();

  std::vector<T> host_dest(num_elements);
  cudaMemcpy(static_cast<void *>(host_dest.data()), dest.Data(), num_bytes,
             cudaMemcpyDeviceToHost);

  ASSERT_EQ(host_dest, gold);

  if (print_bandwidth) {
    // effective_bandwidth (GB/s) = (read_bytes + write_bytes) / (time_seconds *
    // 10^9), for matrix transpose, read_bytes + write_bytes = 2 * num_bytes
    printf("Average time is: %.6f s, effective bandwidth is: %.2f GB/s\n",
           elapsed / num_reps, 2 * num_bytes * 1e-9 * num_reps / elapsed);
  }
}

TEST(OpsTest, TransposeGpuTest) {
  {
    // test with some corner cases
    std::vector<std::pair<int32_t, int32_t>> shapes = {
        {0, 0}, {1, 1}, {5, 4}, {100, 0}, {128, 64}, {15, 13}, {115, 180},
    };
    for (const auto &v : shapes) {
      GpuTransposeTest<int32_t>(v.first, v.second);
    }
  }

  {
    // test with random shapes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(0, 1000);
    for (int32_t i = 0; i != 20; ++i) {
      auto rows = dis(gen);
      auto cols = dis(gen);
      GpuTransposeTest<int32_t>(rows, cols);
    }
  }

  {
    // speed test for different data type
    // TODO(haowen): we may need to allocate different size of shared memory for
    // different data type to get the best performance
    GpuTransposeTest<char>(1000, 2000, 100, true);
    GpuTransposeTest<short>(1000, 2000, 100, true);
    GpuTransposeTest<int>(1000, 2000, 100, true);
    GpuTransposeTest<float>(1000, 2000, 100, true);
    GpuTransposeTest<double>(1000, 2000, 100, true);
  }
}

}  // namespace k2
