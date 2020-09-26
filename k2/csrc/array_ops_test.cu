/**
 * @brief
 * ops_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/math.h"
#include "k2/csrc/timer.h"

namespace k2 {

template <typename T>
void MatrixTanspose(int32_t num_rows, int32_t num_cols, const T *src, T *dest) {
  for (int32_t i = 0; i < num_rows; ++i) {
    for (int32_t j = 0; j < num_cols; ++j) {
      dest[j * num_rows + i] = src[i * num_cols + j];
    }
  }
}

template <typename T, DeviceType d>
void TestTranspose(int32_t num_rows, int32_t num_cols, int32_t num_reps = 1,
                   bool print_bandwidth = false) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  int32_t num_elements = num_rows * num_cols;
  std::vector<T> host_src(num_elements);
  std::iota(host_src.begin(), host_src.end(), 0);
  std::vector<T> gold(num_elements);
  MatrixTanspose<T>(num_rows, num_cols, host_src.data(), gold.data());

  int32_t num_bytes = num_elements * sizeof(T);
  auto src_region = NewRegion(context, num_bytes);
  Array2<T> src(num_rows, num_cols, num_cols, 0, src_region);
  auto kind = GetMemoryCopyKind(*cpu, *src.Context());
  MemoryCopy(static_cast<void *>(src.Data()),
             static_cast<const void *>(host_src.data()), num_bytes, kind);

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
  kind = GetMemoryCopyKind(*dest.Context(), *cpu);
  MemoryCopy(static_cast<void *>(host_dest.data()),
             static_cast<const void *>(dest.Data()), num_bytes, kind);

  ASSERT_EQ(host_dest, gold);

  if (print_bandwidth) {
    // effective_bandwidth (GB/s) = (read_bytes + write_bytes) / (time_seconds *
    // 10^9), for matrix transpose, read_bytes + write_bytes = 2 * num_bytes
    printf("Average time is: %.6f s, effective bandwidth is: %.2f GB/s\n",
           elapsed / num_reps, 2 * num_bytes * 1e-9 * num_reps / elapsed);
  }
}

TEST(OpsTest, TransposeTest) {
  {
    // test with some corner cases
    std::vector<std::pair<int32_t, int32_t>> shapes = {
        {0, 0}, {1, 1}, {5, 4}, {100, 0}, {128, 64}, {15, 13}, {115, 180},
    };
    for (const auto &v : shapes) {
      TestTranspose<int32_t, kCpu>(v.first, v.second);
      TestTranspose<int32_t, kCuda>(v.first, v.second);
    }
  }

  {
    // test with random shapes
    for (int32_t i = 0; i != 20; ++i) {
      auto rows = RandInt(0, 3000);
      auto cols = RandInt(0, 3000);
      TestTranspose<int32_t, kCpu>(rows, cols);
      TestTranspose<int32_t, kCuda>(rows, cols);
    }
  }

  {
    // speed test for different data type
    // TODO(haowen): we may need to allocate different size of shared memory for
    // different data type to get the best performance
    TestTranspose<char, kCuda>(1000, 2000, 100, true);
    TestTranspose<int16_t, kCuda>(1000, 2000, 100, true);
    TestTranspose<int32_t, kCuda>(1000, 2000, 100, true);
    TestTranspose<float, kCuda>(1000, 2000, 100, true);
    TestTranspose<double, kCuda>(1000, 2000, 100, true);
  }
}

template <typename S, typename T>
void ComputeExclusiveSum(const std::vector<S> &src, std ::vector<T> *dest) {
  auto &dst = *dest;
  K2_CHECK_GE(dst.size(), src.size());
  T sum = T(0);
  auto dst_size = dst.size();
  auto src_size = src.size();
  for (auto i = 0; i != dst_size; ++i) {
    dst[i] = sum;
    if (i >= src_size) break;
    sum += src[i];
  }
}

template <typename S, typename T>
void CheckExclusiveSumArray1Result(const std::vector<S> &src_data,
                                   const Array1<T> &dest) {
  // copy data from CPU to CPU/GPU
  std::vector<T> dest_data(dest.Dim());
  auto kind = GetMemoryCopyKind(*dest.Context(), *GetCpuContext());
  MemoryCopy(static_cast<void *>(dest_data.data()),
             static_cast<const void *>(dest.Data()),
             dest.Dim() * dest.ElementSize(), kind);
  std::vector<T> expected_data(dest.Dim());
  ComputeExclusiveSum(src_data, &expected_data);
  ASSERT_EQ(dest_data.size(), expected_data.size());
  for (auto i = 0; i != dest_data.size(); ++i) {
    EXPECT_EQ(dest_data[i], expected_data[i]);
  }
}

template <typename S, typename T, DeviceType d>
void TestExclusiveSumArray1(int32_t num_elem) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // Test ExclusiveSum(Array1<T> &src)
    std::vector<S> data(num_elem);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    Array1<S> src(context, data);
    Array1<S> dest = ExclusiveSum(src);
    CheckExclusiveSumArray1Result(data, dest);
  }

  {
    // Test ExclusiveSum(Array1<S> &src, Array1<T> *dest) with
    // dest.Dim() == src.Dim()
    std::vector<S> data(num_elem);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    Array1<S> src(context, data);
    Array1<T> dest(context, num_elem);
    ExclusiveSum(src, &dest);
    CheckExclusiveSumArray1Result(data, dest);
  }

  {
    // Test ExclusiveSum(Array1<T> &src, Array1<T> *dest) where
    // &dest = &src
    std::vector<S> data(num_elem);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    Array1<S> src(context, data);
    ExclusiveSum(src, &src);
    CheckExclusiveSumArray1Result(data, src);
  }

  {
    // Test ExclusiveSum(Array1<T> &src, Array1<T> *dest) with
    // dest.Dim() == src.Dim() + 1
    int32_t src_dim = num_elem - 1;
    std::vector<S> data(src_dim);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    // note we allocate one extra element in region for `src`,
    // but its value will not be set.
    int32_t num_bytes = num_elem * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    S *region_data = region->GetData<S, d>();
    auto kind = GetMemoryCopyKind(*cpu, *region->context);
    MemoryCopy(static_cast<void *>(region_data),
               static_cast<const void *>(data.data()), src_dim * sizeof(T),
               kind);
    Array1<S> src(src_dim, region, 0);
    Array1<T> dest(context, num_elem);
    ASSERT_EQ(dest.Dim(), src.Dim() + 1);
    ExclusiveSum(src, &dest);
    CheckExclusiveSumArray1Result(data, dest);
  }

  {
    // Test ExclusiveSumDeref(Array1<S*> &src, Array1<S> *dest) with
    // dest.Dim() == src.Dim()
    std::vector<S> data(num_elem);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    Array1<S> src(context, data);
    S *src_data = src.Data();
    auto lambda_set_values = [=] __host__ __device__(int32_t i) -> S * {
      return &src_data[i];
    };
    Array1<S *> src_ptr(context, num_elem, lambda_set_values);
    Array1<S> dest(context, num_elem);
    ExclusiveSumDeref(src_ptr, &dest);
    CheckExclusiveSumArray1Result(data, dest);
  }

  {
    // Test ExclusiveSumDeref(Array1<S*> &src, Array1<S> *dest) with
    // dest.Dim() == src.Dim() + 1
    int32_t src_dim = num_elem - 1;
    std::vector<S> data(num_elem);
    int32_t start = RandInt(0, 2);
    std::iota(data.begin(), data.end(), static_cast<S>(start));
    // note we allocate one extra element in region for `src`,
    // but its value will not be set.
    Array1<S> src(context, data);
    S *src_data = src.Data();
    int32_t num_bytes = num_elem * sizeof(S *);
    RegionPtr region = NewRegion(context, num_bytes);
    S **region_data = region->GetData<S *, d>();
    auto lambda_set_values = [=] __host__ __device__(int32_t i) -> void {
      region_data[i] = &src_data[i];
    };
    Eval(context, num_elem, lambda_set_values);
    // not src_ptr.Dim() == src_dim == num_elem - 1
    Array1<S *> src_ptr(src_dim, region, 0);
    Array1<S> dest(context, num_elem);
    ASSERT_EQ(dest.Dim(), src_ptr.Dim() + 1);
    ExclusiveSumDeref(src_ptr, &dest);
    CheckExclusiveSumArray1Result(data, dest);
  }
}

TEST(OpsTest, ExclusiveSumArray1Test) {
  TestExclusiveSumArray1<int32_t, int32_t, kCpu>(1000);
  TestExclusiveSumArray1<int32_t, int32_t, kCuda>(1000);
  TestExclusiveSumArray1<float, double, kCpu>(1000);
  TestExclusiveSumArray1<float, double, kCuda>(1000);
}

template <typename T>
void ComputeExclusiveSumArray2(const std::vector<T> &src, int32_t dest_rows,
                               int32_t dest_cols, std ::vector<T> *dest,
                               int32_t axis) {
  auto &dst = *dest;
  int32_t src_num_elems = static_cast<int32_t>(src.size());
  if (axis == 0) {
    if (dst.size() > src.size()) {
      // dst.rows == src.rows + 1
      K2_CHECK_EQ(src.size(), dest_cols * (dest_rows - 1));
    }
    for (int32_t j = 0; j != dest_cols; ++j) {
      T sum = T(0);
      for (auto i = 0; i != dest_rows; ++i) {
        int32_t dest_pos = i * dest_cols + j;
        dst[dest_pos] = sum;
        int32_t src_pos = i * dest_cols + j;  // src_cols == dest_cols
        if (src_pos >= src_num_elems) break;
        sum += src[src_pos];
      }
    }
  } else {
    K2_CHECK_EQ(axis, 1);
    int32_t src_cols = dest_cols;
    if (dst.size() > src.size()) {
      // dst.cols == src.cols + 1
      K2_CHECK_EQ(src.size(), dest_rows * (dest_cols - 1));
      src_cols = dest_cols - 1;
    }
    for (int32_t i = 0; i != dest_rows; ++i) {
      T sum = T(0);
      for (auto j = 0; j != dest_cols; ++j) {
        int32_t dest_pos = i * dest_cols + j;
        dst[dest_pos] = sum;
        int32_t src_pos = i * src_cols + j;
        if (src_pos >= src_num_elems) break;
        sum += src[src_pos];
      }
    }
  }
}

template <typename T>
void CheckExclusiveSumArray2Result(const std::vector<T> &src_data,
                                   Array2<T> &dest, int32_t axis) {
  int32_t dest_rows = dest.Dim0();
  int32_t dest_cols = dest.Dim1();
  // just make test code simple by call `Flatten` even though it's not so
  // efficient.
  Array1<T> dest_array1 = dest.Flatten();
  // copy data from CPU to CPU/GPU
  std::vector<T> dest_data(dest_rows * dest_cols);
  auto kind = GetMemoryCopyKind(*dest_array1.Context(), *GetCpuContext());
  MemoryCopy(static_cast<void *>(dest_data.data()),
             static_cast<const void *>(dest_array1.Data()),
             dest_array1.Dim() * dest_array1.ElementSize(), kind);
  std::vector<T> expected_data(dest_rows * dest_cols);
  ComputeExclusiveSumArray2(src_data, dest_rows, dest_cols, &expected_data,
                            axis);
  EXPECT_EQ(dest_data, expected_data);
}

template <typename T, DeviceType d>
void TestExclusiveSumArray2(int32_t rows, int32_t cols) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // axis == 0 && &dest == &src, ElementStride0 == cols
    int32_t axis = 0;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> src_array1(context, data);
    Array2<T> src(src_array1, rows, cols);
    ExclusiveSum(src, &src, axis);
    CheckExclusiveSumArray2Result(data, src, axis);
  }

  {
    // axis == 0 && &dest == &src, ElementStride0 > cols
    int32_t axis = 0;
    int32_t stride0 = RandInt(cols + 1, cols + 10);
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> data_array(context, data);
    const T *src_data = data_array.Data();
    // allocate extra memory as there stride0 > 0
    int32_t num_bytes = rows * stride0 * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    T *region_data = region->GetData<T, d>();
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      region_data[i * stride0 + j] = src_data[i * cols + j];
    };
    Eval2(context, rows, cols, lambda_set_elems);
    Array2<T> src(rows, cols, stride0, 0, region);
    ExclusiveSum(src, &src, axis);
    CheckExclusiveSumArray2Result(data, src, axis);
  }

  {
    // axis == 0 && dest.Dim0() == src.Dim0(), ElementStride0 == cols
    int32_t axis = 0;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> src_array1(context, data);
    Array2<T> src(src_array1, rows, cols);
    Array2<T> dest(context, rows, cols);
    ExclusiveSum(src, &dest, axis);
    CheckExclusiveSumArray2Result(data, dest, axis);
  }

  {
    // axis == 0 && dest.Dim0() == src.Dim0(), ElementStride0 > cols
    int32_t axis = 0;
    int32_t stride0 = RandInt(cols + 1, cols + 10);
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> data_array(context, data);
    const T *src_data = data_array.Data();
    // allocate extra memory as there stride0 > 0
    int32_t num_bytes = rows * stride0 * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    T *region_data = region->GetData<T, d>();
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      region_data[i * stride0 + j] = src_data[i * cols + j];
    };
    Eval2(context, rows, cols, lambda_set_elems);
    Array2<T> src(rows, cols, stride0, 0, region);
    Array2<T> dest(context, rows, cols);
    ExclusiveSum(src, &dest, axis);
    CheckExclusiveSumArray2Result(data, dest, axis);
  }

  {
    // axis == 0 && dest.Dim0() == src.Dim0() + 1, we need to allocate one extra
    // element for src
    int32_t axis = 0;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> data_array(context, data);
    const T *src_data = data_array.Data();
    int32_t num_bytes = (rows * cols + 1) * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    T *region_data = region->GetData<T, d>();
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      region_data[i * cols + j] = src_data[i * cols + j];
    };
    Eval2(context, rows, cols, lambda_set_elems);
    Array2<T> src(rows, cols, cols, 0, region);
    {
      // dest.stride0 == dest.cols
      Array2<T> dest(context, rows + 1, cols);
      ExclusiveSum(src, &dest, axis);
      CheckExclusiveSumArray2Result(data, dest, axis);
    }
    {
      // dest.stride0 > dest.cols
      int32_t dest_stride0 = cols + 5;
      int32_t dest_rows = rows + 1;
      RegionPtr dest_region =
          NewRegion(context, dest_rows * dest_stride0 * sizeof(T));
      Array2<T> dest(dest_rows, cols, dest_stride0, 0, dest_region);
      ExclusiveSum(src, &dest, axis);
      CheckExclusiveSumArray2Result(data, dest, axis);
    }
  }

  {
    // axis == 1 && &dest == &src, ElementStride0 == cols
    int32_t axis = 1;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> src_array1(context, data);
    Array2<T> src(src_array1, rows, cols);
    ExclusiveSum(src, &src, axis);
    CheckExclusiveSumArray2Result(data, src, axis);
  }

  {
    // axis == 1 && &dest == &src, ElementStride0 > cols
    int32_t axis = 1;
    int32_t stride0 = RandInt(cols + 1, cols + 10);
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> data_array(context, data);
    const T *src_data = data_array.Data();
    // allocate extra memory as there stride0 > 0
    int32_t num_bytes = rows * stride0 * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    T *region_data = region->GetData<T, d>();
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      region_data[i * stride0 + j] = src_data[i * cols + j];
    };
    Eval2(context, rows, cols, lambda_set_elems);
    Array2<T> src(rows, cols, stride0, 0, region);
    ExclusiveSum(src, &src, axis);
    CheckExclusiveSumArray2Result(data, src, axis);
  }

  {
    // axis == 1 && dest.Dim1() == src.Dim1(), ElementStride0 == cols
    int32_t axis = 1;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> src_array1(context, data);
    Array2<T> src(src_array1, rows, cols);
    Array2<T> dest(context, rows, cols);
    ExclusiveSum(src, &dest, axis);
    CheckExclusiveSumArray2Result(data, dest, axis);
  }

  {
    // axis == 1 && dest.Dim1() == src.Dim1() + 1, we need to allocate one extra
    // element for src
    int32_t axis = 1;
    int32_t num_elems = rows * cols;
    std::vector<T> data(num_elems);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> data_array(context, data);
    const T *src_data = data_array.Data();
    int32_t num_bytes = (rows * cols + 1) * sizeof(T);
    RegionPtr region = NewRegion(context, num_bytes);
    T *region_data = region->GetData<T, d>();
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      region_data[i * cols + j] = src_data[i * cols + j];
    };
    Eval2(context, rows, cols, lambda_set_elems);
    Array2<T> src(rows, cols, cols, 0, region);
    {
      // dest.stride0 == dest.cols
      Array2<T> dest(context, rows, cols + 1);
      ExclusiveSum(src, &dest, axis);
      CheckExclusiveSumArray2Result(data, dest, axis);
    }
    {
      // dest.stride0 > dest.cols
      int32_t dest_stride0 = cols + 5;
      int32_t dest_cols = cols + 1;
      RegionPtr dest_region =
          NewRegion(context, rows * dest_stride0 * sizeof(T));
      Array2<T> dest(rows, dest_cols, dest_stride0, 0, dest_region);
      ExclusiveSum(src, &dest, axis);
      CheckExclusiveSumArray2Result(data, dest, axis);
    }
  }
}

TEST(OpsTest, ExclusiveSumArray2Test) {
  int32_t rows = RandInt(500, 1000);
  int32_t cols = RandInt(500, 1000);
  TestExclusiveSumArray2<int32_t, kCpu>(rows, cols);
  TestExclusiveSumArray2<int32_t, kCuda>(rows, cols);
}

template <typename T, DeviceType d>
void TestMaxPerSubListTest() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // empty case
    const std::vector<int32_t> row_splits = {0};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = 0;
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    Array1<T> values(context, 0);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    ASSERT_EQ(num_rows, 0);
    Array1<T> max_values(context, num_rows);
    // just run to check if there's any error
    MaxPerSublist(ragged, 1, &max_values);
    EXPECT_EQ(max_values.Dim(), 0);
  }

  {
    const std::vector<int32_t> row_splits = {0, 2, 2, 5, 6};
    RaggedShapeDim shape_dim;
    shape_dim.row_splits = Array1<int32_t>(context, row_splits);
    shape_dim.cached_tot_size = row_splits.back();
    std::vector<RaggedShapeDim> axes = {shape_dim};
    RaggedShape shape(axes, true);
    const std::vector<T> values_vec = {1, 3, 2, 8, 0, -1};
    Array1<T> values(context, values_vec);
    Ragged<T> ragged(shape, values);

    int32_t num_rows = ragged.shape.Dim0();
    Array1<T> max_values(context, num_rows);
    T default_value = 2;
    MaxPerSublist(ragged, default_value, &max_values);
    // copy memory from GPU/CPU to CPU
    std::vector<T> cpu_data(max_values.Dim());
    auto kind = GetMemoryCopyKind(*max_values.Context(), *cpu);
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<const void *>(max_values.Data()),
               max_values.Dim() * max_values.ElementSize(), kind);
    std::vector<T> expected_data = {3, default_value, 8, default_value};
    EXPECT_EQ(cpu_data, expected_data);
  }

  {
    // test with random large size
    const int32_t min_num_elements = 2000;
    // not random shape is on CPU
    RaggedShape shape = RandomRaggedShape(false, 2, 2, min_num_elements, 5000);
    ASSERT_EQ(shape.NumAxes(), 2);
    RaggedShape gpu_shape;
    if (d == kCuda) {
      // copy shape to GPU
      const Array1<T> &row_splits = shape.RowSplits(1);
      RaggedShapeDim shape_dim;
      shape_dim.row_splits = row_splits.To(GetCudaContext());
      shape_dim.cached_tot_size = shape.NumElements();
      std::vector<RaggedShapeDim> axes = {shape_dim};
      gpu_shape = RaggedShape(axes, true);
    }

    int32_t num_elems = shape.NumElements();
    std::vector<T> data(num_elems);
    for (int32_t i = 0; i != 10; ++i) {
      std::iota(data.begin(), data.end(), 0);
      // randomly set data[pos] = num_elems which is
      // greater than any element in data
      int32_t pos = RandInt(0, num_elems - 1);
      data[pos] = num_elems;
      // find the corresponding row
      int32_t num_rows = shape.Dim0();
      const int32_t *row_splits_data = shape.RowSplits(1).Data();
      int32_t row = 0;
      for (int32_t i = 0; i < num_rows; ++i) {
        if (pos >= row_splits_data[i] && pos < row_splits_data[i + 1]) {
          row = i;
          break;
        }
      }

      Array1<T> values(context, data);
      Ragged<T> ragged(d == kCuda ? gpu_shape : shape, values);
      Array1<T> max_values(context, num_rows);
      T default_value = 0;
      MaxPerSublist(ragged, default_value, &max_values);
      EXPECT_EQ(max_values[row], num_elems);
    }
  }
}

TEST(OpsTest, MaxPerSubListTest) {
  TestMaxPerSubListTest<int32_t, kCpu>();
  TestMaxPerSubListTest<int32_t, kCuda>();
}
template <typename T, DeviceType d>
void TestAppend() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // a case with small size
    std::vector<T> data1 = {3, 1, 2};
    std::vector<T> data2 = {5, 6, 7, 8};
    std::vector<T> data3 = {};  // empty
    std::vector<T> data4 = {9};
    std::vector<T> expected_data = {3, 1, 2, 5, 6, 7, 8, 9};

    Array1<T> array1(context, data1);
    Array1<T> array2(context, data2);
    Array1<T> array3(context, data3);
    Array1<T> array4(context, data4);

    {
      // test Append(int32_t, Array1<T>**)
      std::vector<const Array1<T> *> arrays = {&array1, &array2, &array3,
                                               &array4};
      const Array1<T> **src = arrays.data();
      Array1<T> dst = Append(4, src);
      EXPECT_EQ(dst.Dim(), 8);
      // copy memory from GPU/CPU to CPU
      std::vector<T> cpu_data(dst.Dim());
      auto kind = GetMemoryCopyKind(*dst.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(dst.Data()),
                 dst.Dim() * dst.ElementSize(), kind);
      EXPECT_EQ(cpu_data, expected_data);
    }

    {
      // test Append(int32_t, Array1<T>*)
      std::vector<Array1<T>> arrays = {array1, array2, array3, array4};
      const Array1<T> *src = arrays.data();
      Array1<T> dst = Append(4, src);
      EXPECT_EQ(dst.Dim(), 8);

      // copy memory from GPU/CPU to CPU
      std::vector<T> cpu_data(dst.Dim());
      auto kind = GetMemoryCopyKind(*dst.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(dst.Data()),
                 dst.Dim() * dst.ElementSize(), kind);
      EXPECT_EQ(cpu_data, expected_data);
    }
  }

  {
    // test with random large size, the arrays' sizes are fairly balanced.
    for (int32_t i = 0; i != 5; ++i) {
      int32_t num_array = RandInt(10, 1000);
      std::vector<Array1<T>> arrays_vec(num_array);
      std::vector<const Array1<T> *> arrays(num_array);
      int32_t total_size = 0;
      for (int32_t j = 0; j != num_array; ++j) {
        int32_t curr_array_size = RandInt(0, 10000);
        std::vector<T> data(curr_array_size);
        std::iota(data.begin(), data.end(), total_size);
        total_size += curr_array_size;
        arrays_vec[j] = Array1<T>(context, data);
        arrays[j] = &arrays_vec[j];
      }
      const Array1<T> **src = arrays.data();
      Array1<T> dst = Append(num_array, src);
      EXPECT_EQ(dst.Dim(), total_size);
      // copy memory from GPU/CPU to CPU
      std::vector<T> cpu_data(dst.Dim());
      auto kind = GetMemoryCopyKind(*dst.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(dst.Data()),
                 dst.Dim() * dst.ElementSize(), kind);
      std::vector<T> expected_data(dst.Dim());
      std::iota(expected_data.begin(), expected_data.end(), 0);
      EXPECT_EQ(cpu_data, expected_data);
    }
  }

  {
    // test with random large size: the arrays' sizes are not balanced.
    for (int32_t i = 0; i != 5; ++i) {
      int32_t num_array = RandInt(10, 1000);
      std::vector<Array1<T>> arrays_vec(num_array);
      std::vector<const Array1<T> *> arrays(num_array);
      int32_t total_size = 0, max_size = 0;
      // notice `j != num_array - 1`, we would push a very long array
      // after the loop
      for (int32_t j = 0; j != num_array - 1; ++j) {
        int32_t curr_array_size = RandInt(0, 10000);
        std::vector<T> data(curr_array_size);
        std::iota(data.begin(), data.end(), total_size);
        total_size += curr_array_size;
        arrays_vec[j] = Array1<T>(context, data);
        arrays[j] = &arrays_vec[j];
        if (curr_array_size > max_size) max_size = curr_array_size;
      }
      // generate an array with very large size
      {
        int32_t average_size = total_size / num_array;
        int32_t long_size = average_size * 10;
        std::vector<T> data(long_size);
        std::iota(data.begin(), data.end(), total_size);
        total_size += long_size;
        arrays_vec[num_array - 1] = Array1<T>(context, data);
        arrays[num_array - 1] = &arrays_vec[num_array - 1];
      }
      const Array1<T> **src = arrays.data();
      Array1<T> dst = Append(num_array, src);
      EXPECT_EQ(dst.Dim(), total_size);
      // copy memory from GPU/CPU to CPU
      std::vector<T> cpu_data(dst.Dim());
      auto kind = GetMemoryCopyKind(*dst.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(dst.Data()),
                 dst.Dim() * dst.ElementSize(), kind);
      std::vector<T> expected_data(dst.Dim());
      std::iota(expected_data.begin(), expected_data.end(), 0);
      EXPECT_EQ(cpu_data, expected_data);
    }
  }
}

TEST(OpsTest, AppendTest) {
  TestAppend<int32_t, kCpu>();
  TestAppend<int32_t, kCuda>();
  TestAppend<float, kCpu>();
  TestAppend<float, kCuda>();
}

}  // namespace k2
