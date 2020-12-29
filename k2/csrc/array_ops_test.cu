/**
 * @brief
 * ops_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu, Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/test_utils.h"
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
  cpu->CopyDataTo(num_bytes, host_src.data(), src.Context(), src.Data());

  auto dest_region = NewRegion(context, num_bytes);
  Array2<T> dest(num_cols, num_rows, num_rows, 0, dest_region);

  // warm up in case that the first kernel launch takes longer time.
  Transpose<T>(context, src, &dest);

  Timer t(dest.Context());
  for (int32_t i = 0; i < num_reps; ++i) {
    Transpose<T>(context, src, &dest);
  }
  double elapsed = t.Elapsed();

  std::vector<T> host_dest(num_elements);
  dest.Context()->CopyDataTo(num_bytes, dest.Data(), cpu, host_dest.data());

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
    for (int32_t i = 0; i != 5; ++i) {
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
  size_t dst_size = dst.size();
  size_t src_size = src.size();
  for (size_t i = 0; i != dst_size; ++i) {
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
  dest.Context()->CopyDataTo(dest.Dim() * dest.ElementSize(), dest.Data(),
                             GetCpuContext(), dest_data.data());
  std::vector<T> expected_data(dest.Dim());
  ComputeExclusiveSum(src_data, &expected_data);
  ASSERT_EQ(dest_data.size(), expected_data.size());
  for (size_t i = 0; i != dest_data.size(); ++i) {
    EXPECT_EQ(dest_data[i], expected_data[i]);
  }
}

template <typename S, typename T>
void TestExclusiveSumArray1(int32_t num_elem) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
      cpu->CopyDataTo(src_dim * sizeof(T), data.data(), region->context,
                      region->data);
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
      Array1<const S *> src_ptr(context, num_elem);
      const S **src_ptr_data = src_ptr.Data();
      K2_EVAL(context, num_elem, lambda_set_values, (int32_t i) -> void {
          src_ptr_data[i] = src_data + i;
        });
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
      S **region_data = region->GetData<S *>();
      K2_EVAL(
          context, num_elem, lambda_set_values,
          (int32_t i)->void { region_data[i] = &src_data[i]; });
      // not src_ptr.Dim() == src_dim == num_elem - 1
      Array1<const S *> src_ptr(src_dim, region, 0);
      Array1<S> dest(context, num_elem);
      ASSERT_EQ(dest.Dim(), src_ptr.Dim() + 1);
      ExclusiveSumDeref(src_ptr, &dest);
      CheckExclusiveSumArray1Result(data, dest);
    }
  }
}

TEST(OpsTest, ExclusiveSumArray1Test) {
  TestExclusiveSumArray1<int32_t, int32_t>(1000);
  TestExclusiveSumArray1<float, double>(1000);
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
      K2_CHECK_EQ((int32_t)src.size(), dest_cols * (dest_rows - 1));
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
      K2_CHECK_EQ((int32_t)src.size(), dest_rows * (dest_cols - 1));
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
  dest_array1.Context()->CopyDataTo(
      dest_array1.Dim() * dest_array1.ElementSize(), dest_array1.Data(),
      GetCpuContext(), dest_data.data());
  std::vector<T> expected_data(dest_rows * dest_cols);
  ComputeExclusiveSumArray2(src_data, dest_rows, dest_cols, &expected_data,
                            axis);
  EXPECT_EQ(dest_data, expected_data);
}

template <typename T>
void TestExclusiveSumArray2(int32_t rows, int32_t cols) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
      T *region_data = region->GetData<T>();
      K2_EVAL2(
          context, rows, cols, lambda_set_elems, (int32_t i, int32_t j)->void {
            region_data[i * stride0 + j] = src_data[i * cols + j];
          });
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
      T *region_data = region->GetData<T>();
      K2_EVAL2(
          context, rows, cols, lambda_set_elems, (int32_t i, int32_t j)->void {
            region_data[i * stride0 + j] = src_data[i * cols + j];
          });
      Array2<T> src(rows, cols, stride0, 0, region);
      Array2<T> dest(context, rows, cols);
      ExclusiveSum(src, &dest, axis);
      CheckExclusiveSumArray2Result(data, dest, axis);
    }

    {
      // axis == 0 && dest.Dim0() == src.Dim0() + 1, we need to allocate one
      // extra element for src
      int32_t axis = 0;
      int32_t num_elems = rows * cols;
      std::vector<T> data(num_elems);
      std::iota(data.begin(), data.end(), 0);
      Array1<T> data_array(context, data);
      const T *src_data = data_array.Data();
      int32_t num_bytes = (rows * cols + 1) * sizeof(T);
      RegionPtr region = NewRegion(context, num_bytes);
      T *region_data = region->GetData<T>();
      K2_EVAL2(
          context, rows, cols, lambda_set_elems, (int32_t i, int32_t j)->void {
            region_data[i * cols + j] = src_data[i * cols + j];
          });
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
      T *region_data = region->GetData<T>();
      K2_EVAL2(
          context, rows, cols, lambda_set_elems, (int32_t i, int32_t j)->void {
            region_data[i * stride0 + j] = src_data[i * cols + j];
          });
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
      // axis == 1 && dest.Dim1() == src.Dim1() + 1, we need to allocate one
      // extra element for src
      int32_t axis = 1;
      int32_t num_elems = rows * cols;
      std::vector<T> data(num_elems);
      std::iota(data.begin(), data.end(), 0);
      Array1<T> data_array(context, data);
      const T *src_data = data_array.Data();
      int32_t num_bytes = (rows * cols + 1) * sizeof(T);
      RegionPtr region = NewRegion(context, num_bytes);
      T *region_data = region->GetData<T>();
      K2_EVAL2(
          context, rows, cols, lambda_set_elems, (int32_t i, int32_t j)->void {
            region_data[i * cols + j] = src_data[i * cols + j];
          });
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
}

TEST(OpsTest, ExclusiveSumArray2Test) {
  int32_t rows = RandInt(500, 1000);
  int32_t cols = RandInt(500, 1000);
  TestExclusiveSumArray2<int32_t>(rows, cols);
}

template <typename T>
void TestArrayMaxAndOr() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // Max
      const std::vector<T> values = {1, 3, 2, 8, 0, -1};
      Array1<T> src(context, values);
      Array1<T> dst(context, 1);
      T default_value = 0;
      Max(src, default_value, &dst);
      EXPECT_EQ(dst[0], 8);
    }

    {
      // Max, dst is one of element of src
      const std::vector<T> values = {1, 3, 2, 8, 0, -1};
      Array1<T> src(context, values);
      Array1<T> dst = src.Range(2, 1);
      T default_value = 0;
      Max(src, default_value, &dst);
      EXPECT_EQ(dst[0], 8);
      // src has been changed as well
      EXPECT_EQ(src[2], 8);
      // other values are not changed
      src = src.To(cpu);
      std::vector<T> cpu_data(src.Data(), src.Data() + src.Dim());
      const std::vector<T> expected_data = {1, 3, 8, 8, 0, -1};
      EXPECT_EQ(cpu_data, expected_data);
    }

    {
      // Max, with random large size
      int32_t num_elems = RandInt(1000, 10000);
      std::vector<T> data(num_elems);
      std::iota(data.begin(), data.end(), num_elems);
      // random set a value to  `max_value`
      int32_t pos = RandInt(0, num_elems - 1);
      T max_value = static_cast<T>(num_elems * 2);
      data[pos] = max_value;
      Array1<T> src(context, data);
      Array1<T> dst(context, 1);
      T default_value = 0;
      Max(src, default_value, &dst);
      EXPECT_EQ(dst[0], max_value);
    }

    {
      // And
      const std::vector<T> values = {3, 6, 11};
      Array1<T> src(context, values);
      Array1<T> dst(context, 1);
      T default_value = -1;
      And(src, default_value, &dst);
      EXPECT_EQ(dst[0], 2);
    }

    {
      // Or
      const std::vector<T> values = {3, 6, 4};
      Array1<T> src(context, values);
      Array1<T> dst(context, 1);
      T default_value = 0;
      Or(src, default_value, &dst);
      EXPECT_EQ(dst[0], 7);
    }
  }
}

TEST(OpsTest, ArrayMaxAndOrTest) { TestArrayMaxAndOr<int32_t>(); }

template <typename T>
void TestAppend() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
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
        dst.Context()->CopyDataTo(dst.Dim() * dst.ElementSize(), dst.Data(),
                                  cpu, cpu_data.data());
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
        dst.Context()->CopyDataTo(dst.Dim() * dst.ElementSize(), dst.Data(),
                                  cpu, cpu_data.data());
        EXPECT_EQ(cpu_data, expected_data);
      }
    }

    {
      // test with random large size, the arrays' sizes are fairly balanced.
      for (int32_t i = 0; i != 2; ++i) {
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
        dst.Context()->CopyDataTo(dst.Dim() * dst.ElementSize(), dst.Data(),
                                  cpu, cpu_data.data());
        std::vector<T> expected_data(dst.Dim());
        std::iota(expected_data.begin(), expected_data.end(), 0);
        EXPECT_EQ(cpu_data, expected_data);
      }
    }

    {
      // test with random large size: the arrays' sizes are not balanced.
      for (int32_t i = 0; i != 2; ++i) {
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
        dst.Context()->CopyDataTo(dst.Dim() * dst.ElementSize(), dst.Data(),
                                  cpu, cpu_data.data());
        std::vector<T> expected_data(dst.Dim());
        std::iota(expected_data.begin(), expected_data.end(), 0);
        EXPECT_EQ(cpu_data, expected_data);
      }
    }
  }
}

TEST(OpsTest, AppendTest) {
  TestAppend<int32_t>();
  TestAppend<float>();
}

TEST(OpsTest, SpliceRowSplitsTest) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // a case with small size
      std::vector<int32_t> data1 = {0, 2, 5};
      std::vector<int32_t> data2 = {0, 2, 2, 3};
      std::vector<int32_t> data3 = {0};
      std::vector<int32_t> data4 = {0, 3};
      std::vector<int32_t> expected_data = {0, 2, 5, 7, 7, 8, 11};

      Array1<int32_t> array1(context, data1);
      Array1<int32_t> array2(context, data2);
      Array1<int32_t> array3(context, data3);
      Array1<int32_t> array4(context, data4);

      std::vector<const Array1<int32_t> *> arrays = {&array1, &array2, &array3,
                                                     &array4};
      const Array1<int32_t> **src = arrays.data();
      Array1<int32_t> dst = SpliceRowSplits(4, src);
      EXPECT_EQ(dst.Dim(), expected_data.size());
      // copy memory from GPU/CPU to CPU
      dst = dst.To(cpu);
      std::vector<int32_t> cpu_data(dst.Data(), dst.Data() + dst.Dim());
      EXPECT_EQ(cpu_data, expected_data);
    }

    {
      // test with random large size, the arrays' sizes are fairly balanced.
      for (int32_t i = 0; i != 2; ++i) {
        int32_t num_array = RandInt(10, 1000);
        std::vector<Array1<int32_t>> arrays_vec(num_array);
        std::vector<const Array1<int32_t> *> arrays(num_array);
        std::vector<int32_t> expected_data;
        int32_t data_offset = 0;
        for (int32_t j = 0; j != num_array; ++j) {
          int32_t curr_array_size = RandInt(0, 10000);
          RaggedShape shape =
              RandomRaggedShape(false, 2, 2, curr_array_size, curr_array_size);
          ASSERT_EQ(shape.NumAxes(), 2);
          Array1<int32_t> cpu_row_splits = shape.RowSplits(1).To(cpu);
          int32_t num_splits = cpu_row_splits.Dim();
          ASSERT_GE(num_splits, 1);
          const int32_t *splits_data = cpu_row_splits.Data();
          for (int32_t n = 0; n < num_splits; ++n) {
            expected_data.push_back(splits_data[n] + data_offset);
          }
          if (j + 1 < num_array) expected_data.pop_back();
          data_offset += splits_data[num_splits - 1];
          Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
          ASSERT_GE(row_splits.Dim(), 1);
          arrays_vec[j] = row_splits;
          arrays[j] = &arrays_vec[j];
        }
        const Array1<int32_t> **src = arrays.data();
        Array1<int32_t> dst = SpliceRowSplits(num_array, src);
        EXPECT_EQ(dst.Dim(), expected_data.size());
        // copy memory from GPU/CPU to CPU
        dst = dst.To(cpu);
        std::vector<int32_t> cpu_data(dst.Data(), dst.Data() + dst.Dim());
        EXPECT_EQ(cpu_data, expected_data);
      }
    }

    {
      // test with random large size: the arrays' sizes are not balanced.
      for (int32_t i = 0; i != 2; ++i) {
        int32_t num_array = RandInt(10, 1000);
        std::vector<Array1<int32_t>> arrays_vec(num_array);
        std::vector<const Array1<int32_t> *> arrays(num_array);
        std::vector<int32_t> expected_data;
        int32_t data_offset = 0;
        int32_t max_size = 0;
        for (int32_t j = 0; j != num_array - 1; ++j) {
          int32_t curr_array_size = RandInt(0, 10000);
          RaggedShape shape =
              RandomRaggedShape(false, 2, 2, curr_array_size, curr_array_size);
          ASSERT_EQ(shape.NumAxes(), 2);
          Array1<int32_t> cpu_row_splits = shape.RowSplits(1).To(cpu);
          int32_t num_splits = cpu_row_splits.Dim();
          ASSERT_GE(num_splits, 1);
          const int32_t *splits_data = cpu_row_splits.Data();
          for (int32_t n = 0; n < num_splits; ++n) {
            expected_data.push_back(splits_data[n] + data_offset);
          }
          expected_data.pop_back();
          data_offset += splits_data[num_splits - 1];
          Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
          ASSERT_GE(row_splits.Dim(), 1);
          arrays_vec[j] = row_splits;
          arrays[j] = &arrays_vec[j];
          if (num_splits > max_size) max_size = num_splits;
        }
        // generate an array with very large size
        {
          int32_t total_size = static_cast<int32_t>(expected_data.size());
          int32_t average_size = total_size / num_array;
          int32_t long_size = average_size * 10;

          RaggedShape shape =
              RandomRaggedShape(false, 2, 2, long_size, long_size);
          ASSERT_EQ(shape.NumAxes(), 2);
          Array1<int32_t> cpu_row_splits = shape.RowSplits(1).To(cpu);
          int32_t num_splits = cpu_row_splits.Dim();
          ASSERT_GE(num_splits, 1);
          const int32_t *splits_data = cpu_row_splits.Data();
          for (int32_t n = 0; n < num_splits; ++n) {
            expected_data.push_back(splits_data[n] + data_offset);
          }
          Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
          ASSERT_GE(row_splits.Dim(), 1);
          arrays_vec[num_array - 1] = row_splits;
          arrays[num_array - 1] = &arrays_vec[num_array - 1];
        }
        const Array1<int32_t> **src = arrays.data();
        Array1<int32_t> dst = SpliceRowSplits(num_array, src);
        EXPECT_EQ(dst.Dim(), expected_data.size());
        // copy memory from GPU/CPU to CPU
        dst = dst.To(cpu);
        std::vector<int32_t> cpu_data(dst.Data(), dst.Data() + dst.Dim());
        EXPECT_EQ(cpu_data, expected_data);
      }
    }
  }
}

template <typename T>
void TestRangeAndRandomArray1() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // test Range with small size
      Array1<T> result = Range<T>(context, 6, 3, 2);
      const std::vector<T> values = {3, 5, 7, 9, 11, 13};
      result = result.To(cpu);
      std::vector<T> cpu_data(result.Data(), result.Data() + result.Dim());
      EXPECT_EQ(cpu_data, values);
    }

    {
      // test Range with random large size
      int32_t num_elems = RandInt(1000, 10000);
      std::vector<T> data(num_elems);
      std::iota(data.begin(), data.end(), 0);
      Array1<T> result = Range<T>(context, num_elems, 0);
      result = result.To(cpu);
      std::vector<T> cpu_data(result.Data(), result.Data() + result.Dim());
      EXPECT_EQ(cpu_data, data);
    }

    {
      // test RandUniformArray1
      Array1<T> result = RandUniformArray1<T>(context, 1000, 0, 10000);
      result = result.To(cpu);
    }
  }
}

TEST(OpsTest, RangeTest) {
  TestRangeAndRandomArray1<int32_t>();
  TestRangeAndRandomArray1<float>();
  TestRangeAndRandomArray1<double>();
}

TEST(OpsTest, ValidateRowSplitsAndIdsTest) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // test RowSplitsToRowIds and RowIdsToRowSplits
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      {
        Array1<int32_t> row_splits(context, row_splits_vec);
        Array1<int32_t> row_ids(context, row_ids_vec.size());
        RowSplitsToRowIds(row_splits, &row_ids);
        row_ids = row_ids.To(cpu);
        std::vector<int32_t> cpu_data(row_ids.Data(),
                                      row_ids.Data() + row_ids.Dim());
        EXPECT_EQ(cpu_data, row_ids_vec);
      }
      {
        Array1<int32_t> row_ids(context, row_ids_vec);
        Array1<int32_t> row_splits(context, row_splits_vec.size());
        RowIdsToRowSplits(row_ids, &row_splits);
        row_splits = row_splits.To(cpu);
        std::vector<int32_t> cpu_data(row_splits.Data(),
                                      row_splits.Data() + row_splits.Dim());
        EXPECT_EQ(cpu_data, row_splits_vec);
      }
    }

    {
      // empty case for row splits and row ids
      const std::vector<int32_t> row_splits_vec;
      const std::vector<int32_t> row_ids_vec;
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_FALSE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // valid case for row splits and row ids
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_TRUE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // valid case for row splits and row ids with random size
      for (int32_t i = 0; i != 5; ++i) {
        RaggedShape shape = RandomRaggedShape(true, 2, 2, 2000, 10000);
        ASSERT_EQ(shape.NumAxes(), 2);
        // note shape is on CPU
        Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
        Array1<int32_t> row_ids = shape.RowIds(1).To(context);

        EXPECT_TRUE(ValidateRowSplits(row_splits));
        EXPECT_TRUE(ValidateRowIds(row_ids));
        EXPECT_TRUE(ValidateRowSplitsAndIds(row_splits, row_ids));
      }
    }

    {
      // provided tmp storage
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);

      {
        Array1<int32_t> tmp(context, 3, 2);
        EXPECT_TRUE(ValidateRowSplits(row_splits, &tmp));
        // check elments
        tmp = tmp.To(cpu);
        std::vector<int32_t> cpu_data(tmp.Data(), tmp.Data() + tmp.Dim());
        EXPECT_THAT(cpu_data, ::testing::ElementsAre(0, 2, 2));
      }

      {
        Array1<int32_t> tmp(context, 3, 2);
        EXPECT_TRUE(ValidateRowIds(row_ids, &tmp));
        // check elments
        tmp = tmp.To(cpu);
        std::vector<int32_t> cpu_data(tmp.Data(), tmp.Data() + tmp.Dim());
        EXPECT_THAT(cpu_data, ::testing::ElementsAre(0, 2, 2));
      }

      {
        Array1<int32_t> tmp(context, 3, 2);
        EXPECT_TRUE(ValidateRowSplitsAndIds(row_splits, row_ids, &tmp));
        // check elments
        tmp = tmp.To(cpu);
        std::vector<int32_t> cpu_data(tmp.Data(), tmp.Data() + tmp.Dim());
        EXPECT_THAT(cpu_data, ::testing::ElementsAre(0, 2, 2));
      }
    }

    {
      // bad case for row splits, not starts with 0
      const std::vector<int32_t> row_splits_vec = {1,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_FALSE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case for row splits, contains negative value
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  -5, 8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_FALSE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case for row splits, not non-decreasing
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  1,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_FALSE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case row ids, contains negative value
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, -2, 3, 3, 3,
                                                4, 5, 5, 5, 6,  7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_FALSE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case row ids, not non-decreasing
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 6, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_FALSE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case row ids and row splits don't agree with each other
      // i < row_splits[row_ids[i]]
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 8, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // another bad case that row ids and row splits don't agree with each
      // other i > = row_splits[row_ids[i]]
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 5, 7, 7, 9};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }

    {
      // bad case for row ids, num_elems != row_splits[-1]
      const std::vector<int32_t> row_splits_vec = {0,  2,  3,  5,  8, 9,
                                                   12, 13, 15, 15, 16};
      const std::vector<int32_t> row_ids_vec = {0, 0, 1, 2, 2, 3, 3, 3,
                                                4, 5, 5, 5, 6, 7, 7};
      Array1<int32_t> row_ids(context, row_ids_vec);
      Array1<int32_t> row_splits(context, row_splits_vec);
      EXPECT_TRUE(ValidateRowSplits(row_splits));
      EXPECT_TRUE(ValidateRowIds(row_ids));
      EXPECT_FALSE(ValidateRowSplitsAndIds(row_splits, row_ids));
    }
  }
}

TEST(OpsTest, GetCountsTest) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      int32_t n = 0;
      std::vector<int32_t> values;
      Array1<int32_t> src(context, values);
      Array1<int32_t> ans = GetCounts(src, n);
      EXPECT_EQ(ans.Dim(), 0);
    }

    {
      // simple case
      int32_t n = 8;
      std::vector<int32_t> values = {0, 1, 2, 1, 5, 5, 7, 6, 3, 2};
      std::vector<int32_t> expected_data = {1, 2, 2, 1, 0, 2, 1, 1};
      Array1<int32_t> src(context, values);
      Array1<int32_t> ans = GetCounts(src, n);
      ans = ans.To(cpu);
      std::vector<int32_t> data(ans.Data(), ans.Data() + ans.Dim());
      EXPECT_EQ(data, expected_data);
    }

    {
      // random large case
      for (int32_t i = 0; i != 2; ++i) {
        int32_t n = RandInt(1, 10000);
        int32_t src_dim = RandInt(0, 10000);
        Array1<int32_t> src = RandUniformArray1(context, src_dim, 0, n - 1);
        Array1<int32_t> ans = GetCounts(src, n);
        ans = ans.To(cpu);
        std::vector<int32_t> data(ans.Data(), ans.Data() + ans.Dim());
        src = src.To(cpu);
        int32_t *src_data = src.Data();
        std::vector<int32_t> expected_data(n, 0);
        for (int32_t j = 0; j < src.Dim(); ++j) {
          ++expected_data[src_data[j]];
        }
        EXPECT_EQ(data, expected_data);
      }
    }
  }
}

template <typename S, typename T>
void TestMonotonicLowerBound() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      std::vector<S> values;
      Array1<S> src(context, values);
      Array1<T> dest(context, 0);
      MonotonicLowerBound(src, &dest);
      EXPECT_EQ(dest.Dim(), 0);
    }

    {
      // simple case
      std::vector<S> values = {2, 1, 3, 7, 5, 8, 20, 15};
      std::vector<T> expected_data = {1, 1, 3, 5, 5, 8, 15, 15};
      ASSERT_EQ(values.size(), expected_data.size());
      Array1<S> src(context, values);
      Array1<T> dest(context, static_cast<int32_t>(values.size()));
      MonotonicLowerBound(src, &dest);
      dest = dest.To(cpu);
      std::vector<T> data(dest.Data(), dest.Data() + dest.Dim());
      EXPECT_EQ(data, expected_data);
    }

    {
      // simple case with dest = &src
      std::vector<S> values = {2, 1, 3, 7, 5, 8, 20, 15};
      std::vector<T> expected_data = {1, 1, 3, 5, 5, 8, 15, 15};
      ASSERT_EQ(values.size(), expected_data.size());
      Array1<S> src(context, values);
      MonotonicLowerBound(src, &src);
      src = src.To(cpu);
      std::vector<T> data(src.Data(), src.Data() + src.Dim());
      EXPECT_EQ(data, expected_data);
    }

    {
      // random large case
      for (int32_t i = 0; i != 2; ++i) {
        int32_t n = RandInt(1, 10000);
        int32_t src_dim = RandInt(0, 10000);
        Array1<S> src = RandUniformArray1(context, src_dim, 0, n - 1);
        Array1<T> dest(context, src_dim);
        MonotonicLowerBound(src, &dest);
        dest = dest.To(cpu);
        std::vector<T> data(dest.Data(), dest.Data() + dest.Dim());
        src = src.To(cpu);
        int32_t *src_data = src.Data();
        S min_value = std::numeric_limits<S>::max();
        std::vector<T> expected_data(src_dim);
        for (int32_t i = src_dim - 1; i >= 0; --i) {
          min_value = std::min(src_data[i], min_value);
          expected_data[i] = min_value;
        }
        EXPECT_EQ(data, expected_data);
      }
    }
  }
}

TEST(OpsTest, MonotonicLowerBoundTest) {
  TestMonotonicLowerBound<int32_t, int32_t>();
  TestMonotonicLowerBound<int32_t, double>();
}

template <typename S, typename T>
void TestMonotonicDecreasingUpperBound() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      std::vector<S> values;
      Array1<S> src(context, values);
      Array1<T> dest(context, 0);
      MonotonicDecreasingUpperBound(src, &dest);
      EXPECT_EQ(dest.Dim(), 0);
    }

    {
      // simple case
      std::vector<S> values = {10, 7, 3, 5, 4, 1, 0, 2};
      std::vector<T> expected_data = {10, 7, 5, 5, 4, 2, 2, 2};
      ASSERT_EQ(values.size(), expected_data.size());
      Array1<S> src(context, values);
      Array1<T> dest(context, static_cast<int32_t>(values.size()));
      MonotonicDecreasingUpperBound(src, &dest);
      dest = dest.To(cpu);
      std::vector<T> data(dest.Data(), dest.Data() + dest.Dim());
      EXPECT_EQ(data, expected_data);
    }

    {
      // simple case with dest = &src
      std::vector<S> values = {10, 7, 3, 5, 4, 1, 0, 2};
      std::vector<T> expected_data = {10, 7, 5, 5, 4, 2, 2, 2};
      ASSERT_EQ(values.size(), expected_data.size());
      Array1<S> src(context, values);
      MonotonicDecreasingUpperBound(src, &src);
      src = src.To(cpu);
      std::vector<T> data(src.Data(), src.Data() + src.Dim());
      EXPECT_EQ(data, expected_data);
    }

    {
      // random large case
      for (int32_t i = 0; i != 2; ++i) {
        int32_t n = RandInt(1, 10000);
        int32_t src_dim = RandInt(0, 10000);
        Array1<S> src = RandUniformArray1(context, src_dim, 0, n - 1);
        Array1<T> dest(context, src_dim);
        MonotonicDecreasingUpperBound(src, &dest);
        dest = dest.To(cpu);
        std::vector<T> data(dest.Data(), dest.Data() + dest.Dim());
        src = src.To(cpu);
        int32_t *src_data = src.Data();
        S max_value = std::numeric_limits<S>::min();
        std::vector<T> expected_data(src_dim);
        for (int32_t i = src_dim - 1; i >= 0; --i) {
          max_value = std::max(src_data[i], max_value);
          expected_data[i] = max_value;
        }
        EXPECT_EQ(data, expected_data);
      }
    }
  }
}

TEST(OpsTest, MonotonicDecreasingUpperBoundTest) {
  TestMonotonicDecreasingUpperBound<int32_t, int32_t>();
  TestMonotonicDecreasingUpperBound<int32_t, double>();
}

TEST(OpsTest, InvertMonotonicDecreasingTest) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // empty case
      std::vector<int32_t> values;
      Array1<int32_t> src(context, values);
      Array1<int32_t> dest = InvertMonotonicDecreasing(src);
      EXPECT_EQ(dest.Dim(), 0);
    }

    {
      // simple case
      std::vector<int32_t> values = {6, 4, 4, 2};
      Array1<int32_t> src(context, values);
      Array1<int32_t> dest = InvertMonotonicDecreasing(src);
      EXPECT_EQ(dest.Dim(), 6);
      dest = dest.To(cpu);
      std::vector<int32_t> data(dest.Data(), dest.Data() + dest.Dim());
      std::vector<int32_t> expected_data = {4, 4, 3, 3, 1, 1};
      EXPECT_EQ(data, expected_data);

      // convert back
      dest = dest.To(context);
      Array1<int32_t> src1 = InvertMonotonicDecreasing(dest);
      EXPECT_TRUE(Equal(src1, src));
    }

    {
      // random large case
      for (int32_t i = 0; i != 2; ++i) {
        int32_t n = RandInt(1, 1000);
        int32_t src_dim = RandInt(0, 1000);
        Array1<int32_t> src = RandUniformArray1(context, src_dim, 1, n);
        Sort<int32_t, GreaterThan<int32_t>>(&src);
        Array1<int32_t> dest = InvertMonotonicDecreasing(src);
        // convert back
        Array1<int32_t> src1 = InvertMonotonicDecreasing(dest);
        EXPECT_TRUE(Equal(src1, src));
      }
    }
  }
}

template <typename T>
void ArrayPlusTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i != 2; ++i) {
      {
        // normal case
        int32_t dim = RandInt(0, 1000);
        Array1<T> src1 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> src2 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> dest(context, dim);
        Plus(src1, src2, &dest);
        Array1<T> ans = Plus(src1, src2);
        EXPECT_EQ(ans.Dim(), dim);

        src1.To(cpu);
        src2.To(cpu);
        Array1<T> expected(cpu, dim);
        T *expected_data = expected.Data();
        for (int32_t n = 0; n != dim; ++n) {
          expected_data[n] = src1[n] + src2[n];
        }
        CheckArrayData(dest, expected);
        CheckArrayData(ans, expected);
      }
      {
        // special case: &src1 == &src2 == dest
        int32_t dim = RandInt(0, 1000);
        Array1<T> src = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> src_copy = src.Clone();
        Plus(src, src, &src);
        src_copy.To(cpu);
        Array1<T> expected(cpu, dim);
        T *expected_data = expected.Data();
        for (int32_t n = 0; n != dim; ++n) {
          expected_data[n] = src_copy[n] + src_copy[n];
        }
        CheckArrayData(src, expected);
      }
    }
  }
}

TEST(OpsTest, PlusTest) {
  ArrayPlusTest<int32_t>();
  ArrayPlusTest<float>();
}

template <typename T>
void ArrayMinusTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i != 2; ++i) {
      {
        // normal case
        int32_t dim = RandInt(0, 1000);
        Array1<T> src1 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> src2 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> dest(context, dim);
        Minus(src1, src2, &dest);
        Array1<T> ans = Minus(src1, src2);
        EXPECT_EQ(ans.Dim(), dim);

        src1.To(cpu);
        src2.To(cpu);
        Array1<T> expected(cpu, dim);
        T *expected_data = expected.Data();
        for (int32_t n = 0; n != dim; ++n) {
          expected_data[n] = src1[n] - src2[n];
        }
        CheckArrayData(dest, expected);
        CheckArrayData(ans, expected);
      }
      {
        // special case: &src1 == &src2 == dest
        int32_t dim = RandInt(0, 1000);
        Array1<T> src = RandUniformArray1<T>(context, dim, 0, 1000);
        Minus(src, src, &src);
        Array1<T> expected(context, dim, T(0));
        CheckArrayData(src, expected);
      }
    }
  }
}

TEST(OpsTest, MinusTest) {
  ArrayMinusTest<int32_t>();
  ArrayMinusTest<float>();
}

template <typename T>
void ArrayTimesTest() {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i != 2; ++i) {
      {
        // normal case
        int32_t dim = RandInt(0, 1000);
        Array1<T> src1 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> src2 = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> dest(context, dim);
        Times(src1, src2, &dest);
        Array1<T> ans = Times(src1, src2);
        EXPECT_EQ(ans.Dim(), dim);

        src1.To(cpu);
        src2.To(cpu);
        Array1<T> expected(cpu, dim);
        T *expected_data = expected.Data();
        for (int32_t n = 0; n != dim; ++n) {
          expected_data[n] = src1[n] * src2[n];
        }
        CheckArrayData(dest, expected);
        CheckArrayData(ans, expected);
      }
      {
        // special case: &src1 == &src2 == dest
        int32_t dim = RandInt(0, 1000);
        Array1<T> src = RandUniformArray1<T>(context, dim, 0, 1000);
        Array1<T> src_copy = src.Clone();
        Times(src, src, &src);
        src_copy.To(cpu);
        Array1<T> expected(cpu, dim);
        T *expected_data = expected.Data();
        for (int32_t n = 0; n != dim; ++n) {
          expected_data[n] = src_copy[n] * src_copy[n];
        }
        CheckArrayData(src, expected);
      }
    }
  }
}

TEST(OpsTest, TimesTest) {
  ArrayTimesTest<int32_t>();
  ArrayTimesTest<float>();
}

TEST(OpsTest, Array1IndexTest) {
  for (int loop = 0; loop < 2; loop++) {
    ContextPtr c = (loop == 0 ? GetCpuContext() : GetCudaContext()),
               cpu_context = GetCpuContext();

    int32_t src_dim = RandInt(1, 10), ans_dim = RandInt(1, 10);

    using T = int64_t;
    Array1<T> src = RandUniformArray1<T>(c, src_dim, 0, 100);

    Array1<int32_t> indexes_no_minus_one =
                        RandUniformArray1<int32_t>(c, ans_dim, 0, src_dim - 1),
                    indexes_minus_one =
                        RandUniformArray1<int32_t>(c, ans_dim, -1, src_dim - 1);
    Array1<T> ans_no_minus_one = Index(src, indexes_no_minus_one, false),
              ans_no_minus_one_check = src[indexes_no_minus_one],
              ans_no_minus_one_check2 = Index(src, indexes_no_minus_one, true);
    ASSERT_TRUE(Equal(ans_no_minus_one, ans_no_minus_one_check));
    ASSERT_TRUE(Equal(ans_no_minus_one, ans_no_minus_one_check2));

    Array1<T> ans_minus_one = Index(src, indexes_minus_one, true);

    ans_minus_one = ans_minus_one.To(cpu_context);
    src = src.To(cpu_context);
    indexes_minus_one = indexes_minus_one.To(cpu_context);
    for (int32_t i = 0; i < indexes_minus_one.Dim(); i++) {
      int32_t index = indexes_minus_one[i];
      ASSERT_EQ(ans_minus_one[i], (index < 0 ? 0 : src[index]));
    }
  }
}

TEST(OpsTest, InvertPermutationTest) {
  for (int loop = 0; loop < 2; loop++) {
    ContextPtr c = (loop == 0 ? GetCpuContext() : GetCudaContext()),
               cpu_context = GetCpuContext();
    for (int i = 0; i < 10; i++) {
      int32_t len = RandInt(0, 10);
      std::vector<int32_t> permutation(len);
      std::iota(permutation.begin(), permutation.end(), 0);
      std::random_shuffle(permutation.begin(), permutation.end());
      Array1<int32_t> permutation_array(c, permutation);
      Array1<int32_t> permutation_array_inv =
          InvertPermutation(permutation_array);
      Array1<int32_t> range = permutation_array[permutation_array_inv],
                      range2 = Range(c, len, 0);
      K2_CHECK(Equal(range, range2));
    }
  }
}

TEST(OpsTest, Array2IndexTest) {
  for (int loop = 0; loop < 2; loop++) {
    ContextPtr c = (loop == 0 ? GetCpuContext() : GetCudaContext()),
               cpu_context = GetCpuContext();

    int32_t src_dim0 = RandInt(1, 10), src_dim1 = RandInt(1, 10),
            ans_dim0 = RandInt(1, 10);

    using T = int64_t;
    Array2<T> src = RandUniformArray2<T>(c, src_dim0, src_dim1, 0, 100);

    Array1<int32_t> indexes_no_minus_one = RandUniformArray1<int32_t>(
                        c, ans_dim0, 0, src_dim0 - 1),
                    indexes_minus_one = RandUniformArray1<int32_t>(
                        c, ans_dim0, -1, src_dim0 - 1);

    Array2<T> ans_no_minus_one = IndexRows(src, indexes_no_minus_one, false),
              ans_no_minus_one_check =
                  IndexRows(src, indexes_no_minus_one, true);
    ASSERT_TRUE(Equal(ans_no_minus_one, ans_no_minus_one_check));

    Array2<T> ans_minus_one = IndexRows(src, indexes_minus_one, true);

    ans_minus_one = ans_minus_one.To(cpu_context);
    src = src.To(cpu_context);
    indexes_minus_one = indexes_minus_one.To(cpu_context);

    auto src_acc = src.Accessor(), ans_minus_one_acc = ans_minus_one.Accessor();
    K2_LOG(INFO) << "src = " << src << ", indexes = " << indexes_minus_one
                 << ", ans = " << ans_minus_one;
    for (int32_t i = 0; i < ans_dim0; i++) {
      int32_t index = indexes_minus_one[i];
      for (int32_t j = 0; j < src_dim1; j++) {
        ASSERT_EQ(ans_minus_one_acc(i, j), (index < 0 ? 0 : src_acc(index, j)));
      }
    }
  }
}

template <typename T>
static void Array1SortTestSimple() {
  std::vector<T> data = {3, 2, 5, 1};
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // with index map
      Array1<T> array(context, data);
      Array1<int32_t> index_map;
      Sort(&array, &index_map);
      CheckArrayData(array, std::vector<T>{1, 2, 3, 5});
      CheckArrayData(index_map, std::vector<int32_t>{3, 1, 0, 2});
    }

    {
      // without index map
      Array1<T> array(context, data);
      Sort(&array);
      CheckArrayData(array, std::vector<T>{1, 2, 3, 5});
    }
  }
}

template <typename T>
static void Array1SortTestEmpty() {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<T> array(context, 0);
    Array1<int32_t> index_map;
    Sort(&array, &index_map);
    EXPECT_EQ(array.Dim(), 0);
    EXPECT_EQ(index_map.Dim(), 0);
  }
}

template <typename T>
static void Array1SortTestRandom() {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    int32_t dim = RandInt(0, 10000);
    int32_t min_value = RandInt(-1000, 1000);
    int32_t max_value = min_value + RandInt(0, 3000);
    {
      // with index map
      Array1<T> array =
          RandUniformArray1<T>(context, dim, min_value, max_value);
      Array1<T> data = array.Clone();

      Array1<int32_t> index_map;
      Sort(&array, &index_map);
      array = array.To(GetCpuContext());
      EXPECT_TRUE(std::is_sorted(array.Data(), array.Data() + array.Dim()));

      index_map = index_map.To(GetCpuContext());
      for (int32_t i = 0; i != array.Dim(); ++i)
        EXPECT_EQ(array[i], data[index_map[i]]);
    }

    {
      // without index_map
      Array1<T> array =
          RandUniformArray1<T>(context, dim, min_value, max_value);
      Sort(&array);
      array = array.To(GetCpuContext());
      EXPECT_TRUE(std::is_sorted(array.Data(), array.Data() + array.Dim()));
    }
  }
}

TEST(OpsTest, Array1Sort) {
  Array1SortTestSimple<int32_t>();
  Array1SortTestSimple<float>();

  Array1SortTestEmpty<int32_t>();
  Array1SortTestEmpty<float>();

  Array1SortTestRandom<int32_t>();
  Array1SortTestRandom<float>();
}

TEST(OpsTest, Array2Assign) {
  for (int loop = 0; loop < 10; loop++) {
    ContextPtr c = ((loop % 2) == 0 ? GetCpuContext() : GetCudaContext());

    int32_t src_dim0 = RandInt(1, 10), src_dim1 = RandInt(1, 10);

    using T = int64_t;
    Array2<T> src = RandUniformArray2<T>(c, src_dim0, src_dim1, 0, 100);

    Array2<T> dest = RandUniformArray2<T>(c, src_dim0, src_dim1, 0, 100);

    Assign(src, &dest);

    K2_CHECK(Equal(src, dest));

    ContextPtr c_other = ((loop % 2) != 0 ? GetCpuContext() : GetCudaContext());
    Array2<T> dest2 = RandUniformArray2<T>(c_other, src_dim0, src_dim1, 0, 100);

    if (src.ElemStride0() == src_dim1 && dest2.ElemStride0() == src_dim1) {
      // test cross-device copy, which is only supported for contiguous input
      Assign(src, &dest2);
      K2_CHECK(Equal(src.To(c_other), dest2));
    }
  }
}

TEST(OpsTest, SizesToMergeMapTest) {
  for (int loop = 0; loop < 2; loop++) {
    ContextPtr c = ((loop % 2) == 0 ? GetCpuContext() : GetCudaContext());
    std::vector<int32_t> sizes = {3, 5, 1};
    Array1<uint32_t> merge_map = SizesToMergeMap(c, sizes);
    std::vector<uint32_t> expected_map = {0, 3, 6, 1, 4, 7, 10, 13, 2};
    K2_LOG(INFO) << "merge_map is " << merge_map;
    CheckArrayData(merge_map, expected_map);
  }
}

}  // namespace k2
