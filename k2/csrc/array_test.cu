/**
 * @brief
 * array_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/tensor.h"

namespace k2 {

template <typename T>
void CheckArrayEqual(const Array1<T> &a, const Array1<T> &b) {
  ASSERT_TRUE(a.Context()->IsCompatible(*b.Context()));
  ASSERT_EQ(a.Dim(), b.Dim());

  const T *da = a.Data();
  const T *db = b.Data();
  auto n = a.Dim();
  auto compare = [=] __host__ __device__(int32_t i) -> void {
    K2_CHECK_EQ(da[i], db[i]);
  };
  Eval(a.Context(), n, compare);
}

template <typename T, DeviceType d>
void TestArray1() {
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
               array.Dim() * array.ElementSize(), kind,
               nullptr);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(array[i], i);
    }
  }

  {
    // test operator=(T t) and "operator[](int32_t i) const"
    Array1<T> array(context, 5);
    ASSERT_EQ(array.Dim(), 5);
    // operator=(T t)
    array = 2;
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(array[i], 2);
    }
  }

  {
    // test Back()"
    std::vector<T> data(5);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    EXPECT_EQ(array.Back(), 4);
  }

  {
    // created with Array1(ContextPtr, int32_t size, T elem)
    Array1<T> array(context, 5, T(2));
    ASSERT_EQ(array.Dim(), 5);
    // copy data from CPU/GPU to CPU
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(array[i], 2);
    }
  }

  {
    // created with Array1(ContextPtr, int32_t size, Callable &&callable)
    auto lambda_set_values = [] __host__ __device__(int32_t i) -> T {
      return i * i;
    };
    Array1<T> array(context, 5, lambda_set_values);
    ASSERT_EQ(array.Dim(), 5);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(array[i], i * i);
    }
  }

  {
    // created with Array(ContextPtr, const std:vector<T>&)
    std::vector<T> data(5);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    ASSERT_EQ(array.Dim(), 5);
    for (int32_t i = 0; i < array.Dim(); ++i) {
      EXPECT_EQ(array[i], data[i]);
    }
  }

  {
    // test Range(start, size)
    std::vector<T> data(10);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    int32_t start = 2;
    int32_t size = 6;
    Array1<T> sub_array = array.Range(start, size);
    ASSERT_EQ(sub_array.Dim(), size);
    for (int32_t i = 0; i < sub_array.Dim(); ++i) {
      EXPECT_EQ(sub_array[i], data[i + start]);
    }
  }

  {
    // test Range(start, size, inc)
    std::vector<T> data(20);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    int32_t start = 3;
    int32_t size = 8;
    int32_t inc = 2;
    Tensor sub_tensor = array.Range(start, size, inc);
    Dtype type = DtypeOf<T>::dtype;
    EXPECT_EQ(sub_tensor.GetDtype(), type);
    Shape shape = sub_tensor.GetShape();
    EXPECT_EQ(shape.NumAxes(), 1);
    EXPECT_EQ(shape.Nelement(), size);
    EXPECT_EQ(shape.StorageSize(), (size - 1) * inc + 1);
    EXPECT_EQ(shape.Dim(0), size);
    EXPECT_EQ(shape.Stride(0), inc);
    // copy data from CPU/GPU to CPU
    const T *sub_tensor_data = sub_tensor.Data<T>();
    auto kind = GetMemoryCopyKind(*(sub_tensor.GetRegion()->context), *cpu);
    std::vector<T> cpu_data(shape.StorageSize());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<const void *>(sub_tensor_data),
               shape.StorageSize() * TraitsOf(sub_tensor.GetDtype()).NumBytes(),
               kind, nullptr);
    int32_t dim0 = shape.Dim(0);
    int32_t stride0 = shape.Stride(0);
    for (int32_t i = 0, j = start; i < dim0; ++i, j += inc) {
      EXPECT_EQ(cpu_data[i * stride0], data[j]);
    }

    Array1<T> arr(sub_tensor);
    ASSERT_EQ(arr.Dim(), dim0);
    for (int32_t i = 0, j = start; i < dim0; ++i, j += inc) {
      EXPECT_EQ(arr[i], data[j]);
    }
  }

  {
    // test ToTensor
    int32_t size = 20;
    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array(context, data);
    Tensor tensor = array.ToTensor();
    Dtype type = DtypeOf<T>::dtype;
    EXPECT_EQ(tensor.GetDtype(), type);
    Shape shape = tensor.GetShape();
    EXPECT_EQ(shape.NumAxes(), 1);
    EXPECT_EQ(shape.Nelement(), size);
    EXPECT_EQ(shape.StorageSize(), size);
    EXPECT_EQ(shape.Dim(0), size);
    EXPECT_EQ(shape.Stride(0), 1);
    // copy data from CPU/GPU to CPU
    const T *tensor_data = tensor.Data<T>();
    auto kind = GetMemoryCopyKind(*(tensor.GetRegion()->context), *cpu);
    std::vector<T> cpu_data(shape.StorageSize());
    MemoryCopy(static_cast<void *>(cpu_data.data()),
               static_cast<const void *>(tensor_data),
               shape.StorageSize() * TraitsOf(tensor.GetDtype()).NumBytes(),
               kind, nullptr);
    int32_t dim0 = shape.Dim(0);
    int32_t stride0 = shape.Stride(0);
    for (int32_t i = 0, j = 0; i < dim0; ++i, ++j) {
      EXPECT_EQ(cpu_data[i * stride0], data[j]);
    }
  }

  {
    // test To(context)
    std::vector<T> data = {0, 1, 2, 3};
    Array1<T> cpu_array(cpu, data);
    Array1<T> cuda_array(GetCudaContext(), data);

    Array1<T> src(context, data);
    auto cpu_dst = src.To(cpu);
    auto cuda_dst = src.To(GetCudaContext());

    CheckArrayEqual(cpu_array, cpu_dst);
    CheckArrayEqual(cuda_array, cuda_dst);
  }

  {
    // test operator <<
    std::vector<T> data = {0, 1, 2, 3};
    Array1<T> src(context, data);
    K2_LOG(INFO) << src;
  }

  {
    // test operator[](const Array1<int32_t> &indexes) const
    std::vector<T> data(20);
    std::iota(data.begin(), data.end(), 1);
    Array1<T> array(context, data);
    std::vector<int32_t> indexes = {0, 1, 2, 5, 1, 6, 8, 9, 2, 4, 6, 3};
    Array1<int32_t> indexes_array(context, indexes);
    std::vector<T> expected_data = {1, 2, 3, 6, 2, 7, 9, 10, 3, 5, 7, 4};
    Array1<T> ans_array = array[indexes_array];
    ASSERT_EQ(ans_array.Dim(), expected_data.size());
    for (int32_t i = 0; i < ans_array.Dim(); ++i) {
      EXPECT_EQ(ans_array[i], expected_data[i]);
    }
  }

  {
    // test Resize
    std::vector<T> data(5);
    std::iota(data.begin(), data.end(), 1);
    Array1<T> array(context, data);
    EXPECT_EQ(array.Dim(), data.size());

    // new_size <= array.Dim()
    int32_t new_size = 3;
    array.Resize(new_size);
    EXPECT_EQ(array.Dim(), new_size);

    // re-initialize...
    array = Array1<T>(context, data);
    // new_size > array.Dim()
    new_size = 7;
    array.Resize(new_size);
    EXPECT_EQ(array.Dim(), new_size);
    // new_size > array.Dim()
    new_size = 8;
    array.Resize(new_size);
    EXPECT_EQ(array.Dim(), new_size);
    // copy data from CPU/GPU to CPU
    const T *array_data = array.Data();
    // data.size() == 5, array.Dim() == 8, there are 3 uninitialized elements.
    for (int32_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(array[i], data[i]);
    }
  }
}

template <typename T, DeviceType d>
void TestArray2() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // test To(context)
    // case 1: contiguous
    //
    // 0 1 2
    // 3 4 5
    //
    constexpr auto kDim0 = 2;
    constexpr auto kDim1 = 3;
    std::vector<T> data = {0, 1, 2, 3, 4, 5};
    ASSERT_EQ(static_cast<int32_t>(data.size()), kDim0 * kDim1);

    Array1<T> arr1(context, data);
    Array2<T> array(arr1, kDim0, kDim1);

    auto cpu_array = array.To(cpu);
    auto cuda_array = array.To(GetCudaContext());

    ASSERT_EQ(cpu_array.ElemStride0(), cpu_array.Dim1());
    ASSERT_EQ(cuda_array.ElemStride0(), cuda_array.Dim1());

    auto k = 0;
    for (auto r = 0; r != kDim0; ++r)
      for (auto c = 0; c != kDim1; ++c) {
        // WARNING: it's inefficient to access elements of Array2
        // with operator [][]
        EXPECT_EQ(cpu_array[r][c], k);
        EXPECT_EQ(cuda_array[r][c], k);
        ++k;
      }

    // test operator <<
    K2_LOG(INFO) << array;
  }

  {
    // test To(context)
    // case 2: non-contiguous
    //
    // 0 1 2 x x
    // 3 4 5 x x
    //
    constexpr auto kDim0 = 2;
    constexpr auto kDim1 = 3;
    constexpr auto kElemStride0 = 5;
    std::vector<T> data = {0, 1, 2, -1, -1, 3, 4, 5, -1, -1};
    EXPECT_EQ(static_cast<int32_t>(data.size()), kDim0 * kElemStride0);

    auto region = NewRegion(context, data.size() * sizeof(T));

    auto dst = region->template GetData<T>();
    auto kind = GetMemoryCopyKind(*cpu, *context);
    MemoryCopy(static_cast<void *>(dst), static_cast<const void *>(data.data()),
               data.size() * sizeof(T), kind,
               region->context.get());

    Array2<T> array(kDim0, kDim1, kElemStride0, 0, region);

    auto cpu_array = array.To(cpu);
    auto cuda_array = array.To(GetCudaContext());

    auto k = 0;
    for (auto r = 0; r != kDim0; ++r)
      for (auto c = 0; c != kDim1; ++c) {
        // WARNING: it's inefficient to access elements of Array2
        // with operator [][]
        EXPECT_EQ(cpu_array[r][c], k);
        EXPECT_EQ(cuda_array[r][c], k);
        ++k;
      }

    // test operator <<
    K2_LOG(INFO) << array;
  }

  {
    // created with Array2(Array1, dim0, dim1), contiguous
    std::vector<T> data(20);
    std::iota(data.begin(), data.end(), 0);
    Array1<T> array1(context, data);
    Array2<T> array(array1, 4, 5);
    EXPECT_EQ(array.Dim0(), 4);
    EXPECT_EQ(array.Dim1(), 5);
    EXPECT_EQ(array.ElemStride0(), 5);
    T *array_data = array.Data();

    {
      // test Data()
      const T *array_data = array.Data();
      // copy data from CPU/GPU to CPU
      int32_t elem_stride0 = array.ElemStride0();
      int32_t num_element_copy = array.Dim0() * array.ElemStride0();
      std::vector<T> cpu_data(num_element_copy);
      auto kind = GetMemoryCopyKind(*array.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 num_element_copy * array.ElementSize(), kind,
                 nullptr);
      for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(cpu_data[i * elem_stride0 + j], data[n++]);
        }
      }
    }

    {
      // test operator[]
      for (int32_t i = 0; i < array.Dim0(); ++i) {
        Array1<T> sub_array = array[i];
        const T *sub_array_data = sub_array.Data();
        ASSERT_EQ(sub_array.Dim(), array.Dim1());
        auto kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
        std::vector<T> sub_array_cpu_data(sub_array.Dim());
        MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                   static_cast<const void *>(sub_array_data),
                   sub_array.Dim() * sub_array.ElementSize(), kind,
                   nullptr);
        for (int32_t j = 0; j < sub_array.Dim(); ++j) {
          EXPECT_EQ(sub_array_cpu_data[j], data[i * array.ElemStride0() + j]);
        }
      }
    }

    {
      // test Flatten()
      Array1<T> sub_array = array.Flatten();
      const T *sub_array_data = sub_array.Data();
      ASSERT_EQ(sub_array.Dim(), array.Dim0() * array.Dim1());
      auto kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
      std::vector<T> sub_array_cpu_data(sub_array.Dim());
      MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                 static_cast<const void *>(sub_array_data),
                 sub_array.Dim() * sub_array.ElementSize(), kind,
                 nullptr);
      for (int32_t i = 0; i < sub_array.Dim(); ++i) {
        EXPECT_EQ(sub_array_cpu_data[i], data[i]);
      }
    }

    {
      // test ToTensor()
      Tensor tensor = array.ToTensor();
      Dtype array_dtype = DtypeOf<T>::dtype;
      EXPECT_EQ(tensor.GetDtype(), array_dtype);
      Shape shape = tensor.GetShape();
      EXPECT_EQ(shape.NumAxes(), 2);
      EXPECT_EQ(shape.Nelement(), array.Dim0() * array.Dim1());
      EXPECT_EQ(shape.Dim(0), array.Dim0());
      EXPECT_EQ(shape.Dim(1), array.Dim1());
      EXPECT_EQ(shape.Stride(0), array.ElemStride0());
      EXPECT_EQ(shape.Stride(1), 1);
      const T *tensor_data = tensor.Data<T>();
      auto kind = GetMemoryCopyKind(*tensor.GetRegion()->context, *cpu);
      std::vector<T> cpu_tensor_data(shape.StorageSize());
      MemoryCopy(static_cast<void *>(cpu_tensor_data.data()),
                 static_cast<const void *>(tensor_data),
                 shape.StorageSize() * TraitsOf(tensor.GetDtype()).NumBytes(),
                 kind, nullptr);
      for (int32_t m = 0; m < shape.Dim(0); ++m) {
        for (int32_t n = 0; n < shape.Dim(1); ++n) {
          int32_t value =
              cpu_tensor_data[m * shape.Stride(0) + n * shape.Stride(1)];
          EXPECT_EQ(value, data[m * array.ElemStride0() + n]);
        }
      }
    }

    {
      // test constAccessor
      const auto &const_array = array;
      ConstArray2Accessor<T> accessor = const_array.Accessor();
      if (array.Context()->GetDeviceType() == kCpu) {
        EXPECT_EQ(accessor(0, 0), data[0 * array.ElemStride0() + 0]);
        EXPECT_EQ(accessor(2, 3), data[2 * array.ElemStride0() + 3]);
      }
    }

    {
      Array2Accessor<T> accessor = array.Accessor();
      if (array.Context()->GetDeviceType() == kCpu) {
        EXPECT_EQ(accessor(0, 0), 0);
        accessor(0, 0) = -10;
        EXPECT_EQ(accessor(0, 0), -10);
        EXPECT_EQ(array.Data()[0], -10);
      }
    }
  }

  {
    // created with region
    const int32_t element_size = TraitsOf(DtypeOf<T>::dtype).NumBytes();
    const int32_t num_element = 20;
    RegionPtr region = NewRegion(context, num_element * element_size);
    std::vector<T> src_data(num_element);
    std::iota(src_data.begin(), src_data.end(), 0);
    T *data = region->GetData<T, d>();
    auto kind = GetMemoryCopyKind(*cpu, *region->context);
    MemoryCopy(static_cast<void *>(data),
               static_cast<const void *>(src_data.data()),
               num_element * element_size, kind,
               nullptr);

    {
      // created with region, contiguous on 0 aixs
      Array2<T> array(4, 5, 5, 0, region);
      EXPECT_EQ(array.Dim0(), 4);
      EXPECT_EQ(array.Dim1(), 5);
      EXPECT_EQ(array.ElemStride0(), 5);
      EXPECT_EQ(array.ElementSize(), element_size);
      // test Data()
      const T *array_data = array.Data();
      // copy data from CPU/GPU to CPU
      int32_t elem_stride0 = array.ElemStride0();
      int32_t num_element_copy = array.Dim0() * array.ElemStride0();
      std::vector<T> cpu_data(num_element_copy);
      kind = GetMemoryCopyKind(*array.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 num_element_copy * array.ElementSize(), kind, nullptr);
      for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(cpu_data[i * elem_stride0 + j], src_data[n++]);
        }
      }

      {
        // test Flatten()
        Array1<T> sub_array = array.Flatten();
        const T *sub_array_data = sub_array.Data();
        ASSERT_EQ(sub_array.Dim(), array.Dim0() * array.Dim1());
        kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
        std::vector<T> sub_array_cpu_data(sub_array.Dim());
        MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                   static_cast<const void *>(sub_array_data),
                   sub_array.Dim() * sub_array.ElementSize(), kind, nullptr);
        for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
          for (int32_t j = 0; j < array.Dim1(); ++j) {
            EXPECT_EQ(sub_array_cpu_data[n++], src_data[i * elem_stride0 + j]);
          }
        }
      }
    }

    {
      // created with region, non-contiguous on 0 aixs
      int32_t element_offset = 2;
      Array2<T> array(3, 5, 6, element_offset * element_size, region);
      EXPECT_EQ(array.Dim0(), 3);
      EXPECT_EQ(array.Dim1(), 5);
      EXPECT_EQ(array.ElemStride0(), 6);
      EXPECT_EQ(array.ElementSize(), element_size);
      // test Data()
      const T *array_data = array.Data();
      // copy data from CPU/GPU to CPU
      int32_t elem_stride0 = array.ElemStride0();
      int32_t num_element_copy = array.Dim0() * array.ElemStride0();
      std::vector<T> cpu_data(num_element_copy);
      kind = GetMemoryCopyKind(*array.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 num_element_copy * array.ElementSize(), kind, nullptr);
      for (int32_t i = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(cpu_data[i * elem_stride0 + j],
                    src_data[element_offset + i * elem_stride0 + j]);
        }
      }

      {
        // test Flatten()
        Array1<T> sub_array = array.Flatten();
        const T *sub_array_data = sub_array.Data();
        ASSERT_EQ(sub_array.Dim(), array.Dim0() * array.Dim1());
        kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
        std::vector<T> sub_array_cpu_data(sub_array.Dim());
        MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                   static_cast<const void *>(sub_array_data),
                   sub_array.Dim() * sub_array.ElementSize(), kind, nullptr);
        for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
          for (int32_t j = 0; j < array.Dim1(); ++j) {
            EXPECT_EQ(sub_array_cpu_data[n++],
                      src_data[element_offset + i * elem_stride0 + j]);
          }
        }
      }
    }
  }

  {
    // created with tensor, stride on 1st axis is 1
    const int32_t element_size = TraitsOf(DtypeOf<T>::dtype).NumBytes();
    const int32_t num_element = 24;
    RegionPtr region = NewRegion(context, num_element * element_size);
    std::vector<T> src_data(num_element);
    std::iota(src_data.begin(), src_data.end(), 0);
    T *data = region->GetData<T, d>();
    auto kind = GetMemoryCopyKind(*cpu, *region->context);
    MemoryCopy(static_cast<void *>(data),
               static_cast<const void *>(src_data.data()),
               num_element * element_size, kind, nullptr);
    std::vector<int32_t> dims = {2, 4};
    std::vector<int32_t> strides = {10, 1};
    Shape shape(dims, strides);
    const int32_t element_offset = 4;
    const int32_t bytes_offset = element_offset * element_size;
    Tensor tensor(DtypeOf<T>::dtype, shape, region, bytes_offset);
    Array2<T> array(tensor, false);
    int32_t elem_stride0 = array.ElemStride0();
    int32_t elem_stride1 = tensor.GetShape().Stride(1);
    EXPECT_EQ(elem_stride1, 1);
    {
      // check_data
      const T *array_data = array.Data();
      int32_t num_element_copy = array.Dim0() * array.ElemStride0();
      std::vector<T> cpu_data(num_element_copy);
      kind = GetMemoryCopyKind(*array.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 num_element_copy * array.ElementSize(), kind, nullptr);
      for (int32_t i = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(
              cpu_data[i * elem_stride0 + j],
              src_data[element_offset + i * elem_stride0 + j * elem_stride1]);
        }
      }
    }

    {
      // test Flatten()
      Array1<T> sub_array = array.Flatten();
      const T *sub_array_data = sub_array.Data();
      ASSERT_EQ(sub_array.Dim(), array.Dim0() * array.Dim1());
      kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
      std::vector<T> sub_array_cpu_data(sub_array.Dim());
      MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                 static_cast<const void *>(sub_array_data),
                 sub_array.Dim() * sub_array.ElementSize(), kind, nullptr);
      for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(
              sub_array_cpu_data[n++],
              src_data[element_offset + i * elem_stride0 + j * elem_stride1]);
        }
      }
    }
  }

  {
    // created with tensor, stride on 1st axis is not 1
    const int32_t element_size = TraitsOf(DtypeOf<T>::dtype).NumBytes();
    const int32_t num_element = 24;
    RegionPtr region = NewRegion(context, num_element * element_size);
    std::vector<T> src_data(num_element);
    std::iota(src_data.begin(), src_data.end(), 0);
    T *data = region->GetData<T, d>();
    auto kind = GetMemoryCopyKind(*cpu, *region->context);
    MemoryCopy(static_cast<void *>(data),
               static_cast<const void *>(src_data.data()),
               num_element * element_size, kind, nullptr);
    std::vector<int32_t> dims = {2, 4};
    std::vector<int32_t> strides = {10, 2};
    Shape shape(dims, strides);
    const int32_t element_offset = 4;
    const int32_t bytes_offset = element_offset * element_size;
    Tensor tensor(DtypeOf<T>::dtype, shape, region, bytes_offset);
    Array2<T> array(tensor, true);
    int32_t elem_stride0 = array.ElemStride0();
    int32_t elem_stride1 = tensor.GetShape().Stride(1);
    {
      // check_data
      const T *array_data = array.Data();
      int32_t num_element_copy = array.Dim0() * array.ElemStride0();
      std::vector<T> cpu_data(num_element_copy);
      kind = GetMemoryCopyKind(*array.Context(), *cpu);
      MemoryCopy(static_cast<void *>(cpu_data.data()),
                 static_cast<const void *>(array_data),
                 num_element_copy * array.ElementSize(), kind, nullptr);
      for (int32_t i = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(
              cpu_data[i * elem_stride0 + j],
              src_data[element_offset + i * elem_stride0 + j * elem_stride1]);
        }
      }
    }

    {
      // test Flatten()
      Array1<T> sub_array = array.Flatten();
      const T *sub_array_data = sub_array.Data();
      ASSERT_EQ(sub_array.Dim(), array.Dim0() * array.Dim1());
      kind = GetMemoryCopyKind(*sub_array.Context(), *cpu);
      std::vector<T> sub_array_cpu_data(sub_array.Dim());
      MemoryCopy(static_cast<void *>(sub_array_cpu_data.data()),
                 static_cast<const void *>(sub_array_data),
                 sub_array.Dim() * sub_array.ElementSize(), kind, nullptr);
      for (int32_t i = 0, n = 0; i < array.Dim0(); ++i) {
        for (int32_t j = 0; j < array.Dim1(); ++j) {
          EXPECT_EQ(
              sub_array_cpu_data[n++],
              src_data[element_offset + i * elem_stride0 + j * elem_stride1]);
        }
      }
    }
  }
}

TEST(ArrayTest, Array1Test) {
  TestArray1<int32_t, kCpu>();
  TestArray1<int32_t, kCuda>();
  TestArray1<double, kCpu>();
  TestArray1<double, kCuda>();
}

TEST(ArrayTest, Array2Test) {
  TestArray2<int32_t, kCpu>();
  TestArray2<int32_t, kCuda>();
  TestArray2<double, kCpu>();
  TestArray2<double, kCuda>();
}

}  // namespace k2
