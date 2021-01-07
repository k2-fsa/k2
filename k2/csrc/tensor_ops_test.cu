/**
 * @brief Unittest for tensor_ops.
 *
 * @copyright
// Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

/* Return a 1-D tensor with random entries.

   @param [in] context  It specifies the device where the output tensor resides.
   @param [in] dim      Number of elements contained in the returned tensor.
   @param [in] stride   A positive number indicating the expected stride
                        of the output `tensor`.

   @return Returns a 1-D tensor with the given `dim` and `stride`.
 */
template <typename T>
static Tensor GenerateRandTensor1D(ContextPtr context, int32_t dim,
                                   int32_t stride) {
  K2_CHECK_GT(stride, 0);

  std::vector<T> data_vec(dim);
  for (T &d : data_vec) d = RandInt(-1000, 1000);

  Shape shape({dim}, {stride});

  Array1<T> array(context, data_vec);
  const T *array_data = array.Data();

  Tensor ans(context, DtypeOf<T>::dtype, shape);
  T *ans_data = ans.Data<T>();
  K2_EVAL(
      context, dim, lambda_set,
      (int32_t i)->void { ans_data[i * stride] = array_data[i]; });
  return ans;
}

/* Return a 2-D tensor filled with random values.

   @param [in] context   It specifies the device where the output tensor
                         resides.
   @param [in] num_rows  Number of rows in the returned tensor.
   @param [in] num_cols  Number of columns in the returned tensor.
   @param [in] stride    A positive number indicating the expected row stride
                         of the output `tensor`.

   @return Returns a 2-D tensor with the given `num_rows`, `num_cols` and
           `stride`.
 */
template <typename T>
static Tensor GenerateRandTensor2D(ContextPtr context, int32_t num_rows,
                                   int32_t num_cols, int32_t stride) {
  int32_t num_tensor_elements = num_rows * num_cols;
  K2_CHECK_GT(num_cols, 0);
  K2_CHECK_GE(stride, num_cols);
  K2_CHECK_GE(num_rows, 0);

  std::vector<T> data_vec(num_tensor_elements);
  for (T &d : data_vec) d = RandInt(-1000, 1000);

  Shape shape({num_rows, num_cols}, {stride, 1});
  Array1<T> array(context, data_vec);
  const T *array_data = array.Data();

  Tensor ans(context, DtypeOf<T>::dtype, shape);
  T *ans_data = ans.Data<T>();
  K2_EVAL2(
      context, num_rows, num_cols, lambda_set, (int32_t i, int32_t j)->void {
        ans_data[i * stride + j] = array_data[i * num_cols + j];
      });
  return ans;
}

template <typename T>
static void TestIndex1D() {
  bool allow_minus_one;
  int32_t stride;
  int32_t indexes_dim;
  int32_t numel;
  for (int32_t i = 0; i != 8; ++i) {
    stride = RandInt(1, 10);
    allow_minus_one = RandInt(-1000, 1000) & 1;
    indexes_dim = RandInt(1, 20000);
    numel = RandInt(1, 20000);

    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();
    Array1<int32_t> indexes =
        GenerateRandomIndexes(context, allow_minus_one, indexes_dim, numel - 1);

    Tensor src = GenerateRandTensor1D<T>(context, numel, stride);
    Tensor ans = Index(src, indexes, allow_minus_one);
    ASSERT_TRUE(ans.IsContiguous());
    ASSERT_EQ(ans.NumAxes(), 1);
    ASSERT_EQ(ans.Dim(0), indexes.Dim());

    ans = ans.To(GetCpuContext());
    indexes = indexes.To(ans.Context());
    src = src.To(ans.Context());
    ASSERT_TRUE(src.IsContiguous());

    const T *ans_data = ans.Data<T>();
    int32_t ans_dim = ans.Dim(0);
    const T *src_data = src.Data<T>();
    const int32_t *indexes_data = indexes.Data();
    for (int32_t i = 0; i != ans_dim; ++i) {
      int32_t index = indexes[i];
      if (index != -1)
        EXPECT_EQ(ans_data[i], src_data[index]);
      else
        EXPECT_EQ(ans_data[i], 0);
    }
  }
}

template <typename T>
static void TestIndex2D() {
  bool allow_minus_one;
  int32_t stride;
  int32_t num_rows;
  int32_t num_cols;
  int32_t indexes_dim;
  for (int32_t i = 0; i != 8; ++i) {
    num_rows = RandInt(1, 100);
    num_cols = RandInt(1, 100);
    stride = RandInt(0, 10) + num_cols;
    indexes_dim = RandInt(1, 10000);
    allow_minus_one = RandInt(-1000, 1000) & 1;

    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();
    Array1<int32_t> indexes = GenerateRandomIndexes(context, allow_minus_one,
                                                    indexes_dim, num_rows - 1);

    Tensor src = GenerateRandTensor2D<T>(context, num_rows, num_cols, stride);
    Tensor ans = Index(src, indexes, allow_minus_one);

    ASSERT_TRUE(ans.IsContiguous());
    ASSERT_EQ(ans.NumAxes(), 2);
    ASSERT_EQ(ans.Dim(0), indexes.Dim());
    ASSERT_EQ(ans.Dim(1), src.Dim(1));

    ans = ans.To(GetCpuContext());
    indexes = indexes.To(ans.Context());
    src = src.To(ans.Context());
    ASSERT_TRUE(src.IsContiguous());

    const T *ans_data = ans.Data<T>();
    int32_t ans_dim0 = ans.Dim(0);
    int32_t ans_dim1 = ans.Dim(1);
    const T *src_data = src.Data<T>();
    const int32_t *indexes_data = indexes.Data();
    for (int32_t i = 0; i != ans_dim0; ++i) {
      int32_t index = indexes[i];
      if (index != -1) {
        for (int32_t j = 0; j != ans_dim1; ++j) {
          EXPECT_EQ(ans_data[i * ans_dim1 + j], src_data[index * ans_dim1 + j]);
        }
      } else {
        for (int32_t j = 0; j != ans_dim1; ++j)
          EXPECT_EQ(ans_data[i * ans_dim1 + j], 0);
      }
    }
  }
}

TEST(Index, Index1D) {
  TestIndex1D<float>();
  TestIndex1D<int32_t>();
}

TEST(Index, Index2D) {
  TestIndex2D<float>();
  TestIndex2D<int32_t>();
}

template <typename T>
static void TestIndexAdd1D() {
  bool allow_minus_one;
  int32_t src_stride;
  int32_t dest_stride;
  int32_t src_dim;
  int32_t dest_dim;
  for (int32_t i = 0; i != 8; ++i) {
    src_stride = RandInt(1, 10);
    dest_stride = RandInt(1, 10);
    allow_minus_one = RandInt(-1000, 1000) & 1;
    src_dim = RandInt(1, 20000);
    dest_dim = RandInt(1, 20000);

    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();

    Array1<int32_t> indexes =
        GenerateRandomIndexes(context, allow_minus_one, src_dim, dest_dim - 1);

    Tensor src = GenerateRandTensor1D<T>(context, src_dim, src_stride);
    Tensor dest = GenerateRandTensor1D<T>(context, dest_dim, dest_stride);
    Tensor saved_dest = dest.Clone();

    IndexAdd(src, indexes, allow_minus_one, &dest);

    src = src.To(GetCpuContext());
    dest = dest.To(src.Context());
    indexes = indexes.To(dest.Context());
    saved_dest = saved_dest.To(src.Context());
    const T *src_data = src.Data<T>();
    const T *dest_data = dest.Data<T>();
    const int32_t *indexes_data = indexes.Data();
    T *saved_dest_data = saved_dest.Data<T>();
    for (int32_t i = 0; i != src_dim; ++i) {
      int32_t index = indexes_data[i];
      if (index == -1) continue;
      saved_dest_data[index] += src_data[i];
    }

    for (int32_t i = 0; i != dest_dim; ++i)
      EXPECT_EQ(dest_data[i], saved_dest_data[i]);
  }
}

template <typename T>
static void TestIndexAdd2D() {
  bool allow_minus_one;
  int32_t src_stride;
  int32_t dest_stride;
  int32_t num_src_rows;
  int32_t num_dest_rows;
  int32_t num_cols;
  for (int32_t i = 0; i != 8; ++i) {
    num_src_rows = RandInt(1, 100);
    num_dest_rows = RandInt(1, 100);
    num_cols = RandInt(1, 100);
    src_stride = RandInt(0, 10) + num_cols;
    dest_stride = RandInt(0, 10) + num_cols;
    allow_minus_one = RandInt(-1000, 1000) & 1;

    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();
    Array1<int32_t> indexes = GenerateRandomIndexes(
        context, allow_minus_one, num_src_rows, num_dest_rows - 1);

    Tensor src =
        GenerateRandTensor2D<T>(context, num_src_rows, num_cols, src_stride);
    Tensor dest =
        GenerateRandTensor2D<T>(context, num_dest_rows, num_cols, dest_stride);
    Tensor saved_dest = dest.Clone();

    IndexAdd(src, indexes, allow_minus_one, &dest);

    src = src.To(GetCpuContext());
    dest = dest.To(src.Context());
    indexes = indexes.To(dest.Context());
    saved_dest = saved_dest.To(src.Context());
    const T *src_data = src.Data<T>();
    const T *dest_data = dest.Data<T>();
    const int32_t *indexes_data = indexes.Data();
    T *saved_dest_data = saved_dest.Data<T>();
    for (int32_t i = 0; i != num_src_rows; ++i) {
      int32_t index = indexes_data[i];
      if (index == -1) continue;
      for (int j = 0; j != num_cols; ++j)
        saved_dest_data[index * num_cols + j] += src_data[i * num_cols + j];
    }

    int32_t n = num_dest_rows * num_cols;
    for (int32_t i = 0; i != n; ++i)
      EXPECT_EQ(dest_data[i], saved_dest_data[i]);
  }
}

TEST(IndexAdd, IndexAdd1D) {
  TestIndexAdd1D<float>();
  TestIndexAdd1D<double>();
  TestIndexAdd1D<int32_t>();
}

TEST(IndexAdd, IndexAdd2D) {
  TestIndexAdd2D<float>();
  TestIndexAdd2D<double>();
  TestIndexAdd2D<int32_t>();
}

}  // namespace k2
