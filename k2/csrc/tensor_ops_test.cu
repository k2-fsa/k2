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

namespace k2 {

/* Generate random indexes for testing `Index`.

   @param [in] context  The device context specifying where the returned
                        array resides.
   @param [in] allow_minus_one
                        If true, the returned array will contain values
                        in the range [-1, max_value]; [0, max_value] otherwise.
   @param [in] num_indexes_elements
                        It specifies the length of the returned array.
   @param [in] max_value  It specifies the maximum value the returned array can
                          contain.

   @return  Return a 1-D array that can be used for testing `Index`.
 */

static Array1<int32_t> GenerateRandomIndexes(ContextPtr context,
                                             bool allow_minus_one,
                                             int32_t num_indexes_elements,
                                             int32_t max_value) {
  std::vector<int32_t> indexes(num_indexes_elements);
  int32_t start = allow_minus_one ? -1 : 0;
  for (int32_t &i : indexes) {
    int32_t tmp = RandInt(-max_value, max_value);
    i = std::max(tmp, start);
  }

  return Array1<int32_t>(context, indexes);
}

/* Get the test data for `Index`.

   @param [in] context  It specifies the device where the output tensor resides.
   @param [in] stride   A positive number indicating the expected stride
                        of the output `tensor`.
   @param [in] allow_minus_one
                        If it is true, then -1 is a valid entry in the returned
                        `indexes` array. If it is false, all elements in the
                        returned `indexes` array are non-negative.
   @param [out] indexes The output indexes containing entries in the range
                        [0, tensor.Dim(0)) if allow_minus_one is false;
                        [-1, tensor.Dim(0)) if allow_minus_one is true.

   @return Returns a 1-D tensor with the given `stride`.
 */
template <typename T>
static Tensor GenerateRandDataForIndex1D(ContextPtr context, int32_t stride,
                                         bool allow_minus_one,
                                         Array1<int32_t> *indexes) {
  K2_CHECK_NE(indexes, nullptr);
  K2_CHECK_GT(stride, 0);

  int32_t num_tensor_elements = RandInt(1, 100);
  int32_t num_indexes_elements = RandInt(1, 20000);

  *indexes = GenerateRandomIndexes(
      context, allow_minus_one, num_indexes_elements, num_tensor_elements - 1);

  std::vector<T> data_vec(num_tensor_elements);
  for (T &d : data_vec) d = RandInt(-1000, 1000);

  Shape shape({num_tensor_elements}, {stride});

  Array1<T> array(context, data_vec);
  const T *array_data = array.Data();

  Tensor ans(context, DtypeOf<T>::dtype, shape);
  T *ans_data = ans.Data<T>();
  K2_EVAL(
      context, num_tensor_elements, lambda_set,
      (int32_t i)->void { ans_data[i * stride] = array_data[i]; });
  return ans;
}

/* Get the test data for `Index`.

   @param [in] context  It specifies the device where the output tensor resides.
   @param [in] stride   A positive number indicating the expected row stride
                        of the output `tensor`.
   @param [in] num_rows  Number of rows in the returned tensor.
   @param [in] num_cols  Number of columns in the returned tensor.
   @param [in] allow_minus_one
                        If it is true, then -1 is a valid entry in the returned
                        `indexes` array. If it is false, all elements in the
                        returned `indexes` array are non-negative.
   @param [out] indexes The output indexes containing entries in the range
                        [0, tensor.Dim(0)) if allow_minus_one is false;
                        [-1, tensor.Dim(0)) if allow_minus_one is true.

   @return Returns a 2-D tensor with the given `stride`, `num_rows` and
           `num_cols`.
 */
template <typename T>
static Tensor GenerateRandDataForIndex2D(ContextPtr context, int32_t stride,
                                         int32_t num_rows, int32_t num_cols,
                                         bool allow_minus_one,
                                         Array1<int32_t> *indexes) {
  int32_t num_tensor_elements = num_rows * num_cols;
  int32_t num_indexes_elements = RandInt(1, 10);
  K2_CHECK_GT(num_cols, 0);
  K2_CHECK_GE(stride, num_cols);
  K2_CHECK_GE(num_rows, 0);

  *indexes = GenerateRandomIndexes(context, allow_minus_one,
                                   num_indexes_elements, num_rows - 1);

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
  for (int32_t i = 0; i != 8; ++i) {
    stride = RandInt(1, 10);
    allow_minus_one = RandInt(-1000, 1000) & 1;
    Array1<int32_t> indexes;
    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();
    Tensor src = GenerateRandDataForIndex1D<T>(context, stride, allow_minus_one,
                                               &indexes);
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
  for (int32_t i = 0; i != 8; ++i) {
    num_rows = RandInt(1, 100);
    num_cols = RandInt(1, 100);
    stride = RandInt(0, 10) + num_cols;
    allow_minus_one = RandInt(-1000, 1000) & 1;
    Array1<int32_t> indexes;
    ContextPtr context = (i & 1) ? GetCpuContext() : GetCudaContext();
    Tensor src = GenerateRandDataForIndex2D<T>(
        context, stride, num_rows, num_cols, allow_minus_one, &indexes);
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

// TODO(fangjun): use random generated data for testing
template <typename T>
static void TestIndexAdd1D() {
  {
    bool allow_minus_one = true;
    std::vector<T> src_vec = {10, 20, 30, 40, 5};
    std::vector<int32_t> indexes_vec = {-1, 0, 3, 2, 0};

    // ContextPtr context = GetCpuContext();
    ContextPtr context = GetCudaContext();
    Array1<T> src_array(context, src_vec);
    Array1<int32_t> indexes_array(context, indexes_vec);
    Tensor src = src_array.ToTensor();

    Shape shape({4});
    Tensor dest(context, DtypeOf<T>::dtype, shape);
    T *dest_data = dest.Data<T>();

    K2_EVAL(
        context, 4, lambda_set_zero, (int32_t i)->void { dest_data[i] = 0; });
    K2_LOG(INFO) << Array1<T>(src);
    K2_LOG(INFO) << Array1<T>(dest);
    K2_LOG(INFO) << indexes_array;
    IndexAdd(src, indexes_array, allow_minus_one, &dest);
    K2_LOG(INFO) << Array1<T>(dest);
  }

  {
    bool allow_minus_one = false;
    std::vector<T> src_vec = {10, 20, 30, 40, 5, 9, 8};
    std::vector<int32_t> indexes_vec = {0, 2, 3, 2, 0, 1, 1};

    // ContextPtr context = GetCpuContext();
    ContextPtr context = GetCudaContext();
    Array1<T> src_array(context, src_vec);
    Array1<int32_t> indexes_array(context, indexes_vec);
    Tensor src = src_array.ToTensor();

    Shape shape({4});
    Tensor dest(context, DtypeOf<T>::dtype, shape);
    T *dest_data = dest.Data<T>();

    K2_EVAL(
        context, 4, lambda_set_zero, (int32_t i)->void { dest_data[i] = 0; });
    K2_LOG(INFO) << Array1<T>(src);
    K2_LOG(INFO) << Array1<T>(dest);
    K2_LOG(INFO) << indexes_array;
    IndexAdd(src, indexes_array, allow_minus_one, &dest);
    K2_LOG(INFO) << Array1<T>(dest);
  }
}

template <typename T>
static void TestIndexAdd2D() {
  bool allow_minus_one = true;

  std::vector<T> src_vec = {10, 20, 30, 40, 50, 60, 70, 80, 5, 8, 3, 6};
  std::vector<int32_t> indexes_vec = {-1, 0, 1, 1, 0, 0};

  // ContextPtr context = GetCpuContext();
  ContextPtr context = GetCudaContext();
  Array1<T> src_array(context, src_vec);
  const T *src_array_data = src_array.Data();
  Array1<int32_t> indexes_array(context, indexes_vec);
  Shape src_shape({6, 2});
  Tensor src(context, DtypeOf<T>::dtype, src_shape);
  T *src_data = src.Data<T>();
  K2_EVAL(
      context, 12, lambda_copy_data,
      (int32_t i)->void { src_data[i] = src_array_data[i]; });

  Shape shape({2, 2});
  Tensor dest(context, DtypeOf<T>::dtype, shape);
  T *dest_data = dest.Data<T>();

  K2_EVAL(
      context, 4, lambda_set_zero, (int32_t i)->void { dest_data[i] = 0; });
  K2_LOG(INFO) << Array1<T>(src);
  K2_LOG(INFO) << Array1<T>(dest);
  K2_LOG(INFO) << indexes_array;
  IndexAdd(src, indexes_array, allow_minus_one, &dest);
  K2_LOG(INFO) << Array1<T>(dest);
}

TEST(IndexAdd, IndexAdd1D) {
  TestIndexAdd1D<float>();
  TestIndexAdd1D<int32_t>();
}

TEST(IndexAdd, IndexAdd2D) {
  TestIndexAdd2D<float>();
  TestIndexAdd2D<int32_t>();
}

}  // namespace k2
