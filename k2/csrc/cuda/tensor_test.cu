// k2/csrc/cuda/tensor_test.cu

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
TEST(TensorTest, Shape) {
  // empty shape created with the default constructor
  {
    Shape shape;
    EXPECT_EQ(shape.Ndim(), 0);
    EXPECT_EQ(shape.Nelement(), 0);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // empty shape created with empty dims
  {
    std::vector<int32_t> dims;
    Shape shape(dims);
    EXPECT_EQ(shape.Ndim(), 0);
    EXPECT_EQ(shape.Nelement(), 0);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // empty shape created with empty dims and strides
  {
    std::vector<int32_t> dims;
    std::vector<int32_t> strides;
    Shape shape(dims, strides);
    EXPECT_EQ(shape.Ndim(), 0);
    EXPECT_EQ(shape.Nelement(), 0);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // non-empty shape created with dims
  {
    std::vector<int32_t> dims = {3, 2, 5};
    Shape shape(dims);
    EXPECT_EQ(shape.Ndim(), 3);
    EXPECT_EQ(shape.Nelement(), 30);
    EXPECT_TRUE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims(),
                                       shape.Dims() + shape.Ndim());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides(),
                                          shape.Strides() + shape.Ndim());
    EXPECT_THAT(expected_strides, ::testing::ElementsAre(10, 5, 1));
  }

  // non-empty shape created with dims and strides, contiguous
  {
    std::vector<int32_t> dims = {3, 2, 5};
    std::vector<int32_t> strides = {10, 5, 1};
    Shape shape(dims, strides);
    EXPECT_EQ(shape.Ndim(), 3);
    EXPECT_EQ(shape.Nelement(), 30);
    EXPECT_TRUE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims(),
                                       shape.Dims() + shape.Ndim());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides(),
                                          shape.Strides() + shape.Ndim());
    EXPECT_EQ(expected_strides, strides);
  }

  // non-empty shape created with dims and strides, non-contiguous
  {
    std::vector<int32_t> dims = {3, 2, 5};
    std::vector<int32_t> strides = {10, 8, 1};
    Shape shape(dims, strides);
    EXPECT_EQ(shape.Ndim(), 3);
    EXPECT_EQ(shape.Nelement(), 30);
    EXPECT_FALSE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims(),
                                       shape.Dims() + shape.Ndim());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides(),
                                          shape.Strides() + shape.Ndim());
    EXPECT_EQ(expected_strides, strides);
  }
}

TEST(TensorTest, Tensor) {
  // contiguous version created with dims
  {
    std::vector<int32_t> dims = {3, 2, 4};
    Shape shape(dims);
    ContextPtr context = GetCpuContext();
    Tensor tensor(context, kInt32Dtype, shape);
    std::vector<int32_t> src_data(tensor.GetShape().Nelement());
    std::iota(src_data.begin(), src_data.end(), 0);
    int32_t *data = tensor.Data<int32_t>();
    std::copy(src_data.begin(), src_data.end(), data);

    {
      TensorPtr sub_tensor = tensor.Index(0, 1);
      Shape sub_tensor_shape = sub_tensor->GetShape();
      std::vector<int32_t> expected_dims(
          sub_tensor_shape.Dims(),
          sub_tensor_shape.Dims() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(2, 4));
      std::vector<int32_t> expected_strides(
          sub_tensor_shape.Strides(),
          sub_tensor_shape.Strides() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(4, 1));
      int32_t *sub_tensor_data = sub_tensor->Data<int32_t>();
      std::vector<int32_t> sub_tensor_data_copy(
          sub_tensor_data, sub_tensor_data + sub_tensor_shape.Nelement());
      EXPECT_THAT(sub_tensor_data_copy,
                  ::testing::ElementsAre(8, 9, 10, 11, 12, 13, 14, 15));
    }

    {
      TensorPtr sub_tensor = tensor.Index(2, 3);
      Shape sub_tensor_shape = sub_tensor->GetShape();
      std::vector<int32_t> expected_dims(
          sub_tensor_shape.Dims(),
          sub_tensor_shape.Dims() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(3, 2));
      std::vector<int32_t> expected_strides(
          sub_tensor_shape.Strides(),
          sub_tensor_shape.Strides() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(8, 4));
      int32_t *sub_tensor_data = sub_tensor->Data<int32_t>();
      // the element is (3, 7, 11, 15, 19, 23)
      // index = index[0] * stride[0] + index[1] * stride[1]
      EXPECT_EQ(sub_tensor_data[0 * 8 + 0 * 4], 3);
      EXPECT_EQ(sub_tensor_data[0 * 8 + 1 * 4], 7);
      EXPECT_EQ(sub_tensor_data[1 * 8 + 0 * 4], 11);
      EXPECT_EQ(sub_tensor_data[1 * 8 + 1 * 4], 15);
      EXPECT_EQ(sub_tensor_data[2 * 8 + 0 * 4], 19);
      EXPECT_EQ(sub_tensor_data[2 * 8 + 1 * 4], 23);
    }

    {
      TensorPtr sub_tensor = tensor.Index(1, 1);
      Shape sub_tensor_shape = sub_tensor->GetShape();
      std::vector<int32_t> expected_dims(
          sub_tensor_shape.Dims(),
          sub_tensor_shape.Dims() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(3, 4));
      std::vector<int32_t> expected_strides(
          sub_tensor_shape.Strides(),
          sub_tensor_shape.Strides() + sub_tensor_shape.Ndim());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(8, 1));
      int32_t *sub_tensor_data = sub_tensor->Data<int32_t>();
      // check some values
      EXPECT_EQ(sub_tensor_data[0 * 8 + 0 * 1], 4);
      EXPECT_EQ(sub_tensor_data[1 * 8 + 2 * 1], 14);
      EXPECT_EQ(sub_tensor_data[2 * 8 + 3 * 1], 23);
    }
  }

  // TODO(haowen): non-contiguous version created with region and bytes_offset
}

}  // namespace k2
