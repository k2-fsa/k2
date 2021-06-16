/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {
TEST(TensorTest, Shape) {
  // scalar shape created with the default constructor
  {
    Shape shape;
    EXPECT_EQ(shape.NumAxes(), 0);
    EXPECT_EQ(shape.NumElements(), 1);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // empty shape representing scalar, created with empty dims
  {
    std::vector<int32_t> dims;
    Shape shape(dims);
    EXPECT_EQ(shape.NumAxes(), 0);
    EXPECT_EQ(shape.NumElements(), 1);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // empty shape created with empty dims and strides
  {
    std::vector<int32_t> dims;
    std::vector<int32_t> strides;
    Shape shape(dims, strides);
    EXPECT_EQ(shape.NumAxes(), 0);
    EXPECT_EQ(shape.NumElements(), 1);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1);
    EXPECT_TRUE(shape.IsContiguous());
  }

  // non-empty shape created with dims
  {
    std::vector<int32_t> dims = {3, 2, 5};
    Shape shape(dims);
    EXPECT_EQ(shape.NumAxes(), 3);
    EXPECT_EQ(shape.NumElements(), 30);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 30);
    EXPECT_TRUE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides());
    EXPECT_THAT(expected_strides, ::testing::ElementsAre(10, 5, 1));
  }

  // non-empty shape created with dims, but with zero elements due to one dim is
  // zero
  {
    std::vector<int32_t> dims = {3, 0, 5};
    Shape shape(dims);
    EXPECT_EQ(shape.NumAxes(), 3);
    EXPECT_EQ(shape.NumElements(), 0);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 0);
    EXPECT_TRUE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides());
    EXPECT_THAT(expected_strides, ::testing::ElementsAre(0, 5, 1));
  }


  // non-empty shape created with dims and strides, contiguous
  {
    std::vector<int32_t> dims = {3, 2, 5};
    std::vector<int32_t> strides = {10, 5, 1};
    Shape shape(dims, strides);
    EXPECT_EQ(shape.NumAxes(), 3);
    EXPECT_EQ(shape.NumElements(), 30);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 30);
    EXPECT_EQ(shape.StorageSize(), 30);
    EXPECT_TRUE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides());
    EXPECT_EQ(expected_strides, strides);
  }

  // non-empty shape created with dims and strides, non-contiguous
  {
    std::vector<int32_t> dims = {3, 2, 5};
    std::vector<int32_t> strides = {10, 8, 1};
    Shape shape(dims, strides);
    EXPECT_EQ(shape.NumAxes(), 3);
    EXPECT_EQ(shape.NumElements(), 30);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 33);
    EXPECT_FALSE(shape.IsContiguous());
    std::vector<int32_t> expected_dims(shape.Dims());
    EXPECT_EQ(expected_dims, dims);
    std::vector<int32_t> expected_strides(shape.Strides());
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
    std::vector<int32_t> src_data(tensor.GetShape().NumElements());
    std::iota(src_data.begin(), src_data.end(), 0);
    int32_t *data = tensor.Data<int32_t>();
    std::copy(src_data.begin(), src_data.end(), data);

    {
      Tensor sub_tensor = tensor.Index(0, 1);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(2, 4));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(4, 1));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {8, 9, 10, 11, 12, 13, 14, 15};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          // index = index[0] * stride[0] + index[1] * stride[1]
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }

    {
      Tensor sub_tensor = tensor.Index(2, 3);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(3, 2));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(8, 4));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {3, 7, 11, 15, 19, 23};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }

    {
      Tensor sub_tensor = tensor.Index(1, 1);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(3, 4));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(8, 1));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {4,  5,  6,  7,  12, 13,
                                                14, 15, 20, 21, 22, 23};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }
  }

  // non-contiguous version created with dims and strides
  {
    std::vector<int32_t> dims = {2, 2, 4};
    std::vector<int32_t> strides = {24, 10, 2};
    Shape shape(dims, strides);
    int64_t begin, end;
    shape.GetReachableElems(&begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 41);
    ContextPtr context = GetCpuContext();
    Tensor tensor(context, kInt32Dtype, shape);
    // 48 > end - begin = 41
    std::vector<int32_t> src_data(48);
    std::iota(src_data.begin(), src_data.end(), 0);
    int32_t *data = tensor.Data<int32_t>();
    std::copy(src_data.begin(),
              src_data.begin() + end, data);
    std::vector<int32_t> expected_data = {0,  2,  4,  6,  10, 12, 14, 16,
                                          24, 26, 28, 30, 34, 36, 38, 40};
    for (int32_t i = 0, j = 0; i < dims[0]; ++i) {
      for (int32_t m = 0; m < dims[1]; ++m) {
        for (int32_t n = 0; n < dims[2]; ++n) {
          int32_t value =
              data[i * strides[0] + m * strides[1] + n * strides[2]];
          EXPECT_EQ(value, expected_data[j++]);
        }
      }
    }


    // test ToContiguous
    Tensor t = ToContiguous(tensor);
    ASSERT_TRUE(t.IsContiguous());
    int32_t n = t.NumElements();
    const int32_t *t_data = t.Data<int32_t>();
    for (int32_t i = 0; i != t.NumElements(); ++i) {
      EXPECT_EQ(t_data[i], expected_data[i]);
    }

    for (int32_t axis = -3; axis < 3; axis++) {
      Tensor t_flip = Flip(t, axis);
      int32_t *t_flip_data = t_flip.Data<int32_t>();
      int32_t axis_mod = axis;
      if (axis < 0)
        axis_mod += 3;
      K2_CHECK_EQ(t_data, t_flip_data +
                              t_flip.Stride(axis_mod) * (t.Dim(axis_mod) - 1));
    }
  }

  // non-contiguous version created with region and bytes_offset
  {
    ContextPtr context = GetCpuContext();
    const int32_t element_size = TraitsOf(kInt32Dtype).NumBytes();
    const int32_t num_element = 48;
    auto region = NewRegion(context, num_element * element_size);
    std::vector<int32_t> src_data(num_element);
    std::iota(src_data.begin(), src_data.end(), 0);
    int32_t *data = region->GetData<int32_t, kCpu>();
    std::copy(src_data.begin(), src_data.end(), data);
    std::vector<int32_t> dims = {2, 2, 4};
    std::vector<int32_t> strides = {24, 10, 2};
    Shape shape(dims, strides);
    const int32_t bytes_offset = 4 * element_size;  // 4 is element offset
    Tensor tensor(kInt32Dtype, shape, region, bytes_offset);
    const int32_t *tensor_data = tensor.Data<int32_t>();
    std::vector<int32_t> expected_data = {4,  6,  8,  10, 14, 16, 18, 20,
                                          28, 30, 32, 34, 38, 40, 42, 44};
    for (int32_t i = 0, j = 0; i < dims[0]; ++i) {
      for (int32_t m = 0; m < dims[1]; ++m) {
        for (int32_t n = 0; n < dims[2]; ++n) {
          int32_t value =
              tensor_data[i * strides[0] + m * strides[1] + n * strides[2]];
          EXPECT_EQ(value, expected_data[j++]);
        }
      }
    }

    {
      Tensor sub_tensor = tensor.Index(0, 1);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(2, 4));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(10, 2));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {28, 30, 32, 34, 38, 40, 42, 44};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }

    {
      Tensor sub_tensor = tensor.Index(1, 0);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(2, 4));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(24, 2));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {4, 6, 8, 10, 28, 30, 32, 34};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }

    {
      Tensor sub_tensor = tensor.Index(2, 2);
      Shape sub_tensor_shape = sub_tensor.GetShape();
      std::vector<int32_t> expected_dims(sub_tensor_shape.Dims());
      EXPECT_THAT(expected_dims, ::testing::ElementsAre(2, 2));
      std::vector<int32_t> expected_strides(sub_tensor_shape.Strides());
      EXPECT_THAT(expected_strides, ::testing::ElementsAre(24, 10));
      const int32_t *sub_tensor_data = sub_tensor.Data<int32_t>();
      std::vector<int32_t> expected_sub_data = {8, 18, 32, 42};
      for (int32_t m = 0, j = 0; m < expected_dims[0]; ++m) {
        for (int32_t n = 0; n < expected_dims[1]; ++n) {
          int32_t value = sub_tensor_data[m * expected_strides[0] +
                                          n * expected_strides[1]];
          EXPECT_EQ(value, expected_sub_data[j++]);
        }
      }
    }
  }
}

}  // namespace k2
