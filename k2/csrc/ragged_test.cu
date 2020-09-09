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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.cuh"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/tensor.h"

// returns a random ragged shape where the dims on axis 1 are all the same
// (so: can be transposed).
RandomRaggedShapeToTranspose(Context c) {
  Context c_cpu = CpuContext();

  Ragged random =  RandomRagged<T>().To(c);

  int32_t input_dim0 = random.Dim0(), divisor = 1;
  for (int32_t i = 1; i * i <= input_dim0; i++) {
    if (dim0 % i == 0 && i > divisor)
      divisor = i;
  }

  int32_t output_dim0 = divisor, output_dim1 = dim0 / divisor;

  Array1<int32_t> row_splits = Range<int32_t>(c, output_dim0 + 1, 0, output_dim1);
  int32_t cached_tot_size = input_dim0;

  RaggedShape top_level_shape = RaggedShape2(&row_splits, nullptr, cached_tot_size);
  return ComposeRaggedShapes(top_level_shape, random);


  int32_t axis_a_len = RandIntGeometric(5, 100),
          axis_b_len = RandIntGeometric(5, 100);
}

namespace k2 {
template <DeviceType d>
void TestTranspose() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  RaggedShape to_transpose = RandomRaggedShapeToTranspose(context);
  RaggedShape transposed = Transpose(to_transpose);

  if (d != kCpu) {
    ContextPtr c = GetCpuContext();
    to_tranpose = to_transpose.To(c);
    transposed = transposed.To(c);
  }

  for (auto iter = transposed.Iterator(); !iter.Done(); iter.Next()) {
    std::vector<int32_t> index = iter.Value();
    int32_t i = transposed[index];  // Just make sure this doesn't crash, dont
                                    // need the value.
    std::swap(index[0], index[1]);
    i = to_transpose[index];  // dont need the value, just need to make
                              // sure it's an allowable index.
  }
  for (auto iter = to_transpose.Iterator(); !iter.Done(); iter.Next()) {
    std::vector<int32_t> index = iter.Value();
    std::swap(index[0], index[1]);
    int32_t i = transposed[index];  // dont need the value, just need to make
                                      // sure it's an allowable index.
  }
}

  // TODO(dan): add more tests
}

TEST(RaggedTest, TestTranspose) {
  TestTranspose<kCpu>();
  TestTranspose<kCuda>();
}

}  // namespace k2
