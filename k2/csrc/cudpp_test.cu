/**
 * k2/csrc/cudpp_test.cu
 *
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

template <typename T>
static void TestSegmentedExclusiveSum() {
  std::vector<T> expected_v = {0, 1, 3, 6,
                               //
                               0, 3, 7,
                               //
                               0, 5, 11, 18};

  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Ragged<T> src("[ [1 2 3 -1] [3 4 -1] [] [5 6 7 -1] ]");
    src = src.To(c);
    Array1<T> dst(c, src.values.Dim());

    SegmentedExclusiveSum(src, &dst);

    Array1<T> expected(c, expected_v);
    CheckArrayData(dst, expected);

    SegmentedExclusiveSum(src, &src.values);
    CheckArrayData(src.values, expected);
  }
}

TEST(CUDPP, SegmentedExclusiveSum) {
  TestSegmentedExclusiveSum<int32_t>();
  TestSegmentedExclusiveSum<float>();
}

}  // namespace k2
