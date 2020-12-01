/**
 * @brief Unittest for macros.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

static void TestEval() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> array = Range(c, 3, 0);
    int32_t *array_data = array.Data();
    K2_EVAL(
        c, array.Dim(), inc, (int32_t i)->void { array_data[i] += 1; });
    CheckArrayData(array, std::vector<int32_t>{1, 2, 3});

    // with multiple lines
    K2_EVAL(
        c, array.Dim(), inc, (int32_t kk)->void {
          array_data[kk] += 1;
          array_data[kk] += 1;
          array_data[kk] += 1;
        });
    CheckArrayData(array, std::vector<int32_t>{4, 5, 6});
  }
}

TEST(Macros, Eval) { TestEval(); }

}  // namespace k2
