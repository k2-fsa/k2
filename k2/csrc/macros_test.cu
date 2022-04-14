/**
 * Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <gtest/gtest.h>

#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

/*static*/ void TestEval() {
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

/*static*/ void TestEval2() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> array1 = Range(c, 6, 0);
    Array2<int32_t> array(array1, 2, 3);
    int32_t *array_data = array.Data();
    int32_t elem_stride0 = array.ElemStride0();
    ContextPtr &context = array.Context();
    K2_EVAL2(
        context, array.Dim0(), array.Dim1(), lambda_inc,
        (int32_t i, int32_t j)->void {
          array_data[i * elem_stride0 + j] += 1;
        });
    CheckArrayData(array.Flatten(), std::vector<int32_t>{1, 2, 3, 4, 5, 6});
  }
}

TEST(Macros, Eval) { TestEval(); }
TEST(Macros, Eval2) { TestEval2(); }

}  // namespace k2
