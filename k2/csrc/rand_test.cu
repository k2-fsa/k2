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

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/rand.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(RandReal, CUDA) {
  ContextPtr c = GetCudaContext();
  if (c->GetDeviceType() == kCpu) return;  // NO CUDA capable GPU is available
  SetSeed(c, 20210113);
  Array1<float> a = Rand<float>(c, 0.0f, 1.0f, 3);
  EXPECT_EQ(a.Context()->GetDeviceType(), kCuda);
  K2_LOG(INFO) << a;  // [0.411307 0.29121 0.732587]

  Rand(0.0f, 1.0f, &a);
  K2_LOG(INFO) << a;  // [0.575718 0.517407 0.176948]

  Array1<double> b = Rand<double>(c, 0.0, 100.0, 3);
  K2_LOG(INFO) << b;  // [55.2894, 72.0282, 90.6524]
  /*
    import torch
    torch.manual_seed(20210113)
    n = 3
    print(torch.rand(n, device='cuda') # [0.4113, 0.2912, 0.7326]
    print(torch.rand(n, device='cuda') # [0.5757, 0.5174, 0.1769]
    print(torch.rand(n, device='cuda', dtype=torch.float64) # [0.5529, 0.7203,
    0.9065]
  */
}

TEST(RandReal, CPU) {
  ContextPtr context = GetCpuContext();
  SetSeed(context, 20210113);
  Array1<float> a = Rand<float>(context, 0.0f, 1.0f, 3);
  K2_LOG(INFO) << a;
  EXPECT_EQ(a.Context()->GetDeviceType(), kCpu);

  Array1<float> b = Rand<float>(context, 0.0f, 1.0f, 3);
  for (int32_t i = 0; i != a.Dim(); ++i) EXPECT_NE(a[i], b[i]);

  SetSeed(context, 20210113);
  Array1<float> c = Rand<float>(context, 0.0f, 1.0f, 3);
  Array1<float> d = Rand<float>(context, 0.0f, 1.0f, 3);

  CheckArrayData(a, c);
  CheckArrayData(b, d);

  Array1<double> f = Rand<double>(context, 0.0, 1.0, 3);
  EXPECT_EQ(f.Dim(), 3);
}

TEST(RandInt, CPU) {
  ContextPtr c = GetCpuContext();
  SetSeed(c, 20210114);
  int32_t low = 0;
  int32_t high = 5;
  int32_t dim = 100000;
  Array1<int32_t> a = Rand<int32_t>(c, low, high, dim);
  ASSERT_EQ(a.Dim(), dim);
  EXPECT_EQ(a.Context()->GetDeviceType(), kCpu);

  for (int32_t i = 0; i != dim; ++i) {
    EXPECT_GE(a[i], low);
    EXPECT_LT(a[i], high);
  }
}

TEST(RandInt, CUDA) {
  ContextPtr c = GetCudaContext();
  if (c->GetDeviceType() == kCpu) return;  // NO CUDA capable GPU is available
  SetSeed(c, 20210114);
  Array1<int32_t> a = Rand<int32_t>(c, 0, 100, 3);
  K2_LOG(INFO) << a;  // [75, 10, 86]

  Array1<int32_t> b(c, 5);
  Rand<int32_t>(b.Context(), -100, 200, b.Dim(), b.Data());
  K2_LOG(INFO) << b;  // [2, -92, -19, 103, 154]
  /*
    import torch
    torch.manual_seed(20210114)
    print(torch.randint(0, 100, (3,), dtype=torch.int32, device='cuda'))
    # tensor([ 75, 10, 86 ], device = 'cuda:0', dtype = torch.int32)

    print(torch.randint(-100, 200, (5,), dtype=torch.int32, device='cuda'))
    # tensor([  2, -92, -19, 103, 154], device='cuda:0', dtype=torch.int32)
  */
}

template <typename T>
/*static*/ void TestBounds(T low, T high) {
  int32_t dim = 100000;
  ContextPtr cpu = GetCpuContext();
  ContextPtr cuda = GetCudaContext();
  SetSeed(cpu, 20210114);
  SetSeed(cuda, 20210114);
  for (int k = 0; k != 8; ++k) {
    for (auto &context : {cpu, cuda}) {
      Array1<T> array = Rand(context, low, high, dim);
      ASSERT_EQ(array.Dim(), dim);
      EXPECT_TRUE(array.Context()->IsCompatible(*context));

      const T *array_data = array.Data();
      K2_EVAL(
          context, dim, lambda_check, (int32_t i)->void {
            K2_CHECK_GE(array_data[i], low);
            K2_CHECK_LT(array_data[i], high);
          });
    }
  }
}

TEST(Rand, BoundsCheck) {
  TestBounds<float>(0, 1);
  TestBounds<double>(0, 1);
  TestBounds<int32_t>(0, 5);
}

}  // namespace k2
