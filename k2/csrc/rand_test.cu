/**
 * @brief Unin test for rand.cu.
 *
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/rand.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(Rand, CUDA) {
  ContextPtr c = GetCudaContext();
  if (c->GetDeviceType() == kCpu) return;  // NO CUDA capable GPU is available
  SetSeed(c, 20210113);
  Array1<float> a = Rand<float>(c, 3);
  EXPECT_EQ(a.Context()->GetDeviceType(), kCuda);
  K2_LOG(INFO) << a;  // [0.411307 0.29121 0.732587]

  Rand(&a);
  K2_LOG(INFO) << a;  // [0.575718 0.517407 0.176948]

  Array1<double> b = Rand<double>(c, 3);
  K2_LOG(INFO) << b;  // [0.552894, 0.720282, 0.906524]
#if 0
  import torch
  torch.manual_seed(20210113)
  n = 3
  print(torch.rand(n, device='cuda') # [0.4113, 0.2912, 0.7326]
  print(torch.rand(n, device='cuda') # [0.5757, 0.5174, 0.1769]
  print(torch.rand(n, device='cuda', dtype=torch.float64) # [0.5529, 0.7203, 0.9065]
#endif
}

TEST(Rand, CPU) {
  ContextPtr context = GetCpuContext();
  SetSeed(context, 20210113);
  Array1<float> a = Rand<float>(context, 3);
  K2_LOG(INFO) << a;
  EXPECT_EQ(a.Context()->GetDeviceType(), kCpu);

  Array1<float> b = Rand<float>(context, 3);
  for (int32_t i = 0; i != a.Dim(); ++i) EXPECT_NE(a[i], b[i]);

  SetSeed(context, 20210113);
  Array1<float> c = Rand<float>(context, 3);
  Array1<float> d = Rand<float>(context, 3);

  CheckArrayData(a, c);
  CheckArrayData(b, d);

  Array1<double> f = Rand<double>(context, 3);
  EXPECT_EQ(f.Dim(), 3);
}

}  // namespace k2
