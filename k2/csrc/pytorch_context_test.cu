/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#if defined(K2_WITH_HIP)
// The c10/cuda/* headers pull a generated cuda_cmake_macros.h that only exists
// as the hip variant on a ROCm torch, so include the c10/hip/* header on every
// hipify generation; the device-fn namespace is selected below.
#include "c10/hip/HIPFunctions.h"
#else
#include "c10/cuda/CUDAFunctions.h"
#endif

#if defined(K2_WITH_HIP) && !defined(TORCH_HIPIFY_V2)
// torch hipify v1 renamed the device classes; c10::hip is the only spelling.
namespace c10_device = c10::hip;
#else
// hipify v2 masquerading or pure CUDA: the CUDA spelling is public.
namespace c10_device = c10::cuda;
#endif
#include "gtest/gtest.h"
#include "k2/csrc/test_utils.h"
//
#include "k2/csrc/array.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

// Use a separate function because there is a lambda function inside K2_EVAL().
static void TestImpl() {
  K2_LOG(INFO) << "Number of devices: " << c10_device::device_count();

  // Set the default device to 1
  c10_device::set_device(1);
  EXPECT_EQ(c10_device::current_device(), 1);

  ContextPtr c = GetCudaContext(0);
  EXPECT_EQ(c->GetDeviceId(), 0);

  // the default device should still be 1
  EXPECT_EQ(c10_device::current_device(), 1);

  Array1<int32_t> a(c, "[1 2]");
  EXPECT_EQ(a.Context()->GetDeviceId(), 0);

  // b uses the default device, which is 1
  Array1<int32_t> b(GetCudaContext(), "[10 20]");
  EXPECT_EQ(b.Context()->GetDeviceId(), 1);

  int32_t *a_data = a.Data();
  int32_t *b_data = b.Data();

  {
    DeviceGuard guard(0);
    // a is on device 0
    K2_EVAL(
        a.Context(), 2, set_a, (int32_t i)->void { a_data[i] += 1; });
    CheckArrayData(a, {2, 3});
  }

  {
    DeviceGuard guard(1);
    // b is on device 1
    K2_EVAL(
        b.Context(), 2, set_b, (int32_t i)->void { b_data[i] += 2; });

    CheckArrayData(b, {12, 22});
  }
}

TEST(PyTorchContext, GetCudaContext) {
  // skip this test is CUDA is not available
  if (!torch::cuda::is_available()) return;

  // skip it if there are less than two CUDA GPUs.
  if (c10_device::device_count() < 2) return;

  TestImpl();
}

}  // namespace k2
