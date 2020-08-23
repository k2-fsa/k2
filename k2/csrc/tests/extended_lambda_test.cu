// k2/csrc/tests/extended_lambda_test.cu

// Copyright (c) 2020 Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <vector>

namespace k2 {
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

bool test_extended_lambda_type() {
  auto d_lambda = [] __device__ {};
  auto hd_lambda = [] __host__ __device__ {};

  static_assert(
      __nv_is_extended_device_lambda_closure_type(decltype(d_lambda)),
      "");
  static_assert(
      __nv_is_extended_host_device_lambda_closure_type(decltype(hd_lambda)),
      "");

  return true;
}

class CudaFunctionalTest : public ::testing::Test {
 public:
  void saxpy(float *x, float *y, float a, int N) {
    auto r = thrust::counting_iterator<int>(0);

    auto lambda = [=] __host__ __device__ (int i) {
      y[i] = a * x[i] + y[i];
    };

    if(N > gpuThreshold)
      thrust::for_each(thrust::device, r, r+N, lambda);
    else
      thrust::for_each(thrust::host, r, r+N, lambda);
  }

 protected:
  CudaFunctionalTest() {}

  void initialize(int N) {
    x = thrust::host_vector<float>(N, 1.0);
    y = thrust::host_vector<float>(N, 2.0);

    d_x = x;
    d_y = y;
  }

  ~CudaFunctionalTest() override {}

  int gpuThreshold = 1 << 10;
  thrust::host_vector<float> x;
  thrust::host_vector<float> y;
  thrust::device_vector<float> d_x;
  thrust::device_vector<float> d_y;
};

TEST_F(CudaFunctionalTest, ExtendedLambda) {
  // Check if the extended lambdas have the right closure type.
  {
    EXPECT_TRUE(test_extended_lambda_type());
  }

  /* Use extended lambda since cuda-8.0 to initialize template function (for_each),
   * to implement host/device choice at runtime,
   * depends on gpuThreshold.*/
  {
    int N = gpuThreshold / 2; // N < gpuThreshold
    initialize(N);

    // call host lambda as N < gpuThreshold, thus pass the host_vector
    saxpy(thrust::raw_pointer_cast(x.data()),
          thrust::raw_pointer_cast(y.data()),
          2.0,
          N);
    EXPECT_THAT(y, thrust::host_vector<float>(N, 4.0));
  }

  {
    int N = gpuThreshold * 2; // N > gpuThreshold
    initialize(N);

    // call device lambda as N > gpuThreshold, thus pass the device_vector
    saxpy(thrust::raw_pointer_cast(d_x.data()),
          thrust::raw_pointer_cast(d_y.data()),
          2.0,
          N);
    EXPECT_THAT(d_y, thrust::device_vector<float>(N, 4.0));
  }
}
}  // end namespace k2
