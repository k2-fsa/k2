#include <iostream>

#include "cudpp.h"
#include "cudpp_plan.h"
#include "k2/csrc/array.h"

int main() {
  CUDPPConfiguration config;
  config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
  config.bucket_mapper = CUDPP_CUSTOM_BUCKET_MAPPER;

  std::cout << "hello cudpp\n";

  std::vector<int32_t> k{0, 4, 3, 1, 2, 6, 8, 9};
  std::vector<int32_t> v{0, 40, 30, 10, 20, 60, 80, 90};
  using namespace k2;

  Array1<int32_t> keys(GetCudaContext(), k);
  Array1<int32_t> values(GetCudaContext(), v);

  size_t num_elements = keys.Dim();
  size_t num_buckets = 3;
  CUDPPMultiSplitPlan plan(config, num_elements, num_buckets);
  uint32_t kk = 3;
  auto lambda = [kk] __device__(uint32_t i) -> uint32_t { return i % 3; };

  cudppMultiSplitCustomBucketMapper(&plan, (uint32_t *)keys.Data(),
                                    (uint32_t *)values.Data(), num_elements,
                                    num_buckets, lambda);
  K2_LOG(INFO) << keys;
  K2_LOG(INFO) << values;

  return 0;
}
