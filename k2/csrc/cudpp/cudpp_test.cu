#include <iostream>

#include "cudpp.h"
#include "cudpp_plan.h"
#include "k2/csrc/array.h"

int main() {
  CUDPPConfiguration config;
  config.algorithm = CUDPP_MULTISPLIT;
  config.datatype = CUDPP_UINT;
  config.options = CUDPP_OPTION_KEYS_ONLY;
  config.bucket_mapper = CUDPP_DEFAULT_BUCKET_MAPPER;

  std::cout << "hello cudpp\n";

  std::vector<int32_t> v{5, 4, 3, 2, 1, 0};
  using namespace k2;

  Array1<int32_t> keys(GetCudaContext(), v);
  Array1<int32_t> buckets(keys.Context(), 3);

  int32_t num_elements = keys.Dim();
  int32_t num_buckets = buckets.Dim();
  CUDPPMultiSplitPlan plan(config, num_elements, num_buckets);
  cudppMultiSplit(&plan, (uint32_t *)keys.Data(), nullptr, num_elements,
                  num_buckets);
  K2_LOG(INFO) << keys;

  return 0;
}
