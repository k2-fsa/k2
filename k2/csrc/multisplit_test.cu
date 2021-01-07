/**
 * @brief Unittest for MultiSplit
 *
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/multisplit.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

template <typename ValueType>
static void TestKeyValuePairs() {
  ContextPtr context = GetCudaContext();
  // this test requires CUDA capable GPUs
  if (context->GetDeviceType() == kCpu) return;

  std::vector<int32_t> k{0, 0, 3, 1, 2, 1};
  std::vector<ValueType> v{8, 9, 30, 100, 200, 1000};

  Array1<int32_t> in_keys(context, k);
  Array1<ValueType> in_values(context, v);

  int32_t num_buckets = 4;
  int32_t num_elements = in_keys.Dim();

  auto lambda_bucket_mapping = [] __device__(uint32_t i) -> uint32_t {
    return i;
  };

  uint32_t (*bucket_mapping)(uint32_t) = nullptr;
  for (int32_t i = 0; i != 2; ++i) {
    Array1<int32_t> out_keys(context, k.size());
    Array1<ValueType> out_values(context, v.size());

    if (i == 0) {
      MultiSplitKeyValuePairs(
          context, num_elements, num_buckets, bucket_mapping,
          reinterpret_cast<const uint32_t *>(in_keys.Data()),
          reinterpret_cast<const uint32_t *>(in_values.Data()),
          reinterpret_cast<uint32_t *>(in_keys.Data()),
          reinterpret_cast<uint32_t *>(in_values.Data()));
    } else {
      MultiSplitKeyValuePairs(
          context, num_elements, num_buckets, &lambda_bucket_mapping,
          reinterpret_cast<const uint32_t *>(in_keys.Data()),
          reinterpret_cast<const uint32_t *>(in_values.Data()),
          reinterpret_cast<uint32_t *>(in_keys.Data()),
          reinterpret_cast<uint32_t *>(in_values.Data()));
    }
    CheckArrayData(in_keys, {0, 0, 1, 1, 2, 3});
    CheckArrayData(in_values, {8, 9, 100, 1000, 200, 30});
  }
}

static void TestKeysOnly() {
  ContextPtr context = GetCudaContext();
  // this test requires CUDA capable GPUs
  if (context->GetDeviceType() == kCpu) return;

  std::vector<int32_t> k{0, 0, 3, 1, 2, 1};
  Array1<int32_t> in_keys(context, k);

  int32_t num_buckets = 4;
  int32_t num_elements = in_keys.Dim();

  auto lambda_bucket_mapping = [] __device__(uint32_t i) -> uint32_t {
    return i;
  };

  uint32_t (*bucket_mapping)(uint32_t) = nullptr;
  for (int32_t i = 0; i != 2; ++i) {
    if (i == 0) {
      MultiSplitKeysOnly(context, num_elements, num_buckets, bucket_mapping,
                         reinterpret_cast<const uint32_t *>(in_keys.Data()),
                         reinterpret_cast<uint32_t *>(in_keys.Data()));
    } else {
      MultiSplitKeysOnly(context, num_elements, num_buckets,
                         &lambda_bucket_mapping,
                         reinterpret_cast<const uint32_t *>(in_keys.Data()),
                         reinterpret_cast<uint32_t *>(in_keys.Data()));
    }
    CheckArrayData(in_keys, {0, 0, 1, 1, 2, 3});
  }
}

TEST(MultiSplit, TestKeyValuePairs) {
  TestKeyValuePairs<int32_t>();
  TestKeyValuePairs<float>();
}

TEST(MultiSplit, TestKeysOnly) { TestKeysOnly(); }

}  // namespace k2
