/**
 * @brief Unittest for hash
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/test_utils.h"
#include "k2/csrc/hash.h"

namespace k2 {


void TestHashConstruct() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size: { 128, 1024, 2048, 65536, 1048576 }) {
      Hash32 hash(c, size);

      // obviously we're not going to fill it completely... this hash is not
      // resizable.
      int32_t num_elems = size / 2;

      // Some keys may be identical.
      int32_t key_bound = num_elems * 2;
      Array1<int32_t> keys = RandUniformArray1(c, num_elems,
                                                0, key_bound - 1),
                    values = RandUniformArray1(c, num_elems,
                                               0, 10000),
                             success(c, num_elems, 0);

      Array1<int32_t> count_per_key = GetCounts(keys, key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      int32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
           *success_data = success.Data(),
            *counts_data = count_per_key.Data();
      Hash32::Accessor acc = hash.GetAccessor();
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = (uint32_t)keys_data[i];
          int32_t value = (int32_t)values_data[i],
                  count = counts_data[key],
                          success;
          if (Hash32::Insert(acc, key, value, nullptr)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          success_data[i] = success;
        });
    }
  }
}

TEST(Hash, Construct) {
  // This indirection gets around a limitation of the CUDA compiler.
  TestHashConstruct();
}

}  // namespace k2
