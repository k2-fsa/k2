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


template <int32_t NUM_KEY_BITS>
void TestHashConstruct() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : {128, 1024, 2048, 65536, 1048576}) {
      Hash hash(c, size);

      // obviously we're not going to fill it completely... this hash is not
      // resizable.
      int32_t num_elems = size / 2;

      // Some keys may be identical.
      int32_t key_bound = num_elems * 2;
      Array1<uint32_t> keys = RandUniformArray1<uint32_t>(c, num_elems,
                                                0, key_bound - 1),
                     values = RandUniformArray1<uint32_t>(c, num_elems,
                                               0, 10000),
                             success(c, num_elems, 0);

      Array1<int32_t> count_per_key = GetCounts(reinterpret_cast<Array1<int32_t>&>(keys),
                                                key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
            *success_data = success.Data();
      int32_t   *counts_data = count_per_key.Data();
      const int32_t NUM_VALUE_BITS = 64 - NUM_KEY_BITS;
      Hash::Accessor<NUM_KEY_BITS> acc = hash.GetAccessor<NUM_KEY_BITS>();
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
                         success;
          int32_t count = counts_data[key];

          if (acc.Insert(key, value, nullptr)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          success_data[i] = success;
        });

      K2_EVAL(c, num_elems, lambda_check_find, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
               success = success_data[i];

          uint64_t val;
          uint64_t *key_val_addr;
          bool ans = acc.Find(key, &val, &key_val_addr),
              ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
          K2_CHECK(ans);  // key should be present.
          K2_CHECK(!ans2);  // key == key + key_bound should not be present.

          if (success) {
            // if this was the key that won the data race, its value should be
            // present.
            K2_CHECK_EQ(val, value);
            K2_CHECK_EQ(*key_val_addr, ((uint64_t(key) << NUM_VALUE_BITS) | (uint64_t)value));
          }
        });

      K2_EVAL(c, num_elems, lambda_check_delete, (int32_t i) -> void {
          uint32_t key = (uint32_t)keys_data[i];
          uint32_t success = success_data[i];

          if (success)
            acc.Delete(key);
        });
    }
  }
}


void TestHashConstruct2(int32_t num_key_bits) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : {128, 1024, 2048, 65536, 1048576}) {
      Hash hash(c, size);

      // obviously we're not going to fill it completely... this hash is not
      // resizable.
      int32_t num_elems = size / 2;

      // Some keys may be identical.
      int32_t key_bound = num_elems * 2;
      Array1<uint32_t> keys = RandUniformArray1<uint32_t>(c, num_elems,
                                                0, key_bound - 1),
                     values = RandUniformArray1<uint32_t>(c, num_elems,
                                               0, 10000),
                             success(c, num_elems, 0);

      Array1<int32_t> count_per_key = GetCounts(reinterpret_cast<Array1<int32_t>&>(keys),
                                                key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
            *success_data = success.Data();
      int32_t   *counts_data = count_per_key.Data();
      const int32_t num_value_bits = 64 - num_key_bits;

      Hash::GenericAccessor acc = hash.GetGenericAccessor(num_key_bits);
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
                         success;
          int32_t count = counts_data[key];

          if (acc.Insert(key, value, nullptr)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          success_data[i] = success;
        });

      hash.Resize(hash.NumBuckets() * 2, num_key_bits);

      acc = hash.GetGenericAccessor(num_key_bits);

      K2_EVAL(c, num_elems, lambda_check_find, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
               success = success_data[i];

          uint64_t val;
          uint64_t *key_val_addr;
          bool ans = acc.Find(key, &val, &key_val_addr),
              ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
          K2_CHECK(ans);  // key should be present.
          K2_CHECK(!ans2);  // key == key + key_bound should not be present.

          if (success) {
            // if this was the key that won the data race, its value should be
            // present.
            K2_CHECK_EQ(val, value);
            K2_CHECK_EQ(*key_val_addr, ((uint64_t(key) << num_value_bits) | (uint64_t)value));
          }
        });

      K2_EVAL(c, num_elems, lambda_check_delete, (int32_t i) -> void {
          uint32_t key = (uint32_t)keys_data[i];
          uint32_t success = success_data[i];

          if (success)
            acc.Delete(key);
        });
    }
  }
}


TEST(Hash, Construct) {
  // This indirection gets around a limitation of the CUDA compiler.
  TestHashConstruct<32>();

  TestHashConstruct<40>();

  TestHashConstruct<28>();

  for (int32_t key_bits = 28; key_bits <= 40; key_bits += 4)
    TestHashConstruct2(key_bits);
}

}  // namespace k2
