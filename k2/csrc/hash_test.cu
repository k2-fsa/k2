/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
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
#include "k2/csrc/test_utils.h"
#include "k2/csrc/hash.h"

namespace k2 {


template <int32_t NUM_KEY_BITS>
void TestHashConstruct() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : { 128, 1024, 2048, 65536, 1048576}) {
      Hash hash(c, size, NUM_KEY_BITS);

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

      Array1<int32_t> count_per_key =
          GetCounts(reinterpret_cast<Array1<int32_t> &>(keys), key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
            *success_data = success.Data();
      int32_t   *counts_data = count_per_key.Data();
      Hash::Accessor<NUM_KEY_BITS> acc =
          hash.GetAccessor<Hash::Accessor<NUM_KEY_BITS>>();
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
                         success;
          int32_t count = counts_data[key];

          uint64_t *key_value_location;
          if (acc.Insert(key, value, nullptr, &key_value_location)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          uint64_t keyval = *key_value_location;
          if (success) {
            acc.SetValue(key_value_location, key, value);
            K2_DCHECK_EQ(keyval, *key_value_location);
          }
          success_data[i] = success;
        });

      K2_EVAL(c, num_elems, lambda_check_find, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
               success = success_data[i];

          uint64_t val = 0;
          uint64_t *key_val_addr = nullptr;
          bool ans = acc.Find(key, &val, &key_val_addr),
              ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
          K2_CHECK(ans);  // key should be present.
          K2_CHECK(!ans2);  // key == key + key_bound should not be present.

          if (success) {
            // if this was the key that won the data race, its value should be
            // present.
            K2_CHECK_EQ(val, value);
            K2_CHECK_EQ(*key_val_addr,
                        ((uint64_t(key) | ((uint64_t)value << NUM_KEY_BITS))));
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


void TestHashConstructGeneric(int32_t num_key_bits) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : {128, 1024, 2048, 65536, 1048576}) {
      Hash hash(c, size, num_key_bits);

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

      Array1<int32_t> count_per_key =
          GetCounts(reinterpret_cast<Array1<int32_t> &>(keys), key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
            *success_data = success.Data();
      int32_t *counts_data = count_per_key.Data();

      Hash::GenericAccessor acc = hash.GetAccessor<Hash::GenericAccessor>();
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
                         success;
          int32_t count = counts_data[key];

          uint64_t *key_value_location;
          if (acc.Insert(key, value, nullptr, &key_value_location)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          uint64_t keyval = *key_value_location;
          if (success) {
            acc.SetValue(key_value_location, key, value);
            K2_DCHECK_EQ(keyval, *key_value_location);
          }
          success_data[i] = success;
        });

      hash.Resize(hash.NumBuckets() * 2, num_key_bits);

      acc = hash.GetAccessor<Hash::GenericAccessor>();

      K2_EVAL(c, num_elems, lambda_check_find, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
               success = success_data[i];

          uint64_t val = 0;
          uint64_t *key_val_addr = nullptr;
          bool ans = acc.Find(key, &val, &key_val_addr),
              ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
          K2_CHECK(ans);  // key should be present.
          K2_CHECK(!ans2);  // key == key + key_bound should not be present.

          if (success) {
            // if this was the key that won the data race, its value should be
            // present.
            K2_CHECK_EQ(val, value);
            K2_CHECK_EQ(*key_val_addr,
                        ((uint64_t(value) << num_key_bits) | (uint64_t)key));
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


void TestHashConstructPacked(int32_t num_key_bits,
                             int32_t num_value_bits) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : { 2048, 65536, 1048576}) {
      Hash hash(c, size, num_key_bits, num_value_bits);

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

      Array1<int32_t> count_per_key =
          GetCounts(reinterpret_cast<Array1<int32_t> &>(keys), key_bound);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint32_t *keys_data = keys.Data(),
            *values_data = values.Data(),
            *success_data = success.Data();
      int32_t *counts_data = count_per_key.Data();

      Hash::PackedAccessor acc = hash.GetAccessor<Hash::PackedAccessor>();
      K2_EVAL(c, num_elems, lambda_insert_pairs, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
                         success;
          int32_t count = counts_data[key];

          uint64_t *key_value_location;
          if (acc.Insert(key, value, nullptr, &key_value_location)) {
            success = 1;
          } else {
            success = 0;
            K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
          }
          uint64_t keyval = *key_value_location;
          if (success) {
            acc.SetValue(key_value_location, key, value);
            K2_DCHECK_EQ(keyval, *key_value_location);
          }
          success_data[i] = success;
        });

      if (size != 65535)      // just for some variety..
        num_value_bits += 1;  // Try changing the number of value bits, so we
                              // can test Resize() with changes in that.

      hash.Resize(hash.NumBuckets() * 2, num_key_bits,
                  num_value_bits);

      acc = hash.GetAccessor<Hash::PackedAccessor>();
      const uint64_t *hash_data = hash.Data();

      K2_EVAL(c, num_elems, lambda_check_find, (int32_t i) -> void {
          uint32_t key = keys_data[i],
                 value = values_data[i],
              success = success_data[i];

          int32_t num_implicit_key_bits = num_key_bits + num_value_bits - 64,
              num_kept_key_bits = num_key_bits - num_implicit_key_bits;
          uint64_t implicit_key_bits_mask =
              (uint64_t(1) << num_implicit_key_bits) - 1;

          uint64_t val = 0;
          uint64_t *key_val_addr = nullptr;
          bool ans = acc.Find(key, &val, &key_val_addr),
              ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
          K2_CHECK(ans);  // key should be present.
          K2_CHECK(!ans2);  // key == key + key_bound should not be present.

          if (success) {
            // if this was the key that won the data race, its value should be
            // present.
            K2_CHECK_EQ(val, value);
            K2_CHECK_EQ(*key_val_addr,
                        ((uint64_t(value) << num_kept_key_bits) |
                         (((uint64_t)key) >> num_implicit_key_bits)));
            K2_CHECK_EQ(key & implicit_key_bits_mask,
                        (key_val_addr - hash_data) & implicit_key_bits_mask);
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

void TestHash64Construct() {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t size : {128, 1024, 2048, 65536, 1048576}) {
      Hash64 hash(c, size);

      // obviously we're not going to fill it completely... this hash is not
      // resizable.
      int32_t num_elems = size / 2;

      // Some keys may be identical.
      int32_t key_bound = num_elems * 2;
      Array1<uint64_t> keys = RandUniformArray1<uint64_t>(c, num_elems, 0,
                                                          key_bound - 1),
                       values =
                           RandUniformArray1<uint64_t>(c, num_elems, 0, 10000),
                       success(c, num_elems, 0);

      Array1<uint64_t> cpu_keys = keys.To(GetCpuContext());
      Array1<int32_t> count_per_key(GetCpuContext(), key_bound, 0);
      int32_t *count_per_key_data = count_per_key.Data();

      for (int32_t i = 0; i < cpu_keys.Dim(); ++i) {
        ++count_per_key_data[cpu_keys[i]];
      }
      count_per_key = count_per_key.To(c);

      if (size <= 2048) {
        K2_LOG(INFO) << "keys = " << keys << ", values = " << values
                     << ", counts = " << count_per_key;
      }
      uint64_t *keys_data = keys.Data(), *values_data = values.Data(),
               *success_data = success.Data();
      int32_t *counts_data = count_per_key.Data();
      Hash64::Accessor acc = hash.GetAccessor();
      K2_EVAL(
          c, num_elems, lambda_insert_pairs, (int32_t i)->void {
            uint64_t key = keys_data[i], value = values_data[i], success;

            int32_t count = counts_data[key];

            uint64_t *key_value_location;
            if (acc.Insert(key, value, nullptr, &key_value_location)) {
              success = 1;
            } else {
              success = 0;
              K2_CHECK(count > 1) << ", key = " << key << ", i = " << i;
            }
            uint64_t keyval = *key_value_location;
            if (success) {
              acc.SetValue(key_value_location, value);
              K2_DCHECK_EQ(keyval, *key_value_location);
            }
            success_data[i] = success;
          });

      hash.Resize(hash.NumBuckets() * 2);
      acc = hash.GetAccessor();

      K2_EVAL(
          c, num_elems, lambda_check_find, (int32_t i)->void {
            uint64_t key = keys_data[i], value = values_data[i],
                     success = success_data[i];

            uint64_t val = 0;
            uint64_t *key_val_addr = nullptr;
            bool ans = acc.Find(key, &val, &key_val_addr),
                 ans2 = acc.Find(key + key_bound, &val, &key_val_addr);
            K2_CHECK(ans);    // key should be present.
            K2_CHECK(!ans2);  // key == key + key_bound should not be present.

            if (success) {
              // if this was the key that won the data race, its value should be
              // present.
              K2_CHECK_EQ(val, value);
              K2_CHECK_EQ(*key_val_addr, key);
              K2_CHECK_EQ(*(key_val_addr + 1), value);
            }
          });



      K2_EVAL(
          c, num_elems, lambda_check_delete, (int32_t i)->void {
            uint64_t key = (uint64_t)keys_data[i];
            uint64_t success = success_data[i];

            if (success) acc.Delete(key);
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
    TestHashConstructGeneric(key_bits);

  for (int32_t key_bits = 30; key_bits <= 40; key_bits += 4) {
    for (int32_t value_bits = (64 - key_bits) + 1;
         value_bits < (64 - key_bits) + 4;
         ++value_bits)
      TestHashConstructPacked(key_bits, value_bits);
  }
}

TEST(Hash64, Construct) {
  TestHash64Construct();
}

}  // namespace k2
