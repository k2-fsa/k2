/**
 * Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#include "k2/csrc/hash.h"

namespace k2 {

void Hash::CheckEmpty() {
  if (data_.Dim() == 0) return;
  ContextPtr c = Context();
  Array1<int32_t> error(c, 1, -1);
  int32_t *error_data = error.Data();
  uint64_t *hash_data = data_.Data();

  K2_EVAL(Context(), data_.Dim(), lambda_check_data, (int32_t i) -> void {
      if (~(hash_data[i]) != 0) error_data[0] = i;
    });
  int32_t i = error[0];
  if (i >= 0) {  // there was an error; i is the index into the hash where
    // there was an element.
    int64_t elem = data_[i];
    // We don't know the number of bits the user was using for the key vs.
    // value, so print in hex, maybe they can figure it out.
    K2_LOG(FATAL) << "Destroying hash: still contains values: position "
                  << i << ", key,value = " << std::hex << elem;
  }
}

void Hash::Resize(int32_t new_num_buckets, int32_t num_key_bits,
                  int32_t num_value_bits,  // = -1,
                  bool copy_data) {        // = true
  NVTX_RANGE(K2_FUNC);
  if (num_value_bits < 0)
    num_value_bits = 64 - num_key_bits;

  K2_CHECK_GT(new_num_buckets, 0);
  K2_CHECK_EQ(new_num_buckets & (new_num_buckets - 1), 0);  // power of 2.

  ContextPtr c = data_.Context();
  Hash new_hash(c, new_num_buckets,
                num_key_bits,
                num_value_bits);

  if (copy_data) {
    if (num_key_bits == num_key_bits_ &&
        num_value_bits == num_value_bits_ &&
        num_key_bits + num_value_bits == 64) {
      new_hash.CopyDataFromSimple(*this);
    } else {
      // we instantiate 2 versions of CopyDataFrom().
      if (new_hash.NumKeyBits() + new_hash.NumValueBits() == 64) {
        new_hash.CopyDataFrom<GenericAccessor>(*this);
      } else {
        new_hash.CopyDataFrom<PackedAccessor>(*this);
      }
    }
  }

  *this = new_hash;
  new_hash.Destroy();  // avoid failed check in destructor (it would otherwise
                       // expect the hash to be empty when destroyed).
}

}  // namespace k2
