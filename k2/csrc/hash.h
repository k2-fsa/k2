/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Wei kang)
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

#ifndef K2_CSRC_HASH_H_
#define K2_CSRC_HASH_H_

#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/eval.h"
#include "k2/csrc/log.h"
#include "k2/csrc/utils.h"

namespace k2 {


// __host__ __device__ version of CUDA's atomicCAS (copy and swap).  In the host
// case it assumes the calling code is single-threaded and that `compare` is the
// value that was just read from `address`, so it assumes it still has that
// value and returns it.  The compiler should be able to optimize calling code.
unsigned long long int __forceinline__ __host__ __device__ AtomicCAS(
    unsigned long long int* address,
    unsigned long long int compare,
    unsigned long long int val) {
#ifdef __CUDA_ARCH__
  return atomicCAS(address, compare, val);
#else
  // For host code, we assume single-threaded for now).
  unsigned long long int res = *address;
  *address = *address == compare ? val : *address;
  return res;
#endif
}

/*
  How class Hash works:

    - It can function as a map from key=uint32_t to value=uint32_t, or from
      key=uint64_t to value=uint64_t, but you cannot use all 64 bits in the
      key and value because we compress both of them into a single 64-bit
      integer. There are several different modes of using this hash,
      depending which accessor objects you use.  The modes are:

        - Use Accessor<NUM_KEY_BITS> with num_key_bits known at compile time;
          the number of values bits will be 64 - NUM_KEY_BITS.
        - Use GenericAccessor, which is like Accessor but the number of
          key bits is not known at compile time; and they both must still
          sum to 64.
        - Use PackedAccessor, which allows you to have the number of key
          plus value bits greater than 64; the rest of the bits are
          implicit in groups of buckets (the number of buckets must
          be >= 32 * 1 << (num_key_bits + num_value_bits - 64).

    - You must decide the number of key and value bits, and the number of
      buckets, when you create the hash, but you can resize it (manually)
      and when you resize it you can change the number of key and value bits.

   Some constraints:
    - You can store any (key,value) pair allowed by the number of key and value
      bits, except the pair where all the bits of
      both and key and value are set [that is used to mean "nothing here"]
    - The number of buckets must always be a power of 2.
    - When deleting values from the hash you must delete them all at
      once (necessary because there is no concept of a "tombstone".

   Some notes on usage:

   You use it by: constructing it, obtaining its Accessor with GetAccessor()
   with appropriate template args depending on your chosen accessor type; and
   inside kernels (or host code), calling functions Insert(), Find() or Delete()
   of the Accessor object.  Resizing is not automatic; it is the user's
   responsibility to make sure the hash does not get too full (which could cause
   assertion failures in kernels, and will be very slow).

   Some implementation notes:
    - When accessing hash[key], we use bucket_index == key % num_buckets,
      bucket_inc = 1 | (((key * 2) / num_buckets) ^ key).
    - If the bucket at `bucket_index` is occupied, we look in locations
      `(bucket_index + n * bucket_inc)%num_buckets` for n = 1, 2, ...;
      this choice ensures that if multiple keys hash to the same bucket,
      they don't all access the same sequence of locations; and bucket_inc
      being odd ensures we eventually try all locations (of course for
      reasonable hash occupancy levels, we shouldn't ever have to try
      more than two or three).

*/
class Hash {
 public:
  /* Constructor.  Context can be for CPU or GPU.

     @param [in] num_buckets   Number of buckets in the hash; must be
                a power of 2 and >= 128 (this limit was arbitrarily chosen).
                The number of items in the hash cannot exceed the number of
                buckets, or the code will loop infinitely when you try to add
                items; aim for less than 50% occupancy.
     @param [in] num_key_bits   Number of bits in the key of the hash;
                must satisfy 0 < num_key_bits < 64, and keys used must
                be less than (1<<num_key_bits)-1.
     @param [in] num_value_bits  Number of bits in the value of the hash;
                if not specified, will be set to 64 - num_key_bits.  There
                are constraints on the num_value_bits, it interacts with
                which accessor you use.  For Accessor<> or GenericAccessor,
                we require that num_key_bits + num_value_bits == 64.
                For PackedAccessor we allow that num_key_bits + num_value_bits > 64,
                but with the constraint that
                  (num_buckets >> (64 - num_key_bits - num_value_bits)) >= 32
  */
  Hash(ContextPtr c,
       int32_t num_buckets,
       int32_t num_key_bits,
       int32_t num_value_bits=-1):
      num_key_bits_(num_key_bits) {
    std::ostringstream os;
    if (num_value_bits < 0)
      num_value_bits = 64 - num_key_bits;
    os << K2_FUNC << ":num_buckets=" << num_buckets << ", num_key_bits="
       << num_key_bits << ", num_value_bits=" << num_value_bits;
    NVTX_RANGE(os.str().c_str());
    data_ = Array1<uint64_t>(c, num_buckets, ~(uint64_t)0);
    K2_CHECK_GE(num_buckets, 128);
    int32_t n = 2;
    for (buckets_num_bitsm1_ = 0; n < num_buckets;
         n *= 2, buckets_num_bitsm1_++) { }
    K2_CHECK_EQ(num_buckets, 2 << buckets_num_bitsm1_)
        << " num_buckets must be a power of 2.";
    num_value_bits_ = num_value_bits;

    int32_t num_implicit_bits = num_key_bits_ + num_value_bits - 64;
    K2_CHECK_GE(num_implicit_bits, 0);

    // keys that hash to a group of buckets of size (num_buckets >>
    // num_implicit_bits) always need to stay inside that group, so
    // it's not good if that group size is too small; even with
    // a good hashing function, one of those groups may end up
    // becoming full by chance.
    K2_CHECK_GE(num_buckets >> num_implicit_bits, 32) <<
        "Hash being full is too likely; bad configuration.";
  }

  // Only to be used prior to assignment.
  Hash() = default;


  int32_t NumKeyBits() const { return num_key_bits_; }

  int32_t NumValueBits() const { return num_value_bits_; }

  int32_t NumBuckets() const { return data_.Dim(); }

  // Returns data pointer; for testing..
  uint64_t *Data() { return data_.Data(); }

  /*
    Copies all data elements from `src` to `*this`.  Requires
    that num_key_bits_ and num_value_bits_ are the same between *this
    and src, and that num_key_bits_ + num_value_bits_ == 64
    (i.e. no packed data).

    See also CopyDataFrom().
   */
  void CopyDataFromSimple(Hash &src) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_EQ(num_key_bits_, src.num_key_bits_);
    K2_CHECK_EQ(num_value_bits_, src.num_value_bits_);
    K2_CHECK_EQ(num_key_bits_ + num_value_bits_, 64);
    int32_t num_buckets = data_.Dim(),
        src_num_buckets = src.data_.Dim();
    const uint64_t *src_data = src.data_.Data();
    uint64_t *data = data_.Data();
    const uint64_t key_mask = (uint64_t(1) << num_key_bits_) - 1;
    size_t new_num_buckets_mask = static_cast<size_t>(num_buckets) - 1,
        new_buckets_num_bitsm1 = buckets_num_bitsm1_;
    ContextPtr c = data_.Context();
    K2_EVAL(c, src_num_buckets, lambda_copy_data, (int32_t i) -> void {
        uint64_t key_value = src_data[i];
        if (~key_value == 0) return;  // equals -1.. nothing there.
        uint64_t key = key_value & key_mask,
            bucket_inc = 1 | ((key >> new_buckets_num_bitsm1) ^ key);
        size_t cur_bucket = key & new_num_buckets_mask;
        while (1) {
          uint64_t assumed = ~((uint64_t)0),
              old_elem = AtomicCAS((unsigned long long*)(data + cur_bucket),
                                   assumed, key_value);
          if (old_elem == assumed) return;
          cur_bucket = (cur_bucket + bucket_inc) & new_num_buckets_mask;
          // Keep iterating until we find a free spot in the new hash...
        }
      });
  }

  /*
    Copies data from another hash `src`.
    AccessorT is a suitable accessor type for *this* hash.
  */
  template <typename AccessorT> void CopyDataFrom(Hash &src) {
    NVTX_RANGE(K2_FUNC);
    AccessorT this_acc(*this);
    // We handle the general case where `src` may possibly be packed (i.e. we
    // allow num-(key+value)-bits > 64).  This function is only called while
    // resizing if key/value bits doin't match, which isn't that common, so
    // handling this fairly generally shouldn't slow us down much.
    int32_t src_num_key_bits = src.NumKeyBits(),
        src_num_value_bits = src.NumValueBits(),
        src_num_implicit_key_bits = src_num_key_bits + src_num_value_bits - 64,
        src_num_kept_key_bits = src_num_key_bits - src_num_implicit_key_bits;
    const int64_t src_implicit_key_mask =
        (uint64_t(1) << src_num_implicit_key_bits) - 1,
        src_kept_key_mask = (uint64_t(1) << src_num_kept_key_bits) - 1;
    uint64_t *src_data = src.data_.Data();
    ContextPtr c = data_.Context();
    K2_EVAL(c, src.NumBuckets(), lambda_copy_data, (int32_t i) -> void {
        uint64_t key_value = src_data[i];
        if (~key_value != 0) {
          uint64_t kept_key = key_value & src_kept_key_mask,
              value = key_value >> src_num_kept_key_bits,
              key = (kept_key << src_num_implicit_key_bits) |
              (i & src_implicit_key_mask);
          bool insert_success = this_acc.Insert(key, value);
          K2_CHECK_EQ(insert_success, true);
        }
      });
  }

  /* Resize the hash to a new number of buckets.

       @param [in] new_num_buckets   New number of buckets; must be a power of 2,
                  and must be large enough to accommodate all values in the hash
                  (we assume the caller is keeping track of the number of elements
                  in the hash somehow).
       @param [in] num_key_bits  The number of bits used to
                  store the keys, with 0 < num_key_bits < 64 (number of bits in
                  the values is 64 minus this).  This must be the same as was
                  used to add any values that are currently in the hash.
       @param [in] num_value_bits  Number of bits in the value of the hash.
                 If not specified it defaults to the current number of value
                 bits if num_key_bits == -1, else to 64 - num_key_bits; in future
                 we'll allow more bits than that, by making some bits of
                 the key implicit in the bucket index.

     CAUTION: Resizing will invalidate any accessor objects you have; you need
     to re-get the accessors before accessing the hash again.
  */
  void Resize(int32_t new_num_buckets,
              int32_t num_key_bits,
              int32_t num_value_bits = -1,
              bool copy_data = true);

  // Shallow copy
  Hash &operator=(const Hash &src) = default;
  // Copy constructor (shallow copy)
  explicit Hash(const Hash &src) = default;

  ContextPtr &Context() const { return data_.Context(); }

  /*
     class Acccessor is the accessor object that is applicable when
     hash.NumKeyBits() + hash.NumValueBits() == 64, and hash.NumKeyBits() is
     known at compile time.  See also GenericAccessor and PackedAccessor.

     Note: we may decide at some point to have a specific overload of this
     Accessor template for where the number of bits is 32.

     Be careful with these Accessor objects; you have to be consistent, with
     a given Hash object (if it has elements in it), to use only a single type
     of Accessor object.
  */
  template <int32_t NUM_KEY_BITS> class Accessor {
   public:
    Accessor(Hash &hash):
        data_(hash.data_.Data()),
        num_buckets_mask_(uint32_t(hash.NumBuckets())-1),
        buckets_num_bitsm1_(hash.buckets_num_bitsm1_) {
      K2_CHECK_EQ(NUM_KEY_BITS, hash.NumKeyBits());
      K2_CHECK_EQ(hash.NumKeyBits() + hash.NumValueBits(), 64);
    }

    // Copy constructor
    Accessor(const Accessor &src) = default;

   /*
    Try to insert pair (key,value) into hash.
      @param [in] key  Key into hash; it is required that no bits except the lowest-order
                       NUM_KEY_BITS may be set.
      @param [in] value  Value to set; it is is required that no bits except the
                     lowest-order (NUM_VALUE_BITS = 64 - NUM_KEY_BITS) may be set;
                     it is also an error if ~((key << NUM_VALUE_BITS) | value) == 0,
                     i.e. if all the allowed bits of both `key` and `value` are
                     set.
      @param [out] old_value  If not nullptr, this location will be set to
                    the existing value *if this key was already present* in the
                    hash (or set by another thread in this kernel), i.e. only if
                    this function returns false.
      @param [out] key_value_location  If not nullptr, its contents will be
                    set to the address of the (key,value) pair (either the
                    existing or newly-written one).
      @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value,
        uint64_t *old_value = nullptr,
        uint64_t **key_value_location = nullptr) const {
      uint32_t cur_bucket = static_cast<uint32_t>(key) & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      constexpr int64_t KEY_MASK = (uint64_t(1)<<NUM_KEY_BITS) - 1,
          VALUE_MASK = (uint64_t(1)<< (64 - NUM_KEY_BITS)) - 1;

      K2_DCHECK_EQ((key & ~KEY_MASK) | (value & ~VALUE_MASK), 0);

      uint64_t new_elem = (value << NUM_KEY_BITS) | key;
      while (1) {
        uint64_t cur_elem = data_[cur_bucket];
        if ((cur_elem & KEY_MASK) == key) {
          if (old_value) *old_value = (cur_elem >> NUM_KEY_BITS);
          if (key_value_location) *key_value_location = data_ + cur_bucket;
          return false;  // key exists in hash
        }
        else if (~cur_elem == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_elem = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                        cur_elem, new_elem);
          if (old_elem == cur_elem) {
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return true;  // Successfully inserted.
          }
          cur_elem = old_elem;
          if ((cur_elem & KEY_MASK) == key) {
            if (old_value) *old_value = (cur_elem >> NUM_KEY_BITS);
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because bucket_inc is odd (so only satisfies
        // (n * bucket_inc) % num_buckets == 0 for n == num_buckets).
        // Note: n here is the number of times we went around the loop.
        cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
      }
    }



    /*
     Look up this key in this hash; output the value and optionally the
     location of the (key,value) pair if found.

      @param [in] key    Key to look up; bits other than the lowest-order NUM_KEY_BITS
                      bits must not be set.
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with key_value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] key_value_location  (optional) The memory address of the
                       (key,value) pair, in case the caller wants to overwrite
                       the value via SetValue(); must be used for no other
                       purpose.
      @return          Returns true if an item with this key was found in the
                       hash, otherwise false.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Find(
        uint64_t key, uint64_t *value_out,
        uint64_t **key_value_location = nullptr) const {
      constexpr int64_t KEY_MASK = (uint64_t(1) << NUM_KEY_BITS) - 1;

      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (~old_elem == 0) {
          return false;
        } else if ((old_elem & KEY_MASK) == key) {
          *value_out = old_elem >> NUM_KEY_BITS;
          if (key_value_location)
            *key_value_location = data_ + cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] key  Required to be the same key that was provided to
                        Find(); it is an error otherwise.
          @param [in] value  Value to write; bits of higher order than
                       (NUM_VALUE_BITS = 64 - NUM_KEY_BITS) may not be set.
                       It is also an error if ~((key << NUM_VALUE_BITS) | value) == 0,
                       i.e. if all the allowed bits of both `key` and `value` are
                       set; but this is not checked.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ void SetValue(
        uint64_t *key_value_location, uint64_t key, uint64_t value) const {
      *key_value_location = (value << NUM_KEY_BITS) | key;
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find(); this version reads the key from the location rather than accepting
      it as an argument.  It's more efficient to use the other version where
      possible, to avoid the memory load.

          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] key  Required to be the same key that was provided to
                        Find(); it is an error otherwise.
          @param [in] value  Value to write; bits of higher order than
                       (NUM_VALUE_BITS = 64 - NUM_KEY_BITS) may not be set.
                       It is also an error if ~((key << NUM_VALUE_BITS) | value) == 0,
                       i.e. if all the allowed bits of both `key` and `value` are
                       set; but this is not checked.
          @return     Returns the key value present at this location.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ uint64_t SetValue(
        uint64_t *key_value_location, uint64_t value) const {
      uint64_t old_pair = *key_value_location;
      K2_CHECK_NE(~old_pair, 0);  // Check it was not an empty location.
      const int64_t KEY_MASK = (uint64_t(1) << NUM_KEY_BITS) - 1;
      uint64_t key = old_pair & KEY_MASK;
      uint64_t new_pair = key | (value << NUM_KEY_BITS);
      *key_value_location = new_pair;
      return key;
    }



    /* Deletes a key from a hash.  Caution: this cannot be combined with other
       operations on a hash; after you delete a key you cannot do Insert() or
       Find() until you have deleted all keys.  This is an open-addressing hash
       table with no tombstones, which is why this limitation exists).

       @param [in] key   Key to be deleted.   Each key present in the hash must
                         be deleted  by exactly one thread, or it will loop
                         forever!

      Note: the const is with respect to the metadata only; required, to avoid
      compilation errors.
    */
    __forceinline__ __host__ __device__ void Delete(uint64_t key) const {
      constexpr int64_t KEY_MASK = (uint64_t(1) << NUM_KEY_BITS) - 1;
      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if ((old_elem & KEY_MASK) == key) {
          data_[cur_bucket] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

   private:
    // pointer to data
    uint64_t *data_;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask_;
    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t buckets_num_bitsm1_;
  };


  /* class GenericAccessor is the version of the accessor object that is for
     use when hash.NumKeyBits() + hash.NumValueBits() == 64 and
     hash.NumKeyBits() is not known at compile time.  See also Accessor
     and PackedAccessor
   */
  class GenericAccessor {
   public:
    GenericAccessor(Hash &hash):
        num_key_bits_(hash.num_key_bits_),
        buckets_num_bitsm1_(hash.buckets_num_bitsm1_),
        data_(hash.data_.Data()),
        num_buckets_mask_(uint32_t(hash.NumBuckets() - 1)) {
      K2_CHECK_EQ(hash.num_key_bits_ + hash.num_value_bits_, 64);
    }

    // Copy constructor
    GenericAccessor(const GenericAccessor &src) = default;

   /*
    Try to insert pair (key,value) into hash.
      @param [in] key  Key into hash; it is required that no bits except the lowest-order
                    num_key_bits may be set.
      @param [in] value  Value to set; it is is required that no bits except the
                     lowest-order num_value_bits = 64 - num_key_bits may be set;
                     it is also an error if ~((key << num_value_bits) | value) == 0,
                     i.e. if all the allowed bits of both `key` and `value` are
                     set.
      @param [out] old_value  If not nullptr, this location will be set to
                    the existing value *if this key was already present* in the
                    hash (or set by another thread in this kernel), i.e. only if
                    this function returns false.
      @param [out] key_value_location  If not nullptr, its contents will be
                    set to the address of the (key,value) pair (either the
                    existing or newly-written one).
      @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value,
        uint64_t *old_value = nullptr,
        uint64_t **key_value_location = nullptr) const {
      uint32_t cur_bucket = static_cast<uint32_t>(key) & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      const uint32_t num_key_bits = num_key_bits_;
      const uint64_t key_mask = (uint64_t(1) << num_key_bits) - 1,
          not_value_mask = (uint64_t(-1) << (64 - num_key_bits));

      K2_DCHECK_EQ((key & ~key_mask) | (value & not_value_mask), 0);

      uint64_t new_elem =  (value << num_key_bits) | key;
      while (1) {
        uint64_t cur_elem = data_[cur_bucket];
        if ((cur_elem & key_mask) == key) {
          if (old_value) *old_value = cur_elem >> num_key_bits;
          if (key_value_location) *key_value_location = data_ + cur_bucket;
          return false;  // key exists in hash
        }
        else if (~cur_elem == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_elem = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                        cur_elem, new_elem);
          if (old_elem == cur_elem) {
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return true;  // Successfully inserted.
          }
          cur_elem = old_elem;
          if ((cur_elem & key_mask) == key) {
            if (old_value) *old_value = cur_elem >> num_key_bits;
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because bucket_inc is odd (so only satisfies
        // (n * bucket_inc) % num_buckets == 0 for n == num_buckets).
        // Note: n here is the number of times we went around the loop.
        cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
      }
    }

    /*
      Look up this key in the hash; output the value and optionally the
      location of the (key,value) pair if found.

      @param [in] key    Key to look up; bits other than the lowest-order num_key_bits_
                      bits must not be set.
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with key_value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] key_value_location  (optional) The memory address of the
                       (key,value) pair, in case the caller wants to overwrite
                       the value via SetValue(); must be used for no other
                       purpose.
      @return          Returns true if an item with this key was found in the
                       hash, otherwise false.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Find(
        uint64_t key, uint64_t *value_out,
        uint64_t **key_value_location = nullptr) const {
      const uint32_t num_key_bits = num_key_bits_;
      const int64_t key_mask = (uint64_t(1) << num_key_bits) - 1;

      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (~old_elem == 0) {
          return false;
        } else if ((old_elem & key_mask) == key) {
          *value_out = old_elem >> num_key_bits;
          if (key_value_location)
            *key_value_location = data_ + cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] key  Required to be the same key that was provided to
                        Find(); it is an error otherwise.
          @param [in] value  Value to write; bits of higher order than
                       (num_value_bits = 64 - num_key_bits) may not be set.
                       It is also an error if ~((key << num_value_bits) | value) == 0,
                       i.e. if all the allowed bits of both `key` and `value` are
                       set; but this is not checked.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ void SetValue(
        uint64_t *key_value_location, uint64_t key, uint64_t value) const {
      *key_value_location = (value << num_key_bits_) | key;
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().  This overload does not require the user to specify the old key.
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] value  Value to write; bits of higher order than
                       (num_value_bits = 64 - num_key_bits) may not be set.
                       It is also an error if ~((key << num_value_bits) | value) == 0,
                       where `key` is the existing key-- i.e. if all the allowed bits
                       of both `key` and `value` are set; but this is not checked.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ uint64_t SetValue(
        uint64_t *key_value_location, uint64_t value) const {
      uint64_t old_pair = *key_value_location;
      K2_CHECK_NE(~old_pair, 0);  // Check it was not an empty location.
      const int64_t key_mask = (uint64_t(1) << num_key_bits_) - 1;
      uint64_t key = old_pair & key_mask;
      uint64_t new_pair = key | (value << num_key_bits_);
      *key_value_location = new_pair;
      return key;
    }

    /* Deletes a key from a hash.  Caution: this cannot be combined with other
       operations on a hash; after you delete a key you cannot do Insert() or
       Find() until you have deleted all keys.  This is an open-addressing hash
       table with no tombstones, which is why this limitation exists).

       @param [in] key   Key to be deleted.   Each key present in the hash must
                         be deleted  by exactly one thread, or it will loop
                         forever!

      Note: the const is with respect to the metadata only; required, to avoid
      compilation errors.
    */
    __forceinline__ __host__ __device__ void Delete(uint64_t key) const {
      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      const uint64_t key_mask = (uint64_t(1) << num_key_bits_) - 1;
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if ((old_elem & key_mask) == key) {
          data_[cur_bucket] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

   private:
    // A number satisfying 0 < num_key_bits_ < 64; the number of bits
    // (out of 64) used for the key (rest are used for the value).
    uint32_t num_key_bits_;

    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t buckets_num_bitsm1_;

    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask_;

    // pointer to data
    uint64_t *data_;
  };


  /*
    class PackedAccessor is the accessor object that is applicable when
    hash.NumKeyBits() + hash.NumValueBits() >= 64 (i.e. not just for when the
    sum is 64); hash.NumKeyBits() and hash.NumValueBits() do not need to be
    known at compile time.  See also classes Accessor and GenericAccessor.

    Obviously we can't pack more than 64 bits into a 64-bit value; we let the
    lowest-order (num_implicit_bits = num_key_bits + num_value_bits - 64) bits
    of the key be implicit and equal to the `num_implicit_bits` lowest-order
    bits of the index of the hash bucket..
  */
  class PackedAccessor {
   public:
    PackedAccessor(Hash &hash):
        num_key_bits_(hash.num_key_bits_),
        num_kept_key_bits_(64 - hash.num_value_bits_),
        num_implicit_key_bits_(num_key_bits_ - num_kept_key_bits_),
        buckets_num_bitsm1_(hash.buckets_num_bitsm1_),
        data_(hash.data_.Data()),
        num_buckets_mask_(uint32_t(hash.NumBuckets() - 1)) {
      K2_CHECK_GE(hash.num_key_bits_ + hash.num_value_bits_, 64);
      K2_CHECK_GT(num_kept_key_bits_, 0);
      K2_CHECK_GE(num_implicit_key_bits_, 0);
    }

    // Copy constructor
    PackedAccessor(const PackedAccessor &src) = default;


   /*
    Try to insert pair (key,value) into hash.
      @param [in] key  Key into hash; it is required that no bits except the
                    lowest-order num_key_bits may be set.
      @param [in] value  Value to set; it is is required that no bits except the
                    lowest-order num_value_bits may be set; it is also an error
                    if ~((key << num_value_bits) | value) == 0, i.e. if all the
                    allowed bits of both `key` and `value` are set.
      @param [out] old_value  If not nullptr, this location will be set to
                    the existing value *if this key was already present* in the
                    hash (or set by another thread in this kernel), i.e. only if
                    this function returns false.
      @param [out] key_value_location  If not nullptr, its contents will be
                    set to the address of the (key,value) pair (either the
                    existing or newly-written one).
      @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value,
        uint64_t *old_value = nullptr,
        uint64_t **key_value_location = nullptr) const {
      uint32_t cur_bucket = static_cast<uint32_t>(key) & num_buckets_mask_;
      // Shifting `bucket_inc` right by num_implicit_key_bits_ ensures that
      // the lowest-order `num_implicit_key_bits_` bits of the bucket index will
      // not change when we fail over to the next location.  Without this, our
      // scheme would not work.
      uint32_t bucket_inc = (1 | ((key >> buckets_num_bitsm1_) ^ key))
          << num_implicit_key_bits_;
      uint64_t kept_key = key >> num_implicit_key_bits_;

      const uint64_t kept_key_mask = (uint64_t(1) << num_kept_key_bits_) - 1,
          not_value_mask = (uint64_t(-1) << (64 - num_kept_key_bits_));

      K2_DCHECK_EQ((kept_key & ~kept_key_mask) | (value & not_value_mask), 0);

      uint64_t new_elem = (value << num_kept_key_bits_) | kept_key;
      while (1) {
        uint64_t cur_elem = data_[cur_bucket];
        if ((cur_elem & kept_key_mask) == kept_key) {
          if (old_value) *old_value = cur_elem >> num_kept_key_bits_;
          if (key_value_location) *key_value_location = data_ + cur_bucket;
          return false;  // key exists in hash
        }
        else if (~cur_elem == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_elem = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                        cur_elem, new_elem);
          if (old_elem == cur_elem) {
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return true;  // Successfully inserted.
          }
          cur_elem = old_elem;
          if ((cur_elem & kept_key_mask) == kept_key) {
            if (old_value) *old_value = cur_elem >> num_kept_key_bits_;
            if (key_value_location) *key_value_location = data_ + cur_bucket;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.
        cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
      }
    }

    /*
     Look up this key in this hash; output the value and optionally the
     location of the (key,value) pair if found.

      @param [in] key    Key to look up; bits other than the lowest-order num_key_bits_
                      bits must not be set.
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with key_value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] key_value_location  (optional) The memory address of the
                       (key,value) pair, in case the caller wants to overwrite
                       the value via SetValue(); must be used for no other
                       purpose.
      @return          Returns true if an item with this key was found in the
                       hash, otherwise false.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Find(
        uint64_t key, uint64_t *value_out,
        uint64_t **key_value_location = nullptr) const {
      const int64_t kept_key_mask = (uint64_t(1) << num_kept_key_bits_) - 1;

      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = (1 | ((key >> buckets_num_bitsm1_) ^ key))
          << num_implicit_key_bits_;
      uint64_t kept_key = key >> num_implicit_key_bits_;

      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (~old_elem == 0) {
          return false;
        } else if ((old_elem & kept_key_mask) == kept_key) {
          *value_out = old_elem >> num_kept_key_bits_;
          if (key_value_location)
            *key_value_location = data_ + cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] key  Required to be the same key that was provided to
                        Find(); it is an error otherwise.
          @param [in] value  Value to write; bits of higher order than
                       (num_value_bits = 64 - num_key_bits) may not be set.
                       It is also an error if ~((key << num_value_bits) | value) == 0,
                       i.e. if all the allowed bits of both `key` and `value` are
                       set; but this is not checked.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ void SetValue(
        uint64_t *key_value_location, uint64_t key, uint64_t value) const {
      *key_value_location = (value << num_kept_key_bits_) |
          (key >> num_implicit_key_bits_);
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().  This overload does not require the user to specify the old key.
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] value  Value to write; bits of higher order than
                       (num_value_bits = 64 - num_key_bits) may not be set.
                       It is also an error if ~((key << num_value_bits) | value) == 0,
                       where `key` is the existing key-- i.e. if all the allowed bits
                       of both `key` and `value` are set; but this is not checked.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ uint64_t SetValue(
        uint64_t *key_value_location, uint64_t value) const {
      uint64_t old_pair = *key_value_location;
      K2_CHECK_NE(~old_pair, 0);  // Check it was not an empty location.
      const int64_t kept_key_mask = (uint64_t(1) << num_kept_key_bits_) - 1;
      uint64_t kept_key = old_pair & kept_key_mask;
      uint64_t new_pair = kept_key | (value << num_kept_key_bits_);
      *key_value_location = new_pair;
      const int64_t implicit_key_mask = (uint64_t(1) << num_implicit_key_bits_) - 1;
      uint64_t key = ((kept_key << num_implicit_key_bits_) |
              ((key_value_location - data_) & implicit_key_mask));
      return key;
    }

    /* Deletes a key from a hash.  Caution: this cannot be combined with other
       operations on a hash; after you delete a key you cannot do Insert() or
       Find() until you have deleted all keys.  This is an open-addressing hash
       table with no tombstones, which is why this limitation exists).

       @param [in] key   Key to be deleted.   Each key present in the hash must
                         be deleted  by exactly one thread, or it will loop
                         forever!

      Note: the const is with respect to the metadata only; required, to avoid
      compilation errors.
    */
    __forceinline__ __host__ __device__ void Delete(uint64_t key) const {
      uint32_t cur_bucket = key & num_buckets_mask_,
          bucket_inc = (1 | ((key >> buckets_num_bitsm1_) ^ key))
          << num_implicit_key_bits_;
      uint64_t kept_key = key >> num_implicit_key_bits_;
      const uint64_t kept_key_mask = (uint64_t(1) << num_kept_key_bits_) - 1;
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if ((old_elem & kept_key_mask) == kept_key) {
          data_[cur_bucket] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

   private:
    // A number satisfying 0 < num_key_bits_ < 64; the number of bits
    // (out of 64) used for the key.
    int32_t num_key_bits_;

    // This is equal to (num_key_bits_ - num_implicit_key_bits_);
    // it's the number of key bits that are stored in the hash
    // buckets.
    // It will satisfy 0 < num_kept_key_bits_ < num_key_bits_,
    // and num_kept_key_bits_ + num_value_bits. == 64.
    int32_t num_kept_key_bits_;

    // This is equal to (num_key_bits + num_value_bits - 64); it's the
    // number of key bits that are implicit in the bucket location
    // (because there are not enough bits to store them directly).
    // It will satisfy 0 < num_implicit_key_bits_ < num_key_bits_.
    int32_t num_implicit_key_bits_;

    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t buckets_num_bitsm1_;
    // pointer to data
    uint64_t *data_;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask_;
  };


  /*
    Return an Accessor object which can be used in kernel code (or on CPU if the
    context is a CPU context).  This is templated on the accessor type; you have
    to call it like:
       auto acc = hash.GetAccessor<Hash::Accessor<32>>();
    or:
       auto acc = hash.GetAccessor<Hash::GenericAccessor>();
    or:
       auto acc = hash.GetAccessor<Hash::PackedAccessor>();
  */
  template <typename AccessorT>
  AccessorT GetAccessor() {
    return AccessorT(*this);
  }


  // You should call this before the destructor is called if the hash will still
  // contain values when it is destroyed, to bypass a check.
  void Destroy() { data_ = Array1<uint64_t>(); }

  void CheckEmpty();

  // The destructor checks that the hash is empty, if we are in debug mode.
  // If you don't want this, call Destroy() before the destructor is called.
  ~Hash() {
#ifndef NDEBUG
    if (data_.Dim() != 0)
      CheckEmpty();
#endif
  }

 private:
  Array1<uint64_t> data_;

  // A number satisfying 0 < num_key_bits_ < 64; the number of bits
  // (out of 64) used for the key (rest are used for the value).
  // Keys are kept in the lower-order bits of the 64-bit hash elements.
  int32_t num_key_bits_;

  // num_value_bits_ + num_key_bits_ is always >= 64.  If it is greater
  // than 64 we need to use class PackedAccessor as the accessor object.
  int32_t num_value_bits_;


  // number satisfying data_.Dim() == 1 << (1+buckets_num_bitsm1_)
  int32_t buckets_num_bitsm1_;
};


/*
  How class Hash64 works:

    - It can function as a map from key=uint64_t to value=uint64_t, you must
      decide the number of buckets, when you create the hash, but you can resize
      it (manually).

   Note:
     Each bucket contains a pair of key/value, each 64bits, key is stored at
     data[2 * bucket_index] and value is stored at data[2 * bucket_index + 1].

   Some constraints:
    - You can store any (key,value) pair, except the pair where all the bits of
      both key and value are set [that is used to mean "nothing here"]
    - The number of buckets must always be a power of 2.
    - When deleting values from the hash you must delete them all at
      once (necessary because there is no concept of a "tombstone".

   Some notes on usage:

   You use it by: constructing it, obtaining its Accessor with GetAccessor();
   and inside kernels (or host code), calling functions Insert(), Find() or
   Delete() of the Accessor object.  Resizing is not automatic; it is the
   user's responsibility to make sure the hash does not get too full
   (which could cause assertion failures in kernels, and will be very slow).

   Some implementation notes:
    - When accessing hash[key], we use bucket_index == key % num_buckets,
      bucket_inc = 1 | (((key * 2) / num_buckets) ^ key).
    - If the bucket at `bucket_index` is occupied, we look in locations
      `(bucket_index + n * bucket_inc)%num_buckets` for n = 1, 2, ...;
      this choice ensures that if multiple keys hash to the same bucket,
      they don't all access the same sequence of locations; and bucket_inc
      being odd ensures we eventually try all locations (of course for
      reasonable hash occupancy levels, we shouldn't ever have to try
      more than two or three).

*/
class Hash64 {
 public:
  /* Constructor.  Context can be for CPU or GPU.

     @param [in] num_buckets   Number of buckets in the hash; must be
                a power of 2 and >= 128 (this limit was arbitrarily chosen).
                The number of items in the hash cannot exceed the number of
                buckets, or the code will loop infinitely when you try to add
                items; aim for less than 50% occupancy.
  */
  Hash64(ContextPtr c, int64_t num_buckets) {
    K2_CHECK_GE(num_buckets, 128);
    data_ = Array1<uint64_t>(c, num_buckets * 2, ~(uint64_t)0);
    int64_t n = 2;
    for (buckets_num_bitsm1_ = 0; n < num_buckets;
         n *= 2, buckets_num_bitsm1_++) {
    }
    K2_CHECK_EQ(num_buckets, 2 << buckets_num_bitsm1_)
        << " num_buckets must be a power of 2.";
  }

  // Only to be used prior to assignment.
  Hash64() = default;

  int64_t NumBuckets() const { return data_.Dim() / 2; }

  // Returns data pointer; for testing..
  uint64_t *Data() { return data_.Data(); }

  // Shallow copy
  Hash64 &operator=(const Hash64 &src) = default;
  // Copy constructor (shallow copy)
  explicit Hash64(const Hash64 &src) = default;

  ContextPtr &Context() const { return data_.Context(); }

  class Accessor {
   public:
    Accessor(Hash64 &hash)
        : data_(hash.data_.Data()),
          num_buckets_mask_(uint64_t(hash.NumBuckets()) - 1),
          buckets_num_bitsm1_(hash.buckets_num_bitsm1_) {}

    // Copy constructor
    Accessor(const Accessor &src) = default;

    /*
     Try to insert pair (key,value) into hash.
       @param [in] key  Key into hash, it is an error if ~key == 0, i.e. if all
                        the allowed bits of `key` are set.
       @param [in] value  Value to set, it is an error if ~value == 0, i.e. if
                          all the allowed bits `value` are set.
       @param [out] old_value  If not nullptr, this location will be set to
                     the existing value *if this key was already present* in the
                     hash (or set by another thread in this kernel), i.e. only
                     if this function returns false.
       @param [out] key_value_location  If not nullptr, its contents will be
                     set to the address of the (key,value) pair (either the
                     existing or newly-written one).
       @return  Returns true if this (key,value) pair was inserted, false
     otherwise.

       Note: the const is with respect to the metadata only; it is required, to
       avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value, uint64_t *old_value = nullptr,
        uint64_t **key_value_location = nullptr) const {
      uint64_t cur_bucket = key & num_buckets_mask_,
               bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);

      while (1) {
        uint64_t cur_key = data_[2 * cur_bucket];
        uint64_t cur_value = data_[2 * cur_bucket + 1];
        if (cur_key == key) {
          if (old_value) *old_value = cur_value;
          if (key_value_location) *key_value_location = data_ + 2 * cur_bucket;
          return false;  // key exists in hash
        } else if (~cur_key == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_key = AtomicCAS(
              (unsigned long long *)(data_ + 2 * cur_bucket), cur_key, key);
          if (old_key == cur_key) {
            // set value
            data_[2 * cur_bucket + 1] = value;
            if (key_value_location)
              *key_value_location = data_ + 2 * cur_bucket;
            return true;  // Successfully inserted.
          }
          if (old_key == key) {
            if (old_value) *old_value = cur_value;
            if (key_value_location)
              *key_value_location = data_ + 2 * cur_bucket;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because bucket_inc is odd (so only satisfies
        // (n * bucket_inc) % num_buckets == 0 for n == num_buckets).
        // Note: n here is the number of times we went around the loop.
        cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
      }
    }

    /*
     Look up this key in this hash; output the value and optionally the
     location of the (key,value) pair if found.

      @param [in] key    Key to look up;
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with key_value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] key_value_location  (optional) The memory address of the
                       (key,value) pair, in case the caller wants to overwrite
                       the value via SetValue(); must be used for no other
                       purpose.
      @return          Returns true if an item with this key was found in the
                       hash, otherwise false.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Find(
        uint64_t key, uint64_t *value_out,
        uint64_t **key_value_location = nullptr) const {
      uint64_t cur_bucket = key & num_buckets_mask_,
               bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      while (1) {
        uint64_t old_key = data_[2 * cur_bucket];
        uint64_t old_value = data_[2 * cur_bucket + 1];
        if (~old_key == 0) {
          return false;
        } else if (old_key == key) {
          while (~old_value == 0) old_value = data_[2 * cur_bucket + 1];
          *value_out = old_value;
          if (key_value_location) *key_value_location = data_ + 2 * cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

    /*
      Overwrite a value in a (key,value) pair whose location was obtained using
      Find().
          @param [in] key_value_location   Location that was obtained from
                         a successful call to Find().
          @param [in] value  Value to write;

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
     */
    __forceinline__ __host__ __device__ void SetValue(
        uint64_t *key_value_location, uint64_t value) const {
      *(key_value_location + 1) = value;
    }

    /* Deletes a key from a hash.  Caution: this cannot be combined with other
       operations on a hash; after you delete a key you cannot do Insert() or
       Find() until you have deleted all keys.  This is an open-addressing hash
       table with no tombstones, which is why this limitation exists).

       @param [in] key   Key to be deleted.   Each key present in the hash must
                         be deleted  by exactly one thread, or it will loop
                         forever!

      Note: the const is with respect to the metadata only; required, to avoid
      compilation errors.
    */
    __forceinline__ __host__ __device__ void Delete(uint64_t key) const {
      uint64_t cur_bucket = key & num_buckets_mask_,
               bucket_inc = 1 | ((key >> buckets_num_bitsm1_) ^ key);
      while (1) {
        uint64_t old_key = data_[2 * cur_bucket];
        if (old_key == key) {
          data_[2 * cur_bucket] = ~((uint64_t)0);
          data_[2 * cur_bucket + 1] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + bucket_inc) & num_buckets_mask_;
        }
      }
    }

   private:
    // pointer to data
    uint64_t *data_;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint64_t num_buckets_mask_;
    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint64_t buckets_num_bitsm1_;
  };

  /*
    Return an Accessor object which can be used in kernel code (or on CPU if the
    context is a CPU context).
  */
  Accessor GetAccessor() { return Accessor(*this); }

  // You should call this before the destructor is called if the hash will still
  // contain values when it is destroyed, to bypass a check.
  void Destroy() { data_ = Array1<uint64_t>(); }

  void CheckEmpty() const {
    if (data_.Dim() == 0) return;
    ContextPtr c = Context();
    Array1<int64_t> error(c, 1, -1);
    int64_t *error_data = error.Data();
    const uint64_t *hash_data = data_.Data();

    K2_EVAL(
        Context(), data_.Dim(), lambda_check_data, (int64_t i)->void {
          if (~(hash_data[i]) != 0) error_data[0] = i;
        });
    int64_t i = error[0];
    if (i >= 0) {  // there was an error; i is the index into the hash where
      // there was an element.
      int64_t elem = data_[i];
      // We don't know the number of bits the user was using for the key vs.
      // value, so print in hex, maybe they can figure it out.
      K2_LOG(FATAL) << "Destroying hash: still contains values: position " << i
                    << ", content = " << std::hex << elem;
    }
  }

  /* Resize the hash to a new number of buckets.

       @param [in] new_num_buckets   New number of buckets; must be a power of 2,
                  and must be large enough to accommodate all values in the hash
                  (we assume the caller is keeping track of the number of elements
                  in the hash somehow).

     CAUTION: Resizing will invalidate any accessor objects you have; you need
     to re-get the accessors before accessing the hash again.
  */
  void Resize(int64_t new_num_buckets, bool copy_data = true) {
    NVTX_RANGE(K2_FUNC);

    K2_CHECK_GT(new_num_buckets, 0);
    K2_CHECK_EQ(new_num_buckets & (new_num_buckets - 1), 0);  // power of 2.

    ContextPtr c = data_.Context();
    Hash64 new_hash(c, new_num_buckets);

    if (copy_data) {
      new_hash.CopyDataFromSimple(*this);
    }

    *this = new_hash;
    new_hash.Destroy();  // avoid failed check in destructor (it would otherwise
                       // expect the hash to be empty when destroyed).
  }

  /*
    Copies all data elements from `src` to `*this`.
   */
  void CopyDataFromSimple(Hash64 &src) {
    NVTX_RANGE(K2_FUNC);
    int64_t num_buckets = data_.Dim() / 2,
        src_num_buckets = src.data_.Dim() / 2;
    const uint64_t *src_data = src.data_.Data();
    uint64_t *data = data_.Data();
    uint64_t new_num_buckets_mask = static_cast<uint64_t>(num_buckets) - 1,
        new_buckets_num_bitsm1 = buckets_num_bitsm1_;
    ContextPtr c = data_.Context();
    K2_EVAL(c, src_num_buckets, lambda_copy_data, (uint64_t i) -> void {
        uint64_t key = src_data[2 * i];
        uint64_t value = src_data[2 * i + 1];
        if (~key == 0) return;  // equals -1.. nothing there.
        uint64_t bucket_inc = 1 | ((key >> new_buckets_num_bitsm1) ^ key);
        uint64_t cur_bucket = key & new_num_buckets_mask;
        while (1) {
          uint64_t assumed = ~((uint64_t)0),
              old_elem = AtomicCAS((unsigned long long*)(data + 2 * cur_bucket),
                                   assumed, key);
          if (old_elem == assumed) {
            *(data + 2 * cur_bucket + 1) = value;
            return;
          }
          cur_bucket = (cur_bucket + bucket_inc) & new_num_buckets_mask;
          // Keep iterating until we find a free spot in the new hash...
        }
      });
  }

  // The destructor checks that the hash is empty, if we are in debug mode.
  // If you don't want this, call Destroy() before the destructor is called.
  ~Hash64() {
#ifndef NDEBUG
    if (data_.Dim() != 0) CheckEmpty();
#endif
  }

 private:
  Array1<uint64_t> data_;

  // number satisfying data_.Dim() == 1 << (1+buckets_num_bitsm1_)
  uint64_t buckets_num_bitsm1_;
};

/*
  Returns the number of bits needed for an unsigned integer sufficient to
  store the nonnegative value `size`.

  Note: `size` might be the size of an array whose indexes we want to store in
  the hash, i.e. we'll need to store all value 0 <= n < size as possible keys.
  In this case, you'd never actually need to store the value `size` but we can't
  call HighestBitSet(size-1) because the hash code needs to keep "all-ones"
  reserved for "nothing in this hash bin", so if `size` is of the form 2^n we
  still need n+1 bits to store the indexes, because (2^n-1) is actually reserved
 */
inline int32_t NumBitsNeededFor(int64_t size) {
  return 1 + HighestBitSet(size);
}

}  // namespace k2

#endif  // K2_CSRC_HASH_H_
