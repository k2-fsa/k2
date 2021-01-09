/**
 * @brief
 * hash
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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
  *address = val;
  return compare;
#endif
}

/*
  How class Hash works:

    - It can function as a map from key=uint32_t to value=uint32_t, or from
      key=uint64_t to value=uint64_t where you choose NUM_KEY_BITS and
      `key` must have only up to NUM_KEY_BITS set and `value` must have
      only up to (64-NUM_KEY_BITS) set.  You decide NUM_KEY_BITS when
      you call Hash::Accessor<NUM_KEY_BITS>()
    - You can store any (key,value) pair except the pair where all the bits of
      both and key and value are set [that is used to mean "nothing here"]
    - The number of buckets is a power of 2 provided by the user to the constructor;
      currently no resizing is supported.
    - When accessing hash[key], we use bucket_index == key % num_buckets,
      leftover_index = 1 | ((key * 2) / num_buckets).  This is
      leftover part of the index times 2, plus 1.
    - If the bucket at `bucket_index` is occupied, we look in locations
      `(bucket_index + n * leftover_index)%num_buckets` for n = 1, 2, ...;
      this choice ensures that if multiple keys hash to the same bucket,
      they don't all access the same sequence of locations; and leftover_index
      being odd ensures we eventually try all locations (of course for
      reasonable hash occupancy levels, we shouldn't ever have to try
      more than two or three).
    - When deleting values from the hash you must delete them all at
      once (necessary because there is no concept of a "tombstone".

  You use it by: constructing it, obtaining its Accessor with
  GetAccessor<NUM_KEY_BITS>(), and inside kernels (or host code), calling
  functions Insert(), Find() or Delete() of the Accessor object.  There is no
  resizing; sizing it correctly is the caller's responsibility and if the hash
  gets full the code will just loop forever (of course it will get extremely
  slow before it reaches that point).
*/
class Hash {
 public:
  /* Constructor.  Context can be for CPU or GPU.  num_buckets must be a power of 2
     with num_buckets >= 128 (an arbitrarily chosen cutoff) */
  Hash(ContextPtr c, int32_t num_buckets) {
    std::ostringstream os;
    os << K2_FUNC << ":num_buckets=" << num_buckets;
    NVTX_RANGE(os.str().c_str());
    data_ = Array1<uint64_t>(c, num_buckets, ~(uint64_t)0);
    K2_CHECK_GE(num_buckets, 128);
    int32_t n = 2;
    for (buckets_num_bitsm1_ = 0; n < num_buckets;
         n *= 2, buckets_num_bitsm1_++) { }
    K2_CHECK_EQ(num_buckets, 2 << buckets_num_bitsm1_)
        << " num_buckets must be a power of 2.";
  }

  // Only to be used prior to assignment.
  Hash() = default;


  int32_t NumBuckets() const { return data_.Dim(); }

  /* Resize the hash to a new number of buckets.

       @param [in] new_num_buckets   New number of buckets; must be a power of 2,
                  and must be large enough to accommodate all values in the hash
                  (we assume the caller is keeping track of the number of elements
                  in the hash somehow).
       @param [in] num_key_bits  Number of bits used to store the keys,
                  with 0 < num_key_bits < 64 (number of bits in the values
                  is 64 minus this).  This must be the same as was
                  used to add any values that are currently in the hash.

     CAUTION: Resizing will invalidate any accessor objects you have; you need
     to re-get the accessors before accessing the hash again.
  */
  void Resize(int32_t new_num_buckets, int32_t num_key_bits) {
    K2_CHECK_GT(new_num_buckets, 0);
    K2_CHECK_EQ(new_num_buckets & (new_num_buckets - 1), 0);  // power of 2.

    ContextPtr c = data_.Context();
    Hash new_hash(c, new_num_buckets);

    int32_t dim = data_.Dim(),
        num_value_bits = 64 - num_key_bits;
    const uint64_t *this_data = data_.Data();
    uint64_t *new_data = new_hash.data_.Data();
    size_t new_num_buckets_mask = static_cast<size_t>(new_num_buckets) - 1,
        new_buckets_num_bitsm1 = new_hash.buckets_num_bitsm1_;

    K2_EVAL(c, dim, lambda_copy_data, (int32_t i) -> void {
        uint64_t key_value = this_data[i];
        if (~key_value == 0) return;  // equals -1.. nothing there.
        uint64_t key = key_value >> num_value_bits,
            leftover_index = 1 | (key >> new_buckets_num_bitsm1);
        size_t cur_bucket = key & new_num_buckets_mask;
        while (1) {
          uint64_t assumed = ~((uint64_t)0),
              old_elem = AtomicCAS((unsigned long long*)(new_data + cur_bucket),
                                   assumed, key_value);
          if (old_elem == assumed) return;
          cur_bucket = (cur_bucket + leftover_index) & new_num_buckets_mask;
          // Keep iterating until we find a free spot in the new hash...
        }
      });
    *this = new_hash;
    new_hash.Destroy();  // avoid failed check in destructor.
  }

  // Shallow copy
  Hash &operator=(const Hash &src) = default;
  // Copy constructor (shallow copy)
  Hash(const Hash &src) = default;

  ContextPtr &Context() const { return data_.Context(); }


  // Note: this is the templated version of class Accessor, usable for any
  // 0 < NUM_KEY_BITS < 64.  We also have a version called GenericAccessor
  // where the number of bits in the key is provided at run-time.
  // We may decide at some point to have a specific version for where the
  // number of bits is 32.
  template <int32_t NUM_KEY_BITS> class Accessor {
   public:
    // constructor of Accessor is for use by class Hash, not by the user.
    Accessor(uint64_t *data,
             uint32_t num_buckets_mask,
             int32_t buckets_num_bitsm1) :
        data_(data), num_buckets_mask_(num_buckets_mask),
        buckets_num_bitsm1_(buckets_num_bitsm1) { }

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

       @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value,
        uint64_t *old_value = nullptr) const {
      uint32_t cur_bucket = static_cast<uint32_t>(key) & num_buckets_mask_,
          leftover_index = 1 | (key >> buckets_num_bitsm1_);
      constexpr int32_t NUM_VALUE_BITS = 64 - NUM_KEY_BITS;
      constexpr int64_t VALUE_MASK = (uint64_t(1)<<NUM_VALUE_BITS)-1;

      K2_DCHECK_EQ(key & ~((uint64_t(1)<<NUM_KEY_BITS)-1), 0);
      K2_DCHECK_EQ(value & ~VALUE_MASK, 0);

      uint64_t new_elem = (key << (64 - NUM_KEY_BITS)) | value;
      while (1) {
        uint64_t cur_elem = data_[cur_bucket];
        if ((cur_elem >> NUM_VALUE_BITS) == key) {
          if (old_value) *old_value = cur_elem & VALUE_MASK;
          return false;  // key exists in hash
        }
        else if (~cur_elem == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_elem = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                        cur_elem, new_elem);
          if (old_elem == cur_elem) return true;  // Successfully inserted.
          cur_elem = old_elem;
          if (cur_elem >> NUM_VALUE_BITS == key) {
            if (old_value) *old_value = cur_elem & VALUE_MASK;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because leftover_index is odd (so only satisfies
        // (n * leftover_index) % num_buckets == 0 for n == num_buckets).
        // Note: n here is the number of times we went around the loop.
        cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
      }
    }

    /*
    Looks up this key in this hash; outputs value and optionally the
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
      constexpr int32_t NUM_VALUE_BITS = 64 - NUM_KEY_BITS;
      constexpr int64_t VALUE_MASK = (uint64_t(1)<<NUM_VALUE_BITS)-1;

      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> buckets_num_bitsm1_);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (~old_elem == 0) {
          return false;
        } else if ((old_elem >> NUM_VALUE_BITS) == key) {
          *value_out = old_elem & VALUE_MASK;
          if (key_value_location)
            *key_value_location = data_ + cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
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
      constexpr int32_t NUM_VALUE_BITS = 64 - NUM_KEY_BITS;
      *key_value_location = (key << NUM_VALUE_BITS) | value;
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
      constexpr int32_t NUM_VALUE_BITS = 64 - NUM_KEY_BITS;
      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> buckets_num_bitsm1_);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (old_elem >> NUM_VALUE_BITS == key) {
          data_[cur_bucket] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
        }
      }
    }

   private:
    // pointer to data (it really contains struct Element)
    uint64_t *data_;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask_;
    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t buckets_num_bitsm1_;
  };


  class GenericAccessor {
   public:
    // constructor of GenericAccessor is for use by class Hash, not by the user.
    GenericAccessor(uint32_t num_key_bits,
                    uint32_t buckets_num_bitsm1,
                    uint64_t *data,
                    uint32_t num_buckets_mask):
        num_value_bits_(64 - num_key_bits),
        buckets_num_bitsm1_(buckets_num_bitsm1),
        data_(data), num_buckets_mask_(num_buckets_mask) { }

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

       @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint64_t key, uint64_t value,
        uint64_t *old_value = nullptr) const {
      uint32_t cur_bucket = static_cast<uint32_t>(key) & num_buckets_mask_,
           leftover_index = 1 | (key >> buckets_num_bitsm1_);
      const uint32_t num_value_bits = num_value_bits_,
          num_key_bits = 64 - num_value_bits;
      const int64_t VALUE_MASK = (uint64_t(1)<<num_value_bits)-1;

      K2_DCHECK_EQ(key & ~((uint64_t(1)<<num_key_bits)-1), 0);
      K2_DCHECK_EQ(value & ~VALUE_MASK, 0);

      uint64_t new_elem = (key << (64 - num_key_bits)) | value;
      while (1) {
        uint64_t cur_elem = data_[cur_bucket];
        if ((cur_elem >> num_value_bits) == key) {
          if (old_value) *old_value = cur_elem & VALUE_MASK;
          return false;  // key exists in hash
        }
        else if (~cur_elem == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_elem = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                        cur_elem, new_elem);
          if (old_elem == cur_elem) return true;  // Successfully inserted.
          cur_elem = old_elem;
          if (cur_elem >> num_value_bits == key) {
            if (old_value) *old_value = cur_elem & VALUE_MASK;
            return false;  // Another thread inserted this key
          }
        }
        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because leftover_index is odd (so only satisfies
        // (n * leftover_index) % num_buckets == 0 for n == num_buckets).
        // Note: n here is the number of times we went around the loop.
        cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
      }
    }

    /*
    Looks up this key in this hash; outputs value and optionally the
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
      const uint32_t num_value_bits = num_value_bits_;
      const int64_t VALUE_MASK = (uint64_t(1)<<num_value_bits)-1;

      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> buckets_num_bitsm1_);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (~old_elem == 0) {
          return false;
        } else if ((old_elem >> num_value_bits) == key) {
          *value_out = old_elem & VALUE_MASK;
          if (key_value_location)
            *key_value_location = data_ + cur_bucket;
          return true;
        } else {
          cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
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
      *key_value_location = (key << num_value_bits_) | value;
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
           leftover_index = 1 | (key >> buckets_num_bitsm1_);
      while (1) {
        uint64_t old_elem = data_[cur_bucket];
        if (old_elem >> num_value_bits_ == key) {
          data_[cur_bucket] = ~((uint64_t)0);
          return;
        } else {
          cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
        }
      }
    }

   private:
    // A number satisfying 0 < num_value_bits_ < 64; the number of bits
    // (out of 64) used for the value (rest are used for the key).
    uint32_t num_value_bits_;
    // A number satisfying num_buckets == 1 << (1+buckets_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t buckets_num_bitsm1_;
    // pointer to data (it really contains struct Element)
    uint64_t *data_;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask_;
  };




  /*
    Return an Accessor object which can be used in kernel code (or on CPU if the
    context is a CPU context).

    Template argument `NUM_KEY_BITS` may be any number in [1,63] but probably
    will be something like 32 or 40; the number of bits in the key will be 64
    minus that.  The user must be consistent in the choice of num-key-bits
    (except it can be changed if needed when the hash is empty).

    See also non-templated function GetGenericAccessor().
   */
  template <int32_t NUM_KEY_BITS>
  Accessor<NUM_KEY_BITS> GetAccessor() {
    return Accessor<NUM_KEY_BITS>(data_.Data(),
                                  uint32_t(data_.Dim()) - 1,
                                  buckets_num_bitsm1_);
  }

  /*
    Version of accessor object where the number of bits in the key is decided
    at run-time rather than compile time.
   */
  GenericAccessor GetGenericAccessor(int32_t num_key_bits) {
    return GenericAccessor(num_key_bits,
                           buckets_num_bitsm1_,
                           data_.Data(),
                           uint32_t(data_.Dim()) - 1);
  }


  // You should call this before the destructor is called if the hash will still
  // contain values when it is destroyed, to bypass a check.
  void Destroy() { data_ = Array1<uint64_t>(); }

  void CheckEmpty() {
    if (data_.Dim() == 0) return;
    ContextPtr c = Context();
    Array1<int32_t> error(c, 1, -1);
    int32_t *error_data = error.Data();
    uint64_t *hash_data = data_.Data();

    K2_EVAL(Context(), data_.Dim(), lambda_check_data, (int32_t i) -> void {
        if (~(hash_data[i]) != 0) error_data[0] = i;
      });
    int32_t i = error[0];
    if (i >= 0) { // there was an error; i is the index into the hash where
                  // there was an element.
      int64_t elem = data_[i];
      // We don't know the number of bits the user was using for the key vs.
      // value, so print in hex, maybe they can figure it out.
      K2_LOG(FATAL) << "Destroying hash: still contains values: position "
                    << i << ", key,value = " << std::hex << elem;
    }
  }

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
  // number satisfying data_.Dim() == 1 << (1+buckets_num_bitsm1_)
  int32_t buckets_num_bitsm1_;
};

}  // namespace k2

#endif  // K2_CSRC_HASH_H_
