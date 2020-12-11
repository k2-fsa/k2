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
  How Hash32 works:
    - Each bucket has a struct { uint32_t key; int32_t value; }
    - If there is no value present, we set the key to ~0 == -1.
    - num_buckets is a power of 2.
    - When accessing hash[key], we use bucket_index == key % num_buckets,
        leftover_index = 1 + key / num_buckets.  (This is the leftover part
        of the index, shifted to be greater than 0).

    - If the bucket at `bucket_index` is occupied, we look in locations
      `bucket_index + n * leftover_index` for n = 1, 2, ....;  this choice
      ensures that if multiple keys hash to the same bucket, they don't
      all access the same sequence of locations.

    - The method of writing to the hash is:
         bool FindOrAdd(uint32_t, uint32_t value);
      which returns true if it added the element (else it found an existing
      element, but we don't guarantee that the
*/



class Hash32 {
 public:

  Hash32(ContextPtr c, int32_t num_buckets): data_(c, num_buckets, ~(uint64_t)0),
                                             buckets_num_bitsm1_(0) {
    K2_CHECK_GT(num_buckets, 64);
    int32_t n = 2;
    for (; n < num_buckets; n *= 2, buckets_num_bitsm1_++);
    K2_CHECK_EQ(num_buckets, 2 << buckets_num_bitsm1_)
        << " num_buckets must be a power of 2.";
  }


  // Shallow copy
  Hash32 &operator=(const Hash32 &src) = default;
  // Copy constructor (shallow copy)
  Hash32(const Hash32 &src) = default;


  ContextPtr &Context() { return data_.Context(); }

  union Element {
    uint64_t i;
    struct {
      uint32_t key;   // note: if ~key == 0, i.e. key == UINT32_MAX, it means
      // there is no entry here.
      int32_t value;  // value stored.  can be any int32_t.
    } p;
  };


  struct Accessor {
    Accessor(uint64_t *data,
             uint32_t num_buckets_mask,
             int32_t bucket_num_bitsm1) :
        data(data), num_buckets_mask(num_buckets_mask),
        bucket_num_bitsm1(bucket_num_bitsm1) { }

    // Copy constructor
    Accessor(const Accessor &src) = default;

    // pointer to data (it really contains Element)
    uint64_t *data;
    // num_buckets_mask is num_buckets (i.e. size of `data_` array) minus one;
    // num_buckets is a power of 2 so this can be used as a mask to get a number
    // modulo num_buckets.
    uint32_t num_buckets_mask;
    // A number satisfying num_buckets == 1 << (1+bucket_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t bucket_num_bitsm1;
  };

  Accessor GetAccessor() { return Accessor(data_.Data(),
                                           uint32_t(data_.Dim()) - 1,
                                           buckets_num_bitsm1_); }

  /*
    Try to insert pair (key,value) into hash.
    @param [in] acc  Accessor object, on device
    @param [in] key  Key into hash; may be any value except UINT32_MAX (not checked!)
    @param [in] value  Value to set; may be any value
    @param [out] old_value  If not nullptr, this location will be set to
    the existing value *if this key was already present*
    in the hash (or set by another thread in this kernel),
    i.e. only if this function returns false.
    @return  Returns true if this (key,value) pair was inserted, false otherwise.
  */
  static __forceinline__ __host__ __device__ bool Insert(
      Accessor acc, uint32_t key, int32_t value,
      int32_t *old_value = nullptr) {
    uint32_t cur_bucket = key & acc.num_buckets_mask,
         leftover_index = 1 | (key >> acc.bucket_num_bitsm1);

    Element new_elem;
    new_elem.p.key = key;
    new_elem.p.value = value;

    while (1) {
      Element old_elem;
      old_elem.i = acc.data[cur_bucket];
      if (old_elem.p.key == key) return false;  // key exists in hash
      else if (~old_elem.p.key == 0) {
        // we have a version of AtomicCAS that also works on host.
        uint64_t old_i = AtomicCAS((unsigned long long*)(acc.data + cur_bucket),
                                   old_elem.i, new_elem.i);
        if (old_i == old_elem.i) return true;  // Successfully inserted.
        old_elem.i = old_i;
        if (old_elem.p.key == key) return false;  // Another thread inserted this
        // same key.
      }

      // Rotate bucket index until we find a free location.  This will
      // eventually visit all bucket indexes before it returns to the same
      // location, because leftover_index is odd (so only satisfies
      // (n * leftover_index) % num_buckets == 0 for n == num_buckets).
      cur_bucket = (cur_bucket + leftover_index) & acc.num_buckets_mask;
    }
  }


  /*
    Looks up this key in this hash; outputs value and memory location of the value if found.

      @param [in] acc    Accessor of hash
      @param [in] key    Key to look up, may have any value except UINT32_MAX
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] value_location  Memory address of the value corresponding to
                        this key, in case the caller wants to overwrite it.
       @return          Returns true if an item with this key was found in the
                    hash, otherwise false.
  */
  static __forceinline__ __host__ __device__ bool Find(
      Accessor acc, uint32_t key, int32_t *value_out,
      int32_t **value_location_out = nullptr) {
    uint32_t cur_bucket = key & acc.num_buckets_mask,
         leftover_index = 1 | (key >> acc.bucket_num_bitsm1);
    while (1) {
      Element old_elem;
      old_elem.i = acc.data[cur_bucket];
      if (old_elem.p.key == key) {
        *value_out = old_elem.p.value;
        if (value_location_out)
          *value_location_out   =
              &(reinterpret_cast<Element*>(acc.data+cur_bucket)->p.value);
        return true;
      } else if (~old_elem.p.key == 0) {
        return false;
      } else {
        cur_bucket = (cur_bucket + leftover_index) & acc.num_buckets_mask;
      }
    }
  }

  // Deletes a key from a hash.  Caution: this cannot be combined with other
  // operations on a hash; after you delete a key you cannot do Insert() or Find()
  // until you have deleted all keys; and each key must be deleted by exactly one
  // thread.  (This is an open-addressing hash table with no tombstones,
  // which is why this limitation exists).
  //
  // If you delete a key which is not there, or which another thread has already
  // deleted, this will loop forever.
  static __forceinline__ __host__ __device__ void Delete(Accessor acc, uint32_t key) {
    uint32_t cur_bucket = key & acc.num_buckets_mask,
         leftover_index = 1 | (key >> acc.bucket_num_bitsm1);
    while (1) {
      Element old_elem;
      old_elem.i = acc.data[cur_bucket];
      if (old_elem.p.key == key) {
        acc.data[cur_bucket] = ~((uint64_t)0);
        return;
      } else {
        cur_bucket = (cur_bucket + leftover_index) & acc.num_buckets_mask;
      }
    }
  }

  // not actually private because of CUDA compiler limitations
  // private:

  Array1<uint64_t> data_;
  // number satisfying data_.Dim() == 1 << (1+bucket_num_bitsm1_)
  int32_t buckets_num_bitsm1_;

};


}  // namespace k2

#endif  // K2_CSRC_HASH_H_
