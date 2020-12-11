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
    - It's a map from uint32_t to int32_t.  (All keys except
      ~0 == -1 == UINT32_MAX are allowed).
    - Each bucket has a struct { uint32_t key; int32_t value; }
    - If there is no value present, we set the key to ~0 == -1.
    - The number of buckets is a power of 2 provided by the user.
    - When accessing hash[key], we use bucket_index == key % num_buckets,
      leftover_index = 1 | ((key * 2) / num_buckets).  This is
      leftover part of the index times 2, plus 1.
    - If the bucket at `bucket_index` is occupied, we look in locations
      `bucket_index + n * leftover_index` for n = 1, 2, ....;  this choice
      ensures that if multiple keys hash to the same bucket, they don't
      all access the same sequence of locations; and leftover_index being
      odd ensures we eventually try all locations (of course for reasonable
      hash occupancy levels, we shouldn't ever have to try more than two
      or three).
    - When deleting values from the hash you must delete them all at
      once (necessary because there is no concept of a "tombstone".

  You use it by: constructing it, obtaining its Accessor with GetAccessor(), and
  inside kernels (or host code), calling functions Insert(), Find() or Delete()
  of the Accessor object.  There is no resizing; sizing it correctly is the
  caller's responsibility and if the hash gets full the code will just loop
  forever (of course it will get extremely slow before it reaches that point).
*/
class Hash32 {
 public:
  Hash32(ContextPtr c, int32_t num_buckets):
      data_(c, num_buckets, ~(uint64_t)0), buckets_num_bitsm1_(0) {
    K2_CHECK_GT(num_buckets, 64);
    int32_t n = 2;
    for (; n < num_buckets; n *= 2, buckets_num_bitsm1_++) { }
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

  class Accessor {
   public:
    Accessor(uint64_t *data,
             uint32_t num_buckets_mask,
             int32_t bucket_num_bitsm1) :
        data_(data), num_buckets_mask_(num_buckets_mask),
        bucket_num_bitsm1_(bucket_num_bitsm1) { }

    // Copy constructor
    Accessor(const Accessor &src) = default;

   /*
    Try to insert pair (key,value) into hash.
      @param [in] key  Key into hash; may be any value except UINT32_MAX (not checked!)
      @param [in] value  Value to set; may be any value
      @param [out] old_value  If not nullptr, this location will be set to
                    the existing value *if this key was already present* in the
                    hash (or set by another thread in this kernel), i.e. only if
                    this function returns false.

       @return  Returns true if this (key,value) pair was inserted, false otherwise.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
   */
    __forceinline__ __host__ __device__ bool Insert(
        uint32_t key, int32_t value,
        int32_t *old_value = nullptr) const {
      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> bucket_num_bitsm1_);
      Element new_elem;
      new_elem.p.key = key;
      new_elem.p.value = value;
      while (1) {
        Element old_elem;
        old_elem.i = data_[cur_bucket];
        if (old_elem.p.key == key) return false;  // key exists in hash
        else if (~old_elem.p.key == 0) {
          // we have a version of AtomicCAS that also works on host.
          uint64_t old_i = AtomicCAS((unsigned long long*)(data_ + cur_bucket),
                                     old_elem.i, new_elem.i);
          if (old_i == old_elem.i) return true;  // Successfully inserted.
          old_elem.i = old_i;
          if (old_elem.p.key == key) return false;  // Another thread inserted
                                                    // this key
        }

        // Rotate bucket index until we find a free location.  This will
        // eventually visit all bucket indexes before it returns to the same
        // location, because leftover_index is odd (so only satisfies
        // (n * leftover_index) % num_buckets == 0 for n == num_buckets).
        cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
      }
    }

    /*
    Looks up this key in this hash; outputs value and memory location of the
    value if found.

      @param [in] key    Key to look up, may have any value except UINT32_MAX
      @param [out] value_out  If found, value will be written to here.  This may
                        seem redundant with value_location, but this should
                        compile to a local variable, and we want to avoid
                        redundant memory reads.
      @param [out] value_location  Memory address of the value corresponding to
                        this key, in case the caller wants to overwrite it.
      @return          Returns true if an item with this key was found in the
                       hash, otherwise false.

      Note: the const is with respect to the metadata only; it is required, to
      avoid compilation errors.
    */
    __forceinline__ __host__ __device__ bool Find(
        uint32_t key, int32_t *value_out,
        int32_t **value_location_out = nullptr) const {
      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> bucket_num_bitsm1_);
      while (1) {
        Element old_elem;
        old_elem.i = data_[cur_bucket];
        if (old_elem.p.key == key) {
          *value_out = old_elem.p.value;
          if (value_location_out)
            *value_location_out   =
                &(reinterpret_cast<Element*>(data_+cur_bucket)->p.value);
          return true;
        } else if (~old_elem.p.key == 0) {
          return false;
        } else {
          cur_bucket = (cur_bucket + leftover_index) & num_buckets_mask_;
        }
      }
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
    __forceinline__ __host__ __device__ void Delete(uint32_t key) const {
      uint32_t cur_bucket = key & num_buckets_mask_,
           leftover_index = 1 | (key >> bucket_num_bitsm1_);
      while (1) {
        Element old_elem;
        old_elem.i = data_[cur_bucket];
        if (old_elem.p.key == key) {
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
    // A number satisfying num_buckets == 1 << (1+bucket_num_bitsm1_)
    // the number of bits in `num_buckets` minus one.
    uint32_t bucket_num_bitsm1_;
  };

  Accessor GetAccessor() { return Accessor(data_.Data(),
                                           uint32_t(data_.Dim()) - 1,
                                           buckets_num_bitsm1_); }


  void Destroy() { data_ = Array1<uint64_t>(); }

  void CheckNonempty() {
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
      Element elem;
      elem.i = data_[i];
      K2_LOG(FATAL) << "Destroying hash: still contains values: position "
                    << i << ", key = " << elem.p.key
                    << ", value = " << elem.p.value;
    }
  }

  // The destructor checks that the hash is empty, if we are in debug mode.
  // If you don't want this, call Destroy() before the destructor is called.
  ~Hash32() {
#ifndef NDEBUG
    if (data_.Dim() != 0)
      CheckNonempty();
#endif
  }

 private:
  Array1<uint64_t> data_;
  // number satisfying data_.Dim() == 1 << (1+bucket_num_bitsm1_)
  int32_t buckets_num_bitsm1_;
};

}  // namespace k2

#endif  // K2_CSRC_HASH_H_
