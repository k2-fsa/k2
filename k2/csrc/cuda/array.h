// k2/csrc/cuda/array.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ARRAY_H_
#define K2_CSRC_CUDA_ARRAY_H_

#include "k2/csrc/cuda/context.h"

namespace k2 {

/*
  Array1* is a 1-dimensional contiguous array (that doesn't support a stride).
*/
template <typename T> class Array1 {
 public:
  int size() const { return size_; }  // dimension of only (1st) axis

  T *data();  // Returns pointer to 1st elem



  // generally Callable will be some kind of lambda or function object; it should be
  // possible to evaluate it on the CUDA device (if we're compiling with CUDA)
  // and also on the CPU.  We'll do src(i) to evaluate element i.
  // NOTE: we assume this thread is already set to use the device associated with the
  // context in 'ctx', if it's a CUDA context.
  template <typename Callable>
  Array1(ContextPtr ctx, int size, Callable &&callable) {
    Init(ctx, size);

    Eval(ctx->DeviceType(), data(), size, std::forward<Callable>(callable));
  }

  /* Return sub-part of this array
     @param [in] start  First element to cover, 0 <= start < size()
     @param [in] size   Number of elements to include, 0 < size < size()-start
  */
  Array1 Range(int start, int size);

  DeviceType Device() const { return region->device; }

  // Resizes, copying old contents if we could not re-use the same memory location.
  // It will always at least double the allocated size if it has to reallocate.
  // See Region::num_bytes vs. Region::bytes_used.
  void Resize(int new_size);

  ContextPtr &Context() { return region_->context; }

  Array1(const Array1 &other) = default;
 private:
  int size_;
  int byte_offset_;
  std::shared_ptr<Region> region_; // Region that `data` is a part of.  Device
                                   // type is stored here.  For an Array1 with
                                   // zero size (e.g. created using empty
                                   // constructor), will point to an empty
                                   // Region.


  void Init(DeviceType d, int size) {
    // .. takes care of allocation etc.
  }
};



#define MAX_DIM 4  // Will increase this as needed

class Shape {
 public:
  int32_t ndim() { return ndim_; }

  int32_t dim(int32_t i) {
    CHECK_LT(dynamic_cast<uint32_t>(i), dynamic_cast<uint32_t>(ndim_));
    return dims_[i];
  }
  int32_t stride(int32_t i) {
    CHECK_LT(dynamic_cast<uint32_t>(i), dynamic_cast<uint32_t>(ndim_));
    return strides_[i];
  }

  Shape(const std::vector<int32_t> &dims);
  Shape(const std::vector<int32_t> &dims,
        const std::vector<int32_t> strides);

  bool IsContiguous();

 private:
  int32_t ndim_;  // Must be >= 0
  int32_t dims_[MAX_DIM];
  int32_t strides_[MAX_DIM];
};



class Tensor {
 public:

  template <typename T> T *data();  // Returns pointer to elem with index
                                    // all-zeros... will check that the type
                                    // matches the correct one.

 private:


};





}  // namespace k2

#endif  // K2_CSRC_CUDA_ARRAY_H_
