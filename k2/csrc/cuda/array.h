// k2/csrc/cuda/array.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ARRAY_H_
#define K2_CSRC_CUDA_ARRAY_H_

#include "k2/csrc/cuda/context.h"

namespace k2 {

/*
  Array1 is a 1-dimensional contiguous array (that doesn't support a stride).
*/
template <typename T> class Array1 {
 public:
  int32_t Size() const { return size_; }  // dimension of only (1st) axis

  T *Data();  // Returns pointer to 1st elem



  // generally Callable will be some kind of lambda or function object; it should be
  // possible to evaluate it on the CUDA device (if we're compiling with CUDA)
  // and also on the CPU.  We'll do src(i) to evaluate element i.
  // NOTE: we assume this thread is already set to use the device associated with the
  // context in 'ctx', if it's a CUDA context.
  template <typename Callable>
  Array1(ContextPtr ctx, int32_t size, Callable &&callable) {
    Init(ctx, size);

    Eval(ctx->DeviceType(), data(), size, std::forward<Callable>(callable));
  }

  /* Return sub-part of this array
     @param [in] start  First element to cover, 0 <= start < size()
     @param [in] size   Number of elements to include, 0 < size < size()-start
  */
  Array1 Range(int32_t start, int32_t size);

  DeviceType Device() const { return region->device; }

  // Resizes, copying old contents if we could not re-use the same memory location.
  // It will always at least double the allocated size if it has to reallocate.
  // See Region::num_bytes vs. Region::bytes_used.
  void Resize(int32_t new_size);

  ContextPtr &Context() { return region_->context; }

  Array1(const Array1 &other) = default;
 private:
  int32_t size_;
  int32_t byte_offset_;
  RegionPtr region_;               // Region that `data` is a part of.  Device
                                   // type is stored here.  For an Array1 with
                                   // zero size (e.g. created using empty
                                   // constructor), will point to an empty
                                   // Region.


  void Init(DeviceType d, int32_t size) {
    // .. takes care of allocation etc.
  }
};


/*
  Array2 is a 2-dimensional array (== matrix), that is contiguous in the
  2nd dimension, i.e. a row-major marix.
*/
template <typename T> class Array2 {
  /* Could view this as num_rows */
  int32_t Dim0() { return dim0_;  }

  /* Could view this as num_cols */
  int32_t Dim1() { return dim1_;  }

  /*  stride on 0th axis, i.e. row stride, but this is stride in *elements*, so
      we name it 'ElemStride' to distinguish from stride in *bytes* */
  int32_t ElemStride0() { return step0_;   }


  Array1<T> operator [] (int32_t i);  // return a row (indexing on the 0th axis)


  /* Create new array2 with given dimensions.  dim0 and dim1 must be >0. */
  Array2(ContextPtr c, int32_t dim0, int32_t dim1);

  /* stride on 1st axis is 1 (in elements). */
  Array2(int32_t dim0,  int32_t dim1, int32_t elem_stride0,
         int32_t bytes_offset, RegionPtr region);

  TensorPtr AsTensor();

  T *Data();

  /* Construct from Tensor.  Required to have 2 axes; will copy if the tensor
     did not have stride on 2nd axis == sizeof(T)
     @param [in] t                Input tensor, must have 2 axes and dtype == T

     @param [in] copy_for_strides

  */
  Array2(Tensor<T> &t, bool copy_for_strides = true);

 private:
  int32_t dim0_;           // dim on 0th (row) axis
  int32_t elem_stride0_;   // stride *in elements* on 0th (row) axis, must be >= dim1_
  int32_t dim1_;           // dimension on column axis

  int32_t byte_offset_;   // byte offset within region_
  RegionPtr region_;               // Region that `data` is a part of.  Device
                                   // type is stored here.  For an Array2 with
                                   // zero size (e.g. created using empty
                                   // constructor), will point to an empty
                                   // Region.



};




}  // namespace k2

#endif  // K2_CSRC_CUDA_ARRAY_H_
