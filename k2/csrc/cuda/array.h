// k2/csrc/cuda/array.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ARRAY_H_
#define K2_CSRC_CUDA_ARRAY_H_

#include <algorithm>

#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/debug.h"
#include "k2/csrc/cuda/dtype.h"
#include "k2/csrc/cuda/tensor.h"

namespace k2 {

/*
  Array1 is a 1-dimensional contiguous array (that doesn't support a stride).
  T must be POD data type, e.g. basic type, struct.
*/
template <typename T>
class Array1 {
 public:
  using ValueType = T;
  int32_t ElementSize() const { return sizeof(ValueType); }
  int32_t Dim() const { return dim_; }  // dimension of only axis (axis 0)

  // Returns pointer to 1st elem.  Could be a GPU or CPU pointer,
  // depending on the context.
  T *Data() {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(region_->data) +
                                 byte_offset_);
  }

  T *Data() const {
    return reinterpret_cast<const T *>(reinterpret_cast<char *>(region_->data) +
                                       byte_offset_);
  }

  // generally Callable will be some kind of lambda or function object; it
  // should be possible to evaluate it on the CUDA device (if we're compiling
  // with CUDA) and also on the CPU.  We'll do src(i) to evaluate element i.
  // NOTE: we assume this thread is already set to use the device associated
  // with the context in 'ctx', if it's a CUDA context.
  // TODO(Haowen): no corresponding test code now, we may delete this later
  template <typename Callable>
  Array1(ContextPtr ctx, int32_t size, Callable &&callable) {
    Init(ctx, size);

    // TODO(haowen): there's no such definition
    // `Eval(ContextPtr, T*, int, Callable&)` now
    Eval(ctx, Data(), size, std::forward<Callable>(callable));
  }

  Array1(ContextPtr ctx, int32_t size) { Init(ctx, size); }

  // Creates an array that is not valid, e.g. you cannot call Context() on it.
  // TODO(haowen): why do we need this version?
  Array1() : dim_(0), byte_offset_(0), region_(nullptr) {}

  Array1(ContextPtr ctx, int32_t size, T elem) {
    Init(ctx, size);
    T *data = Data();
    // TODO(haowen): need capture *this, not sure it's valid in constructor
    /*
    auto lambda = [=] __host__ __device__(int32_t i) -> void {
      data[i] = elem;
    };
    Eval(ctx, dim_, lambda);
    */
  }

  /* Return sub-part of this array. Note that the returned Array1 is not const,
     the caller should be careful when changing array's data, it will
     also change data in the parent array as they shares the memory.
     in
     @param [in] start  First element to cover, 0 <= start < Dim()
     @param [in] size   Number of elements to include, 0 < size <= Dim()-start
  */
  Array1 Range(int32_t start, int32_t size) {
    K2_CHECK_GE(start, 0);
    K2_CHECK_LT(start, Dim());
    K2_CHECK_GT(size, 0);
    K2_CHECK_LE(size, Dim() - start);
    return Array1(size, region_, byte_offset_ + start * ElementSize());
  }

  /*
   Return sub-part of this array, with a stride (note: increment may be negative
   but not zero).  Becomes a Tensor because Array1 does not support a stride
   that isn't 1.

     @param [in] start  First element of output, 0 <= start < Dim()
     @param [in] size   Number of elements to include, must satisfy
                        size > 0 and   0 <= (start + (size-1)*increment) <
                        Dim().
     @param [in] inc    Increment in original array each time index
                        increases
  */
  // TODO(haowen): does not support inc < 0 with below implementations, we may
  // don't need negative version, will revisit it later
  Tensor Range(int32_t start, int32_t size, int32_t inc) {
    K2_CHECK_GE(start, 0);
    K2_CHECK_LT(start, Dim());
    K2_CHECK_GT(size, 0);
    K2_CHECK_GT(inc, 0);
    k2_CHECK_LT((size - 1) * inc, Dim() - start);
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {size};
    std::vector<int32_t> strides = {inc};
    Shape shape(dims, strides);
    return Tensor(type, shape, region_, byte_offset_ + start * ElementSize());
  }

  // Note that the returned Tensor is not const, the caller should be careful
  // when changing the tensor's data, it will also change data in the parent
  // array as they shares the memory.
  Tensor ToTensor() {
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {Dim()};
    Shape shape(dims);  // strides == 1
    return Tensor(type, shape, region_, byte_offset_);
  }

  DeviceType Device() const { return Context()->GetDeviceType(); }

  // Resizes, copying old contents if we could not re-use the same memory
  // location. It will always at least double the allocated size if it has to
  // reallocate. See Region::num_bytes vs. Region::bytes_used.
  // TODO(haowen): now we only support the case that the current array `curr`
  // (i.e. the array that will be resized) covers the highest used index in
  // the region, that is, for any array `a` uses this region,
  // curr.byte_offset_ + curr.Dim() * curr.ElementSize() == region_->bytes_used
  // >= a.byte_offset + a.Dim() * a.ElementSize()
  void Resize(int32_t new_size) {
    K2_CHECK_EQ(byte_offset_ + Dim() * ElementSize(), region_->bytes_used);
    if (new_size <= Dim()) {
      return;
    }
    if (byte_offset_ + new_size * ElementSize() > region_->num_bytes) {
      // always double the allocated size
      auto tmp = NewRegion(Context(), 2 * region_->num_bytes);
      // copy data
      auto kind = GetMemoryCopyKind(*Context(), *Context());
      MemoryCopy(tmp->data, region_->data, region_->bytes_used, kind);
      // update meta info
      dim_ = new_size;
      tmp->bytes_used = byte_offset_ + new_size * ElementSize();
      std::swap(tmp, region_);
    }
  }

  ContextPtr &Context() { return region_->context; }

  const ContextPtr &Context() const { return region_->context; }

  // Sets the context on this object (Caution: this is not something you'll
  // often need).  'ctx' must be compatible with the current Context(),
  // i.e. `ctx->IsCompatible(*Context())`, and is expected to be a parent of
  // the current context.  It's for use when you create an object with a
  // child context for speed (i.e. to use a different cuda stream) and
  // want to return it with the parent context to keep things simple.
  // (In general we don't expect functions to output things with newly
  // created contexts attached).
  void SetContext(const ContextPtr &ctx) {
    ContextPtr &c = Context();
    K2_CHECK(c->IsCompatible(*ctx));
    c = ctx;
  }

  /* Indexing operator (note: for now, we make all indexes be int32_t).  Returns
     a T on the CPU.  This is fast if this is a CPU array, but could take some
     time if it's a CUDA array, so use this operator sparingly.  If you know
     this is a CPU array, it would have much less overhead to index the Data()
     pointer. */
  T operator[](int32_t i) { return Data()[i]; }

  /* Setting all elements to a scalar */
  void operator=(const T t) {
    T *data = Data();
    auto lambda_set_values = [=] __host__ __device__(int32_t i)->void {
      data[i] = t;
    };
    Eval(Context(), dim_, lambda_set_values);
  }

  /* Gathers elements in current array according to `indexes` and returns it,
     i.e. returned_array[i] = this_array[indexes[i]] for 0 <= i < indexes.Dim().
     Note 'indexes.Context()' must be compatible with the current Context(),
     i.e. `Context()->IsCompatible(indexes.Context())`.
   */
  Array1 operator[](const Array1<int32_t> &indexes) {
    const ContextPtr &c = Context();
    K2_CHECK(c->IsCompatible(*(indexes.Context())));
    int32_t ans_dim = indexes.Dim();
    Array1<T> ans(c, ans_dim);
    const T *this_data = Data();
    T *ans_data = ans.Data();
    int32_t *indexes_data = indexes.Data();
    auto lambda_copy_elems = [=] __host__ __device__(int32_t i)->void {
      ans_data[i] = this_data[indexes_data[i]];
    };
    Eval(c, ans_dim, lambda_copy_elems);
    return ans;
  }

  // constructor from CPU array (transfers to GPU if necessary)
  Array1(ContextPtr ctx, const std::vector<T> &src) {
    Init(ctx, src.size());
    T *data = Data();
    DeviceType d = ctx->GetDeviceType();
    if (d == kCpu) {
      std::copy(src.begin(), src.end(), data);
    } else {
      K2_CHECK_EQ(d, kCuda);
      cudaMemcpy(static_cast<void *>(data), static_cast<void *>(src.data()),
                 src.size() * ElementSize(), cudaMemcpyHostToDevice);
    }
  }

  Array1(const Array1 &other) = default;

 private:
  int32_t dim_;
  int32_t byte_offset_;
  RegionPtr region_;  // Region that `data` is a part of.  Device type is stored
                      // here.  Will be NULL if Array1 was created with default
                      // constructor (invalid array!) but may still be non-NULL
                      // if dim_ == 0; this allows it to keep track of the
                      // context.
  Array1(int32_t dim, RegionPtr region, int32_t byte_offset)
      : dim_(dim), region_(region), byte_offset_(byte_offset) {}

  void Init(ContextPtr context, int32_t size) {
    region_ = NewRegion(context, size * ElementSize());
    dim_ = size;
    byte_offset_ = 0;
  }
};

// Could possibly introduce a debug mode to this that would do bounds checking.
template <typename T>
struct Array2Accessor {
  T *data;
  int32_t elem_stride0;
  __host__ __device__ T &operator()(int32_t i, int32_t j) {
    return data[i * elem_stride0 + j];
  }

  T *Row(int32_t i) { return data + elem_stride0 * i; }
  Array2Accessor(T *data, int32_t elem_stride0)
      : data(data), elem_stride0(elem_stride0) {}
  Array2Accessor(const Array2Accessor &other) = default;
};

template <typename T>
struct ConstArray2Accessor {
  const T *data;
  int32_t elem_stride0;
  __host__ __device__ T operator()(int32_t i, int32_t j) {
    return data[i * elem_stride0 + j];
  }
  ConstArray2Accessor(const T *data, int32_t elem_stride0)
      : data(data), elem_stride0(elem_stride0) {}
  ConstArray2Accessor(const ConstArray2Accessor &other) = default;
};

/*
  Array2 is a 2-dimensional array (== matrix), that is contiguous in the
  2nd dimension, i.e. a row-major marix.
*/
template <typename T>
class Array2 {
 public:
  /* Could view this as num_rows */
  int32_t Dim0() const { return dim0_; }

  /* Could view this as num_cols */
  int32_t Dim1() const { return dim1_; }

  ContextPtr &Context() const { return region_->context; }

  /*  stride on 0th axis, i.e. row stride, but this is stride in *elements*, so
      we name it 'ElemStride' to distinguish from stride in *bytes*.  This
      will satisfy ElemStride0() >= Dim1(). */
  int32_t ElemStride0() { return elem_stride0_; }

  /*  returns a flat version of this, appending the rows; will copy the data if
      it was not contiguous. */
  Array1<T> Flatten();

  Array1<T> operator[](int32_t i);  // return a row (indexing on the 0th axis)

  /* Create new array2 with given dimensions.  dim0 and dim1 must be >0.
     Data will be uninitialized. */
  Array2(ContextPtr c, int32_t dim0, int32_t dim1);

  /* stride on 1st axis is 1 (in elements). */
  Array2(int32_t dim0, int32_t dim1, int32_t elem_stride0, int32_t byte_offset,
         RegionPtr region)
      : dim0_(dim0),
        dim1_(dim1),
        elem_stride0_(elem_stride0),
        byte_offset_(byte_offset),
        region_(region) {}

  TensorPtr AsTensor();

  const T *Data() const {
    return reinterpret_cast<const T *>(reinterpret_cast<char *>(region_->data) +
                                       byte_offset_);
  }

  T *Data() {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(region_->data) +
                                 byte_offset_);
  }

  // Note: array1 doesn't need an accessor because its Data() pointer functions
  // as one already.
  Array2Accessor<T> Accessor() {
    return Array2Accessor<T>(Data(), elem_stride0_);
  }

  ConstArray2Accessor<T> Accessor() const {
    return Array2Accessor<T>(Data(), elem_stride0_);
  }

  /* Construct from Tensor.  Required to have 2 axes; will copy if the tensor
     did not have stride on 2nd axis == sizeof(T)
     @param [in] t                Input tensor, must have 2 axes and dtype == T

     @param [in] copy_for_strides

  */
  Array2(Tensor &t, bool copy_for_strides = true);

  /* Initialize from Array1.  Require dim0 * dim1 == a.Size() and dim0,dim1 >= 0
   */
  Array2(Array1<T> &a, int32_t dim0, int32_t dim1);

 private:
  int32_t dim0_;          // dim on 0th (row) axis
  int32_t elem_stride0_;  // stride *in elements* on 0th (row) axis, must be >=
                          // dim1_
  int32_t dim1_;          // dimension on column axis

  int32_t byte_offset_;  // byte offset within region_
  RegionPtr region_;     // Region that `data` is a part of.  Device
                         // type is stored here.  For an Array2 with
                         // zero size (e.g. created using empty
                         // constructor), will point to an empty
                         // Region.
};

}  // namespace k2

#endif  // K2_CSRC_CUDA_ARRAY_H_
