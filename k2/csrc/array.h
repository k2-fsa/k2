/**
 * @brief
 * array
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_H_
#define K2_CSRC_ARRAY_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/eval.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor.h"

namespace k2 {

/*
  Array1 is a 1-dimensional contiguous array (that doesn't support a stride).
  T must be POD data type, e.g. basic type, struct.
*/
template <typename T>
class Array1 {
 public:
  static_assert(std::is_pod<T>::value, "T must be POD");
  using ValueType = T;
  size_t ElementSize() const { return sizeof(ValueType); }
  int32_t Dim() const { return dim_; }  // dimension of only axis (axis 0)

  // Returns pointer to 1st elem.  Could be a GPU or CPU pointer,
  // depending on the context.
  T *Data() {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(region_->data) +
                                 byte_offset_);
  }

  const T *Data() const {
    return reinterpret_cast<const T *>(reinterpret_cast<char *>(region_->data) +
                                       byte_offset_);
  }

  // Return a copy of this array that does not share the same underlying data.
  Array1<T> Clone() const;

  size_t ByteOffset() const { return byte_offset_; }

  // Called when creating Array2 using Array1, users should not call this for
  // now.
  const RegionPtr &GetRegion() const { return region_; }

  Array1(ContextPtr ctx, int32_t size) { Init(ctx, size); }

  // read in same format as operator<< and operator>>, i.e. "[ 10 20 30 ]"
  explicit Array1(const std::string &str);

  // Construct from context and string
  Array1(ContextPtr ctx, const std::string &str):
      Array1(Array1<T>(str).To(ctx)) { }

  // Creates an array that is not valid, e.g. you cannot call Context() on it.
  Array1() : dim_(0), byte_offset_(0), region_(nullptr) {}

  // Return if the array is valid or not. An array is valid if we can call
  // Context() on it.
  bool IsValid() const { return region_ != nullptr; }

  Array1(int32_t dim, RegionPtr region, size_t byte_offset)
      : dim_(dim), byte_offset_(byte_offset), region_(region) {}

  Array1(ContextPtr ctx, int32_t size, T elem) {
    Init(ctx, size);
    *this = elem;
  }

  /* Return sub-part of this array (shares the underlying data with this
     array).

     @param [in] start  First element to cover, 0 <= start <= Dim();
                        If start == Dim(), it just returns an empty array.
     @param [in] size   Number of elements to include, 0 <= size <= Dim()-start
  */
  Array1<T> Range(int32_t start, int32_t size) const {
    K2_CHECK_GE(start, 0);
    K2_CHECK_LE(start, Dim());
    K2_CHECK_GE(size, 0);
    K2_CHECK_LE(size, Dim() - start);
    return Array1(size, region_, byte_offset_ + start * ElementSize());
  }

  /* Return sub-part of this array (shares the underlying data with this
     array).  Like PyTorch's arange.  'Range' is deprecated.

     @param [in] start  First element to cover, 0 <= start <= Dim();
                        If start == Dim(), it just returns an empty array.
     @param [in] end    One-past-the-last element to cover,
                        start <= end <= Dim().
  */
  Array1<T> Arange(int32_t start, int32_t end) const {
    K2_CHECK_GE(start, 0);
    K2_CHECK_LE(start, dim_);
    K2_CHECK_GE(end, start);
    K2_CHECK_LE(end, dim_);
    return Array1(end - start, region_, byte_offset_ + start * ElementSize());
  }

  /*
   Return sub-part of this array, with a stride.
   Becomes a Tensor because Array1 does not support a stride that isn't 1.

     @param [in] start  First element of output, 0 <= start < Dim()
     @param [in] size   Number of elements to include, must satisfy
                        size > 0 and   0 <= (start + (size-1)*increment) <
                        Dim().
     @param [in] inc    Increment in original array each time index
                        increases.  Require inc > 0.
  */
  // TODO(haowen): does not support inc < 0 with below implementations, we may
  // not need a negative version, will revisit it later
  Tensor Range(int32_t start, int32_t size, int32_t inc) const {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GE(start, 0);
    K2_CHECK_GE(size, 0);
    K2_CHECK_GT(inc, 0);
    K2_CHECK_LE((size - 1) * inc, Dim() - start);
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {size};
    std::vector<int32_t> strides = {inc};
    Shape shape(dims, strides);
    return Tensor(type, shape, region_, byte_offset_ + start * ElementSize());
  }

  // Note that the returned Tensor is not const, the caller should be careful
  // when changing the tensor's data, it will also change data in the parent
  // array as they share the memory.
  Tensor ToTensor() const {
    NVTX_RANGE(K2_FUNC);
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {Dim()};
    Shape shape(dims);  // strides == 1
    return Tensor(type, shape, region_, byte_offset_);
  }

  DeviceType Device() const { return Context()->GetDeviceType(); }

  /*
    Convert to possibly-different context, may require CPU/GPU transfer.
    The returned value may share the same underlying `data` memory as *this.
    This should work even for tensors with dim == 0.

     If dim_ == 0 and region_ is NULL, this will return a direct copy of *this
    (i.e.
     with region_ also NULL)

     If dim == 0 and region_ is non-NULL, it will return a copy of *this with an
     empty region with the supplied context (if different from current region's
     context).
  */
  Array1 To(ContextPtr ctx) const {
    NVTX_RANGE(K2_FUNC);
    if (ctx->IsCompatible(*Context())) return *this;
    Array1 ans(ctx, Dim());
    ans.CopyFrom(*this);
    return ans;
  }

  // Copy from another array of the same dimension and type.
  void CopyFrom(const Array1<T> &src);

  // Convert this array to another array with type S;
  // if S is same with T, then it just returns *this
  template <typename S>
  Array1<typename std::enable_if<!std::is_same<S, T>::value, S>::type> AsType()
      const {
    NVTX_RANGE(K2_FUNC);
    // S != T
    Array1<S> ans(Context(), Dim());
    S *ans_data = ans.Data();
    const T *this_data = Data();
    K2_EVAL(
        Context(), Dim(), lambda_set_values,
        (int32_t i) { ans_data[i] = static_cast<S>(this_data[i]); });
    return ans;
  }
  template <typename S>
  Array1<typename std::enable_if<std::is_same<S, T>::value, S>::type> AsType()
      const {
    // S == T
    return *this;
  }

  /*
    Modify size of array, copying old contents if we could not re-use the same
    memory location. It will always at least double the allocated size if it has
    to reallocate. See Region::num_bytes vs. Region::bytes_used.  We only
    support the case that the current array *this (i.e. the array that will be
    resized) covers the highest used index in the region; this is to avoid
    overwriting memory shared by other arrays in the same region.

    Note: this may change which memory other arrays point to, if they share
    the same Region, but it will be transparent because arrays point to the
    Region and not to the data directly.

      @param [in] new_size     New size in elements to set the array to
      @param [in] reduce_bytes_used  If set to true,
                    will reduce the region->bytes_used to the size you are
                    setting the array to; may only be used if this is the
                    only array that points to that memory (e.g. if just
                    created with a larger size than needed).
  */
  void Resize(int32_t new_size, bool reduce_bytes_used = false) {
    NVTX_RANGE(K2_FUNC);
    if (new_size < dim_) {
      K2_CHECK_GE(new_size, 0);
    } else {
      std::size_t cur_bytes_used = byte_offset_ + sizeof(T) * dim_,
                  new_bytes_used = byte_offset_ + sizeof(T) * new_size;
      // the following avoids a situation where we overwrite data shared with
      // other Array objects.  You can just do *this = Array1<T>(...) and
      // overwrite *this with a new region if that's what you want.
      K2_CHECK_EQ(cur_bytes_used, region_->bytes_used);
      region_->Extend(new_bytes_used);
    }
    if (reduce_bytes_used) {
      std::size_t new_bytes_used = byte_offset_ + sizeof(T) * new_size;
      region_->bytes_used = new_bytes_used;
    }
    dim_ = new_size;
  }

  ContextPtr &Context() const { return region_->context; }

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
  T operator[](int32_t i) const {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GE(i, 0);
    K2_CHECK_LT(i, Dim());
    const T *data = Data() + i;
    DeviceType type = Context()->GetDeviceType();
    if (type == kCpu) {
      return *data;
    } else {
      K2_CHECK_EQ(type, kCuda);
      T ans;
      cudaError_t ret =
          cudaMemcpy(static_cast<void *>(&ans), static_cast<const void *>(data),
                     ElementSize(), cudaMemcpyDeviceToHost);
      K2_CHECK_CUDA_ERROR(ret);
      return ans;
    }
  }

  // return the last element on CPU of *this if dim >= 1,
  // will crash if *this is empty.
  T Back() const {
    K2_CHECK_GE(dim_, 1);
    return operator[](dim_ - 1);
  }

  /* Setting all elements to a scalar */
  void operator=(const T t) {
    NVTX_RANGE(K2_FUNC);
    T *data = Data();

    K2_EVAL(
        Context(), dim_, lambda_set_values, (int32_t i)->void { data[i] = t; });
  }

  /* Gathers elements in current array according to `indexes` and returns it,
     i.e. returned_array[i] = this_array[indexes[i]] for 0 <= i < indexes.Dim().
     Note 'indexes.Context()' must be compatible with the current Context(),
     i.e. `Context()->IsCompatible(indexes.Context())`.

     See also the function Index(), declared in array_ops.h, which is like this
     but also supports -1 as an index.
   */
  Array1 operator[](const Array1<int32_t> &indexes) const {
    NVTX_RANGE(K2_FUNC);
    const ContextPtr &c = GetContext(*this, indexes);
    int32_t ans_dim = indexes.Dim();
    Array1<T> ans(c, ans_dim);
    this->Index(indexes, &ans);
    return ans;
  }

  // Same with above operator[](const Array1<int32_t> &indexes), but with the
  // output array memory allocated by the caller.
  // Must have ans.Dim() == indexes.Dim() and IsCompatible(*ans, *this)
  void Index(const Array1<int32_t> &indexes, Array1<T> *ans) const {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_NE(ans, nullptr);
    const ContextPtr &c = GetContext(*this, indexes, *ans);
    int32_t ans_dim = indexes.Dim();
    K2_CHECK_EQ(ans_dim, indexes.Dim());
    const T *this_data = Data();
    T *ans_data = ans->Data();
    const int32_t *indexes_data = indexes.Data();

    K2_EVAL(
        c, ans_dim, lambda_copy_elems,
        (int32_t i)->void { ans_data[i] = this_data[indexes_data[i]]; });
  }

  // constructor from CPU array (transfers to GPU if necessary)
  Array1(ContextPtr ctx, const std::vector<T> &src) {
    NVTX_RANGE(K2_FUNC);
    Init(ctx, src.size());
    ContextPtr cpu_ctx = GetCpuContext();
    cpu_ctx->CopyDataTo(src.size() * ElementSize(), src.data(), ctx, Data());
  }

  // default constructor
  Array1(const Array1 &) = default;
  // move constructor
  Array1(Array1 &&) = default;
  // assignment operator (shallow); see Assign() for assignment of elements.
  Array1 &operator=(const Array1 &) = default;
  // move assignment operator (shallow)
  Array1 &operator=(Array1 &&) = default;

  /*
    This function checks that T is the same as the data-type of `tensor` and
    that `tensor` has zero or more axes.  If `tensor` is not contiguous it
    will make a contiguous copy.  Then it will construct this Array referring
    to the same data as the (possibly-copied) tensor.
   */
  explicit Array1(const Tensor &tensor) {
    NVTX_RANGE(K2_FUNC);
    Dtype type = DtypeOf<ValueType>::dtype;
    K2_CHECK_EQ(type, tensor.GetDtype());
    if (tensor.IsContiguous()) {
      dim_ = tensor.Nelement();
      byte_offset_ = tensor.ByteOffset();
      region_ = tensor.GetRegion();
      return;
    }

    *this = Array1(ToContiguous(tensor));
  }

 private:
  int32_t dim_;
  size_t byte_offset_;
  RegionPtr region_;  // Region that `data` is a part of.  Device type is stored
                      // here.  Will be NULL if Array1 was created with default
                      // constructor (invalid array!) but may still be non-NULL
                      // if dim_ == 0; this allows it to keep track of the
                      // context.

  void Init(ContextPtr context, int32_t size) {
    region_ = NewRegion(context, static_cast<size_t>(size) * ElementSize());
    dim_ = size;
    byte_offset_ = 0;
  }
};

// Could possibly introduce a debug mode to this that would do bounds checking.
template <typename T>
struct Array2Accessor {
  T *data;
  int32_t elem_stride0;
  __host__ __device__ T &operator()(int32_t i, int32_t j) const {
    return data[i * elem_stride0 + j];
  }

  T *Row(int32_t i) { return data + elem_stride0 * i; }
  Array2Accessor(T *data, int32_t elem_stride0)
      : data(data), elem_stride0(elem_stride0) {}
  __host__ __device__ Array2Accessor(const Array2Accessor &other)
      : data(other.data), elem_stride0(other.elem_stride0) {}
  Array2Accessor &operator=(const Array2Accessor &other) = default;
};

template <typename T>
struct ConstArray2Accessor {
  const T *data;
  int32_t elem_stride0;
  __host__ __device__ T operator()(int32_t i, int32_t j) const {
    return data[i * elem_stride0 + j];
  }
  const T *Row(int32_t i) const { return data + elem_stride0 * i; }
  ConstArray2Accessor(const T *data, int32_t elem_stride0)
      : data(data), elem_stride0(elem_stride0) {}
  ConstArray2Accessor(const ConstArray2Accessor &other) = default;
  ConstArray2Accessor &operator=(const ConstArray2Accessor &other) = default;
};

// NOTE: The following two forward declarations
// are used to prevent calling the non-template `ToContiguous()`
// in `Array2::To()`.
template <typename T>
class Array2;

template <typename T>
Array2<T> ToContiguous(const Array2<T> &src);

/*
  Array2 is a 2-dimensional array (== matrix), that is contiguous in the
  2nd dimension, i.e. a row-major matrix.
*/
template <typename T>
class Array2 {
 public:
  using ValueType = T;
  size_t ElementSize() const { return sizeof(ValueType); }

  /* Could view this as num_rows */
  int32_t Dim0() const { return dim0_; }

  /* Could view this as num_cols */
  int32_t Dim1() const { return dim1_; }

  // Currently ByteOffset and GetRegion is for internal usage, user should never
  // call it for now.
  size_t ByteOffset() const { return byte_offset_; }
  const RegionPtr &GetRegion() const { return region_; }

  ContextPtr &Context() const { return region_->context; }

  /*  stride on 0th axis, i.e. row stride, but this is stride in *elements*, so
      we name it 'ElemStride' to distinguish from stride in *bytes*.  This
      will satisfy ElemStride0() >= Dim1(). */
  int32_t ElemStride0() const { return elem_stride0_; }

  /*  Returns true if this array is contiguous (no gaps between the elements).
      Caution: it just checks elem_stride0_ == dim1_, which may not coincide
      with the "no-gaps" semantics if dim0_ <= 1. */
  bool IsContiguous() const { return elem_stride0_ == dim1_; }

  /*  returns a flat version of this, appending the rows; will copy the data if
      it was not contiguous. */
  Array1<T> Flatten() {
    NVTX_RANGE(K2_FUNC);
    if (dim1_ == elem_stride0_) {
      return Array1<T>(dim0_ * dim1_, region_, byte_offset_);
    } else {
      auto region = NewRegion(region_->context, static_cast<size_t>(dim0_) *
                                                    static_cast<size_t>(dim1_) *
                                                    ElementSize());
      Array1<T> array(dim0_ * dim1_, region, 0);
      const T *this_data = Data();
      T *data = array.Data();
      int32_t dim1 = dim1_;
      int32_t elem_stride0 = elem_stride0_;
      K2_EVAL2(
          region_->context, dim0_, dim1_, lambda_copy_elems,
          (int32_t i, int32_t j)->void {
            data[i * dim1 + j] = this_data[i * elem_stride0 + j];
          });
      return array;
    }
  }

  /*
    Returns an Array2 containing a subset of rows of *this, aliasing the
    same data.  (c.f. arange in PyTorch).  Calling it RowArange rather than
    RowRange to clarify that the 2nd arg is 'end' not 'dim'.

     @param [in] start  First row of output, 0 <= start < Dim0()
     @param [in] end    One-past-the-last row that *should not* be included.
     @param [in] inc    Increment in original array each time index
                        increases.  Require inc > 0.
  */
  Array2<T> RowArange(int32_t start, int32_t end, int32_t inc = 1) const {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GT(inc, 0);
    K2_CHECK_GE(start, 0);
    K2_CHECK_GE(end, start);
    K2_CHECK_LE(end, dim0_);
    int32_t num_rows = (end - start) / inc;
    return Array2<T>(num_rows, dim1_, elem_stride0_ * inc,
                     byte_offset_ + elem_stride0_ * static_cast<size_t>(start) *
                                        ElementSize(),
                     region_);
  }

  /*
    Returns an Array2 containing a subset of columns of *this, aliasing the
    same data.  (c.f. arange in PyTorch).  Calling it ColArange rather than
    ColRange to clarify that the 2nd arg is 'end' not 'dim'.

     @param [in] start  First column of output, 0 <= start < Dim1()
     @param [in] end    One-past-the-last column that *should not* be included.
  */
  Array2<T> ColArange(int32_t start, int32_t end) const {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GE(start, 0);
    K2_CHECK_GE(end, start);
    K2_CHECK_LE(end, dim1_);
    return Array2<T>(dim0_, end - start, elem_stride0_,
                     byte_offset_ + (start * ElementSize()), region_);
  }

  // return a row (indexing on the 0th axis)
  Array1<T> Row(int32_t i) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_GE(i, 0);
    K2_CHECK_LT(i, dim0_);
    int32_t byte_offset = byte_offset_ + i * elem_stride0_ * ElementSize();
    return Array1<T>(dim1_, region_, byte_offset);
  }

  // Instead of: Array2<T> operator[](const Array1<int32_t> &indexes)
  // see IndexRows(), declared in array_ops.h.

  // Creates an array that is not valid, e.g. you cannot call Context() on it.
  Array2()
      : dim0_(0),
        elem_stride0_(0),
        dim1_(0),
        byte_offset_(0),
        region_(nullptr) {}
  /* Create new array2 with given dimensions.  dim0 and dim1 must be >=0.
     Data will be uninitialized. */
  Array2(ContextPtr c, int32_t dim0, int32_t dim1)
      : dim0_(dim0), elem_stride0_(dim1), dim1_(dim1), byte_offset_(0) {
    K2_CHECK_GE(dim0, 0);
    K2_CHECK_GE(dim1, 0);
    region_ = NewRegion(c, static_cast<size_t>(dim0_) *
                               static_cast<size_t>(dim1_) * ElementSize());
  }

  // Create new array2 with given dimensions.  dim0 and dim1 must be >=0.
  // Data will be initialized with `elem`
  Array2(ContextPtr c, int32_t dim0, int32_t dim1, T elem)
      : dim0_(dim0), elem_stride0_(dim1), dim1_(dim1), byte_offset_(0) {
    K2_CHECK_GE(dim0, 0);
    K2_CHECK_GE(dim1, 0);
    region_ = NewRegion(c, static_cast<size_t>(dim0_) *
                               static_cast<size_t>(dim1_) * ElementSize());
    *this = elem;
  }

  explicit Array2(const std::string &str);
  // copy constructor
  Array2(const Array2 &other) = default;
  // move constructor
  Array2(Array2 &&other) = default;
  // assignment operator (shallow); see Assign() for assignment of elements.
  Array2 &operator=(const Array2 &other) = default;
  // move assignment operator (shallow);
  Array2 &operator=(Array2 &&other) = default;

  /* stride on 1st axis is 1 (in elements). */
  Array2(int32_t dim0, int32_t dim1, int32_t elem_stride0, int32_t byte_offset,
         RegionPtr region)
      : dim0_(dim0),
        elem_stride0_(elem_stride0),
        dim1_(dim1),
        byte_offset_(byte_offset),
        region_(region) {
    K2_CHECK_GE(dim0_, 0);
    K2_CHECK_GE(dim1_, 0);
    K2_CHECK_GE(elem_stride0_, dim1_);
  }

  // Setting all elements to a scalar
  void operator=(const T t) {
    NVTX_RANGE(K2_FUNC);
    T *data = Data();
    int32_t elem_stride0 = elem_stride0_;
    K2_EVAL2(
        Context(), dim0_, dim1_, lambda_set_elems,
        (int32_t i, int32_t j)->void { data[i * elem_stride0 + j] = t; });
  }

  /*
    Convert to possibly-different context, may require CPU/GPU transfer.
    The returned value may share the same underlying `data` memory as *this.
    This should work even for tensors with dim == 0.

    Note that the returned array is contiguous in case the required context
    is not compatible with the current context.
  */
  Array2<T> To(ContextPtr ctx) const {
    NVTX_RANGE(K2_FUNC);
    if (ctx->IsCompatible(*Context())) return *this;

    Array2<T> ans(ctx, dim0_, dim1_);

    if (elem_stride0_ == dim1_) {
      // the current array is contiguous, use memcpy
      T *dst = ans.Data();
      const T *src = Data();
      Context()->CopyDataTo(dim0_ * dim1_ * ElementSize(), src, ctx, dst);
      return ans;
    } else {
      return ToContiguous(*this).To(ctx);
    }
  }

  // Note that the returned Tensor is not const, the caller should be careful
  // when changing the tensor's data, it will also change data in the parent
  // array as they share the memory.
  Tensor ToTensor() {
    NVTX_RANGE(K2_FUNC);
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {dim0_, dim1_};
    std::vector<int32_t> strides = {elem_stride0_, 1};
    Shape shape(dims, strides);
    return Tensor(type, shape, region_, byte_offset_);
  }

  // Return one column of this Array2, as a Tensor.  (Will point to the
  // same data).
  Tensor Col(int32_t i) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_LT(static_cast<uint32_t>(i), static_cast<uint32_t>(dim1_));
    Dtype type = DtypeOf<ValueType>::dtype;
    std::vector<int32_t> dims = {dim0_};
    std::vector<int32_t> strides = {elem_stride0_};
    Shape shape(dims, strides);
    return Tensor(type, shape, region_, byte_offset_ + (ElementSize() * i));
  }

  // Note: const-ness is w.r.t. the metadata only.
  T *Data() const {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(region_->data) +
                                 byte_offset_);
  }

  // Note: array1 doesn't need an accessor because its Data() pointer functions
  // as one already.
  Array2Accessor<T> Accessor() {
    return Array2Accessor<T>(Data(), elem_stride0_);
  }

  ConstArray2Accessor<T> Accessor() const {
    return ConstArray2Accessor<T>(Data(), elem_stride0_);
  }

  /* Construct from Tensor.  Required to have 2 axes; will copy if the tensor
     did not have unit stride on 2nd axis
     @param [in] t                Input tensor, must have 2 axes and dtype == T
     @param [in] copy_for_strides Controls the behavior if the tensor did not
                                  have unit stride on 2nd axis. If true, will
                                  copy data in the tensor. If false, will die
                                  with error.

  */
  explicit Array2(Tensor &t, bool copy_for_strides = true) {
    NVTX_RANGE(K2_FUNC);
    auto type = t.GetDtype();
    K2_CHECK_EQ(type, DtypeOf<T>::dtype);
    const auto &shape = t.GetShape();
    K2_CHECK_EQ(shape.NumAxes(), 2);
    dim0_ = shape.Dim(0);
    dim1_ = shape.Dim(1);
    elem_stride0_ = shape.Stride(0);
    auto region = t.GetRegion();
    if (shape.Stride(1) == 1) {
      byte_offset_ = t.ByteOffset();
      region_ = region;
    } else {
      // TODO(haowen): only handle positive stride now
      if (!copy_for_strides) {
        K2_LOG(FATAL) << "non-unit stride on 2nd axis of tensor";
      }
      K2_CHECK_GT(elem_stride0_, 0);
      region_ =
          NewRegion(region->context, static_cast<size_t>(dim0_) *
                                         static_cast<size_t>(ElementSize()) *
                                         static_cast<size_t>(elem_stride0_));
      byte_offset_ = 0;
      CopyDataFromTensor(t);
    }
  }

  /* Initialize from Array1.  Require dim0 * dim1 == a.Dim() and dim0,dim1 >= 0
   */
  Array2(Array1<T> &a, int32_t dim0, int32_t dim1)
      : dim0_(dim0), elem_stride0_(dim1), dim1_(dim1) {
    K2_CHECK_GE(dim0, 0);
    K2_CHECK_GE(dim1, 0);
    K2_CHECK_EQ(dim0_ * dim1_, a.Dim());
    byte_offset_ = a.ByteOffset();
    region_ = a.GetRegion();
  }

  // Caution: user should never call this function, we declare
  // it as public just because the enclosing parent function for an
  // extended __host__ __device__ lambda
  // must have public access.
  void CopyDataFromTensor(const Tensor &t) {
    NVTX_RANGE(K2_FUNC);
    T *this_data = Data();
    const T *t_data = t.Data<T>();
    int32_t elem_stride0 = elem_stride0_;
    int32_t elem_stride1 = t.GetShape().Stride(1);
    K2_EVAL2(
        region_->context, dim0_, dim1_, lambda_copy_elems,
        (int32_t i, int32_t j)->void {
          this_data[i * elem_stride0 + j] =
              t_data[i * elem_stride0 + j * elem_stride1];
        });
  }

 private:
  int32_t dim0_;  // dimension on 0th (row) axis, i.e. the number of rows.
  int32_t elem_stride0_;  // stride *in elements* on 0th (row) axis, must be >=
                          // dim1_
  int32_t dim1_;          // dimension on column axis, i.e. the number of
                          // columns.

  size_t byte_offset_;  // byte offset within region_
  RegionPtr region_;    // Region that `data` is a part of.  Device
                        // type is stored here.  For an Array2 with
                        // zero size (e.g. created using empty
                        // constructor), will point to an empty
                        // Region.
};

inline int32_t ToPrintable(char c) { return static_cast<int32_t>(c); }

// clang-format off
inline
typename std::enable_if<!std::is_same<char, int8_t>::value, int32_t>::type
ToPrintable(int8_t c) {
  return static_cast<int32_t>(c);
}
// clang-format on

template <typename T>
T ToPrintable(T t) {
  return t;
}

// Print the contents of the array, as [ 1 2 3 ].  Intended mostly for
// use in debugging.
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Array1<T> &array);

// read an array (can read output as produced by "operator <<", assuming
// suitable operators exist for type T.  Will produce output on CPU.
template <typename T>
std::istream &operator>>(std::istream &stream, Array1<T> &array);

// Print the contents of the array, as "[[ 1 2 3 ]
// [ 4 5 6 ]]".  Intended mostly for use in debugging.
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Array2<T> &array);
// read an array (can read output as produced by "operator <<", assuming
// suitable operators exist for type T.  Will produce output on CPU.
template <typename T>
std::istream &operator>>(std::istream &stream, Array2<T> &array);

}  // namespace k2

#define IS_IN_K2_CSRC_ARRAY_H_
#include "k2/csrc/array_inl.h"
#undef IS_IN_K2_CSRC_ARRAY_H_

#endif  // K2_CSRC_ARRAY_H_
