// k2/csrc/cuda/tensor_ops.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Fangjun Kuang)

// See ../../LICENSE for clarification regarding multiple authors

#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {

template <typename T>
void CopyTensorElements2d(ContextPtr c, int32_t dim0, int32_t dim1,
                          const T *src_data, int32_t src_stride0,
                          int32_t src_stride1, T *dest_data,
                          int32_t dest_stride0, int32_t dest_stride1) {
  NVTX_RANGE(K2_FUNC);
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    // this is just an optimization, the other branch would work for CPU too.
    for (int32_t i = 0; i < dim0; i++) {
      for (int32_t j = 0; j < dim1; j++) {
        dest_data[i * dest_stride0 + j * dest_stride1] =
            src_data[i * src_stride0 + j * src_stride1];
      }
    }
  } else {
    K2_EVAL2(
        c, dim0, dim1, lambda_set_elems, (int32_t i, int32_t j)->void {
          dest_data[i * dest_stride0 + j * dest_stride1] =
              src_data[i * src_stride0 + j * src_stride1];
        });
  }
}

template <typename T>
void CopyTensorElements1d(ContextPtr c, int32_t dim, const T *src_data,
                          int32_t src_stride, T *dest_data,
                          int32_t dest_stride) {
  NVTX_RANGE(K2_FUNC);
  K2_EVAL(
      c, dim, lambda_set_elems, (int32_t i)->void {
        dest_data[i * dest_stride] = src_data[i * src_stride];
      });
}

// TODO(dan): this is far from ideal in terms of efficiency.  There is no
// attempt to discover the simplest pattern that covers the copy, or to be smart
// about memory loads if it turns out to be a transposition.
void CopyTensorElements(Tensor src, Tensor dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(src.SameDim(dest));
  ContextPtr c = GetContext(src, dest);
  int32_t num_axes = src.NumAxes();
  if (num_axes > 2) {
    // For now, only directly support copies of at most 2 dims.
    int32_t leading_dim = src.Dim(0);
    ParallelRunner pr(c);
    for (int32_t i = 0; i < leading_dim; i++) {
      With(pr.NewStream());
      Tensor src_part = src.Index(0, i), dest_part = dest.Index(0, i);
      CopyTensorElements(src_part, dest_part);
    }
  } else {
    const Shape &src_shape = src.GetShape(), &dest_shape = dest.GetShape();
    int32_t src_stride0 = (num_axes > 0 ? src_shape.Stride(0) : 0),
            dest_stride0 = (num_axes > 0 ? dest_shape.Stride(0) : 0),
            dim0 = (num_axes > 0 ? src_shape.Dim(0) : 1);
    Dtype dtype = src.GetDtype();
    K2_CHECK(dtype == dest.GetDtype());
    int32_t num_axes = src.NumAxes();
    if (num_axes == 2) {
      int32_t src_stride1 = src_shape.Stride(1),
              dest_stride1 = dest_shape.Stride(1), dim1 = src_shape.Dim(1);
      FOR_ALL_DTYPES(dtype, T,
                     CopyTensorElements2d<T>(
                         c, dim0, dim1, src.Data<T>(), src_stride0, src_stride1,
                         dest.Data<T>(), dest_stride0, dest_stride1));
    } else {
      FOR_ALL_DTYPES(
          dtype, T,
          CopyTensorElements1d<T>(c, dim0, src.Data<T>(), src_stride0,
                                  dest.Data<T>(), dest_stride0));
    }
  }
}

Tensor ToContiguous(const Tensor &src) {
  // things like this would be more efficient if we supported something like
  // PyTorch's ArrayRef.  not so critical to address that now though.
  NVTX_RANGE(K2_FUNC);
  Tensor ans(src.Context(), src.GetDtype(), src.GetShape().Dims());
  CopyTensorElements(src, ans);
  return ans;
}

template <typename T, typename U>
void CastTensorElements1dContiguous(ContextPtr c, int32_t dim,
                                    const T *src_data, U *dest_data) {
  NVTX_RANGE(K2_FUNC);
  K2_EVAL(
      c, dim, lambda_cast_elems,
      (int32_t i)->void { dest_data[i] = static_cast<U>(src_data[i]); });
}

Tensor Cast(Tensor src, Dtype new_dtype) {
  NVTX_RANGE(K2_FUNC);
  if (!src.IsContiguous()) src = ToContiguous(src);

  ContextPtr c = src.Context();
  Tensor ans(c, new_dtype, src.GetShape());
  K2_DCHECK(ans.IsContiguous());

  Dtype old_dtype = src.GetDtype();
  int32_t dim = ans.Nelement();

  FOR_ALL_DTYPES(old_dtype, T,
                 FOR_ALL_DTYPES(new_dtype, U,
                                CastTensorElements1dContiguous<T, U>(
                                    c, dim, src.Data<T>(), ans.Data<U>())));
  return ans;
}

// See the documentation of `Index`.
template <typename T>
static void Index1DImpl(ContextPtr context, const T *src_data,
                        int32_t src_stride, int32_t src_dim,
                        const int32_t *indexes_data, bool allow_minus_one,
                        int32_t ans_dim, T *ans_data) {
  if (allow_minus_one) {
    K2_EVAL(
        context, ans_dim, lambda_set_values, (int32_t i)->void {
          int32_t index = indexes_data[i];
          K2_DCHECK_LT(index, src_dim);
          K2_DCHECK(index >= 0 || index == -1);
          T value = (index < 0 ? T(0) : src_data[index * src_stride]);
          ans_data[i] = value;
        });
    return;
  }

  // now handle the case allow_minus_one == false
  K2_EVAL(
      context, ans_dim, lambda_set_values, (int32_t i)->void {
        int32_t index = indexes_data[i];
        K2_DCHECK_LT(index, src_dim);
        K2_DCHECK_GE(index, 0);
        ans_data[i] = src_data[index * src_stride];
      });
}

// See the documentation of `Index`.
template <typename T>
static void Index2DImpl(ContextPtr context, const T *src_data,
                        int32_t src_stride, int32_t src_dim0, int32_t src_dim1,
                        const int32_t *indexes_data, bool allow_minus_one,
                        int32_t ans_dim, int32_t ans_stride, T *ans_data) {
  if (allow_minus_one) {
    if (context->GetDeviceType() == kCpu) {
      for (int32_t i = 0; i != ans_dim; ++i) {
        int32_t index = indexes_data[i];
        K2_DCHECK_LT(index, src_dim0);
        K2_DCHECK_GE(index, -1);
        T *cur_ans_data = ans_data + i * ans_stride;
        const T *cur_src_data = src_data + index * src_stride;
        if (index == -1) {
          memset(cur_ans_data, 0, src_dim1 * sizeof(T));
        } else {
          memcpy(cur_ans_data, cur_src_data, src_dim1 * sizeof(T));
        }
      }
      return;
    }

    // now for CUDA
    auto lambda_set = [=] __device__(int32_t i, int32_t j) -> void {
      int32_t index = indexes_data[i];
      K2_DCHECK_LT(index, src_dim0);
      K2_DCHECK_GE(index, -1);
      T *cur_ans_data = ans_data + i * ans_stride;
      const T *cur_src_data = src_data + index * src_stride;
      if (index == -1)
        cur_ans_data[j] = 0;
      else
        cur_ans_data[j] = cur_src_data[j];
    };
    Eval2Device(context, ans_dim, src_dim1, lambda_set);
    return;
  }

  // now handle the case when allow_minus_one is false

  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != ans_dim; ++i) {
      int32_t index = indexes_data[i];
      K2_DCHECK_LT(index, src_dim0);
      K2_DCHECK_GE(index, 0);
      T *cur_ans_data = ans_data + i * ans_stride;
      const T *cur_src_data = src_data + index * src_stride;
      memcpy(cur_ans_data, cur_src_data, src_dim1 * sizeof(T));
    }
    return;
  }

  // now for CUDA
  auto lambda_set = [=] __device__(int32_t i, int32_t j) -> void {
    int32_t index = indexes_data[i];
    K2_DCHECK_LT(index, src_dim0);
    K2_DCHECK_GE(index, 0);
    T *cur_ans_data = ans_data + i * ans_stride;
    const T *cur_src_data = src_data + index * src_stride;
    cur_ans_data[j] = cur_src_data[j];
  };
  Eval2Device(context, ans_dim, src_dim1, lambda_set);
}

// See the documentation for `Index`.
// This function is for 1-D tensors.
static Tensor Index1D(Tensor &src, Array1<int32_t> &indexes,
                      bool allow_minus_one) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 1);
  K2_CHECK(IsCompatible(src, indexes));

  Dtype dtype = src.GetDtype();
  ContextPtr &context = src.Context();
  Tensor ans(context, dtype, {indexes.Dim()});
  K2_CHECK(ans.IsContiguous());

  int32_t src_stride = src.Stride(0);
  const int32_t *indexes_data = indexes.Data();
  int32_t src_dim = src.Dim(0);
  int32_t ans_dim = ans.Dim(0);
  FOR_ALL_DTYPES(
      dtype, T,
      Index1DImpl<T>(context, src.Data<T>(), src_stride, src_dim, indexes_data,
                     allow_minus_one, ans_dim, ans.Data<T>()));

  return ans;
}

// See the documentation for `Index`.
// This function is for 2-D tensors.
static Tensor Index2D(Tensor &src, Array1<int32_t> &indexes,
                      bool allow_minus_one) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 2);
  K2_CHECK(IsCompatible(src, indexes));

  Dtype dtype = src.GetDtype();
  ContextPtr &context = src.Context();
  Tensor ans(context, dtype, {indexes.Dim(), src.Dim(1)});
  K2_CHECK(ans.IsContiguous());

  int32_t src_stride = src.Stride(0);
  K2_CHECK_EQ(src.Stride(1), 1);

  const int32_t *indexes_data = indexes.Data();
  int32_t src_dim0 = src.Dim(0);
  int32_t src_dim1 = src.Dim(1);
  int32_t ans_dim = ans.Dim(0);
  int32_t ans_stride = ans.Stride(0);

  FOR_ALL_DTYPES(dtype, T,
                 Index2DImpl<T>(context, src.Data<T>(), src_stride, src_dim0,
                                src_dim1, indexes_data, allow_minus_one,
                                ans_dim, ans_stride, ans.Data<T>()));
  return ans;
}

Tensor Index(Tensor &src, Array1<int32_t> &indexes,
             bool allow_minus_one /*= true*/) {
  NVTX_RANGE(K2_FUNC);
  switch (src.NumAxes()) {
    case 1:
      return Index1D(src, indexes, allow_minus_one);
    case 2:
      return Index2D(src, indexes, allow_minus_one);
    default:
      K2_LOG(FATAL) << "Unsupported number of axes: " << src.NumAxes()
                    << "\n. Only 1-D and 2-D tensors are supported.";
      return src;  // prevent compiler warnings
  }
}

template <typename T>
static void IndexAdd1DImpl(ContextPtr context, const T *src_data,
                           int32_t src_dim, int32_t src_stride,
                           const int32_t *indexes_data, bool allow_minus_one,
                           int32_t dest_dim, int32_t dest_stride,
                           T *dest_data) {
  if (allow_minus_one) {
    K2_EVAL(
        context, src_dim, lambda_add, (int32_t i)->void {
          int32_t index = indexes_data[i];
          K2_DCHECK_LT(index, dest_dim);
          K2_DCHECK_GE(index, -1);

          if (index != -1)
            AtomicAdd(dest_data + index * dest_stride,
                      src_data[i * src_stride]);
        });
    return;
  }

  // handle the case: allow_minus_one == false
  K2_EVAL(
      context, src_dim, lambda_add, (int32_t i)->void {
        int32_t index = indexes_data[i];
        K2_DCHECK_LT(index, dest_dim);
        K2_DCHECK_GE(index, 0);
        AtomicAdd(dest_data + index * dest_stride, src_data[i * src_stride]);
      });
}

template <typename T>
static void IndexAdd2DImpl(ContextPtr context, const T *src_data,
                           int32_t src_dim0, int32_t src_dim1,
                           int32_t src_stride0, int32_t src_stride1,
                           const int32_t *indexes_data, bool allow_minus_one,
                           int32_t dest_dim, int32_t dest_stride0,
                           int32_t dest_stride1, T *dest_data) {
  if (allow_minus_one) {
    K2_EVAL2(
        context, src_dim0, src_dim1, lambda_add, (int32_t i, int32_t j)->void {
          int32_t index = indexes_data[i];
          K2_DCHECK_LT(index, dest_dim);
          K2_DCHECK_GE(index, -1);
          if (index != -1)
            AtomicAdd(dest_data + index * dest_stride0 + j * dest_stride1,
                      src_data[i * src_stride0 + j * src_stride1]);
        });
    return;
  }

  K2_EVAL2(
      context, src_dim0, src_dim1, lambda_add, (int32_t i, int32_t j)->void {
        int32_t index = indexes_data[i];
        K2_DCHECK_LT(index, dest_dim);
        K2_DCHECK_GE(index, 0);
        AtomicAdd(dest_data + index * dest_stride0 + j * dest_stride1,
                  src_data[i * src_stride0 + j * src_stride1]);
      });
}

static void IndexAdd1D(Tensor &src, Array1<int32_t> &indexes,
                       bool allow_minus_one, Tensor *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 1);
  K2_CHECK_NE(dest, nullptr);
  K2_CHECK_EQ(dest->NumAxes(), 1);
  ContextPtr context = GetContext(src, indexes, *dest);

  Dtype dtype = src.GetDtype();

  const int32_t *indexes_data = indexes.Data();

  int32_t src_dim = src.Dim(0);
  K2_CHECK_EQ(src_dim, indexes.Dim());
  int32_t src_stride = src.Stride(0);

  int32_t dest_dim = dest->Dim(0);
  int32_t dest_stride = dest->Stride(0);

  // atomiAdd is not available for some types, e.g., int8_t and int16_t
  // see
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
  FOR_REAL_AND_INT32_TYPES(
      dtype, T,
      IndexAdd1DImpl<T>(context, src.Data<T>(), src_dim, src_stride,
                        indexes_data, allow_minus_one, dest_dim, dest_stride,
                        dest->Data<T>()));
}

static void IndexAdd2D(Tensor &src, Array1<int32_t> &indexes,
                       bool allow_minus_one, Tensor *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 2);
  K2_CHECK_NE(dest, nullptr);
  K2_CHECK_EQ(dest->NumAxes(), 2);
  K2_CHECK_EQ(dest->Dim(1), src.Dim(1));

  ContextPtr context = GetContext(src, indexes, *dest);

  Dtype dtype = src.GetDtype();

  int32_t src_dim0 = src.Dim(0);
  int32_t src_dim1 = src.Dim(1);
  K2_CHECK_EQ(src_dim0, indexes.Dim());
  int32_t src_stride0 = src.Stride(0);
  int32_t src_stride1 = src.Stride(1);

  int32_t dest_dim = dest->Dim(0);
  int32_t dest_stride0 = dest->Stride(0);
  int32_t dest_stride1 = dest->Stride(1);

  const int32_t *indexes_data = indexes.Data();

  FOR_REAL_AND_INT32_TYPES(
      dtype, T,
      IndexAdd2DImpl<T>(context, src.Data<T>(), src_dim0, src_dim1, src_stride0,
                        src_stride1, indexes_data, allow_minus_one, dest_dim,
                        dest_stride0, dest_stride1, dest->Data<T>()));
}

void IndexAdd(Tensor &src, Array1<int32_t> &indexes, bool allow_minus_one,
              Tensor *dest) {
  NVTX_RANGE(K2_FUNC);
  switch (src.NumAxes()) {
    case 1:
      IndexAdd1D(src, indexes, allow_minus_one, dest);
      break;
    case 2:
      IndexAdd2D(src, indexes, allow_minus_one, dest);
      break;
    default:
      K2_LOG(FATAL) << "Unsupported number of axes: " << src.NumAxes()
                    << "\n. Only 1-D and 2-D tensors are supported.";
      break;
  }
}

}  // namespace k2
