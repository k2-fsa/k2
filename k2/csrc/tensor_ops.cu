// k2/csrc/cuda/tensor_ops.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {

template <typename T>
void CopyTensorElements2d(ContextPtr c, int32_t dim0, int32_t dim1,
                          const T *src_data, int32_t src_stride0,
                          int32_t src_stride1, T *dest_data,
                          int32_t dest_stride0, int32_t dest_stride1) {
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
    auto lambda_set_elems = [=] __host__ __device__(int32_t i,
                                                    int32_t j) -> void {
      dest_data[i * dest_stride0 + j * dest_stride1] =
          src_data[i * src_stride0 + j * src_stride1];
    };
    Eval2(c, dim0, dim1, lambda_set_elems);
  }
}

template <typename T>
void CopyTensorElements1d(ContextPtr c, int32_t dim, const T *src_data,
                          int32_t src_stride, T *dest_data,
                          int32_t dest_stride) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    // this is just an optimization, the other branch would work for CPU too.
    for (int32_t i = 0; i < dim; i++) {
      dest_data[i * dest_stride] = src_data[i * src_stride];
    }
  } else {
    auto lambda_set_elems = [=] __host__ __device__(int32_t i) -> void {
      dest_data[i * dest_stride] = src_data[i * src_stride];
    };
    Eval(c, dim, lambda_set_elems);
  }
}

// TODO(dan): this is far from ideal in terms of efficiency.  There is no
// attempt to discover the simplest pattern that covers the copy, or to be smart
// about memory loads if it turns out to be a transposition.
void CopyTensorElements(Tensor src, Tensor dest) {
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
  Tensor ans(src.Context(), src.GetDtype(), src.GetShape().Dims());
  CopyTensorElements(src, ans);
  return ans;
}

template <typename T, typename U>
void CastTensorElements1dContiguous(ContextPtr c, int32_t dim,
                                    const T *src_data, U *dest_data) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    // this is just an optimization, the other branch would work for CPU too.
    for (int32_t i = 0; i < dim; i++) {
      dest_data[i] = static_cast<U>(src_data[i]);
    }
  } else {
    auto lambda_cast_elems = [=] __host__ __device__(int32_t i) -> void {
      dest_data[i] = static_cast<U>(src_data[i]);
    };
    Eval(c, dim, lambda_cast_elems);
  }
}

Tensor Cast(Tensor src, Dtype new_dtype) {
  if (!src.IsContiguous()) src = ToContiguous(src);

  ContextPtr c = src.Context();
  Tensor ans(c, new_dtype, src.GetShape());
  K2_DCHECK(ans.IsContiguous());

  Dtype old_dtype = src.GetDtype();
  int32_t dim = ans.Nelement();

  FOR_ALL_DTYPES(old_dtype, T,
                 FOR_ALL_DTYPES(new_dtype, U,
                                (CastTensorElements1dContiguous<T, U>(
                                    c, dim, src.Data<T>(), ans.Data<U>()))));
  return ans;
}

}  // namespace k2
