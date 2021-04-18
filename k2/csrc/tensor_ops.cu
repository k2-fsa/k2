// k2/csrc/cuda/tensor_ops.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Fangjun Kuang,
//                                                   Haowen Qiu)

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
  K2_CHECK(src.SameDims(dest));
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
  int32_t dim = ans.NumElements();

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

template <typename T>
static void SimpleRaggedIndexSelect1DImpl(ContextPtr context, const T *src_data,
                                          int32_t src_stride, int32_t src_dim,
                                          Ragged<int32_t> &indexes,
                                          int32_t ans_dim, T *ans_data) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(indexes.NumAxes(), 2);
  int32_t indexes_dim0 = indexes.Dim0(),
          indexes_num_elems = indexes.NumElements();
  const int32_t *indexes_row_ids_data = indexes.RowIds(1).Data();
  const int32_t *indexes_data = indexes.values.Data();
  K2_CHECK_EQ(ans_dim, indexes_dim0);

  K2_EVAL(
      context, ans_dim, lambda_init_ans,
      (int32_t i)->void { ans_data[i] = 0; });
  Array1<int32_t> non_zero_indexes(context, ans_dim, -1);
  int32_t *non_zero_indexes_data = non_zero_indexes.Data();
  K2_EVAL(
      context, indexes_num_elems, lambda_set_ans_data, (int32_t i)->void {
        int32_t src_index = indexes_data[i];
        K2_CHECK_GE(src_index, 0);
        K2_CHECK_LT(src_index, src_dim);
        T value = src_data[src_index * src_stride];
        int32_t ans_index = indexes_row_ids_data[i];
        if (value != 0) {
          non_zero_indexes_data[ans_index] = i;
          ans_data[ans_index] = value;
        }
      });

  // check if there is at most one non-zero element in src for each sub-list
  Array1<int32_t> status(context, 1, 0);  // 0 -> success; otherwise 1 + row_id
                                          // of bad row in `indexes`
  int32_t *status_data = status.Data();
  K2_EVAL(
      context, indexes_num_elems, lambda_check_status, (int32_t i)->void {
        int32_t src_index = indexes_data[i];
        T value = src_data[src_index * src_stride];
        int32_t ans_index = indexes_row_ids_data[i];
        if (value != 0 && non_zero_indexes_data[ans_index] != i)
          status_data[0] = 1 + ans_index;
      });
  int32_t s = status[0];
  if (s != 0) {
    Array1<T> indexed_values(context, indexes_num_elems);
    T *indexed_values_data = indexed_values.Data();
    K2_EVAL(
        context, indexes_num_elems, lambda_set_values, (int32_t i)->void {
          int32_t src_index = indexes_data[i];
          indexed_values_data[i] = src_data[src_index * src_stride];
        });
    Array1<int32_t> row_splits = indexes.RowSplits(1);
    K2_LOG(FATAL) << "There must be at most one non-zero "
                     "element in src for any sub-list in indexes; sub-list "
                  << (s - 1) << " has too many elements: "
                  << indexed_values.Arange(row_splits[s - 1], row_splits[s]);
  }
}

Tensor SimpleRaggedIndexSelect1D(Tensor &src, Ragged<int32_t> &indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 1);
  K2_CHECK(IsCompatible(src, indexes));

  Dtype dtype = src.GetDtype();
  ContextPtr &context = src.Context();
  Tensor ans(context, dtype, {indexes.Dim0()});
  K2_CHECK(ans.IsContiguous());

  int32_t src_stride = src.Stride(0);
  int32_t src_dim = src.Dim(0);
  int32_t ans_dim = ans.Dim(0);
  // Note below src.Data<T> will check if T is compatible with `dtype`.
  FOR_ALL_DTYPES(dtype, T,
                 SimpleRaggedIndexSelect1DImpl<T>(context, src.Data<T>(),
                                                  src_stride, src_dim, indexes,
                                                  ans_dim, ans.Data<T>()));
  return ans;
}

template <typename Real>
struct DiscountedCumSumElement {
  Real y;      // y is the partial sums of x values.  Initially it is just a
               // single x value.  In general each x is multiplied by all
               // previous gammas.
  Real gamma;  // gamma is the product of gammas along a range of elements
};
template <typename Real>
struct CombineCumSumOp {
  __device__ DiscountedCumSumElement<Real> operator() (
      DiscountedCumSumElement<Real> &a,
      DiscountedCumSumElement<Real> &b) const {
    return DiscountedCumSumElement<Real>{b.y +  b.gamma * a.y, a.gamma * b.gamma};
  }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename Real>
struct BlockPrefixCallbackOp
{
  using Elem = DiscountedCumSumElement<Real>;
  Elem running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(): running_total{0.0, 0.0} { }
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ Elem operator()(Elem block_aggregate)
    {
      Elem old_prefix = running_total;
      running_total = block_aggregate;
      return old_prefix;
    }
};

/*
  Notes for DiscountedCumSum.

    It implements a discounted sum along a sequence.  Suppose we have x_i, gamma_i and
  y_i, for 0 <= i < T.  Then we do:
      y_0 = x_0
      y_i = x_i + y_{i-1} gamma_i
   for 0 < i < T.  (This is done as a generic inclusive-scan/inclusive-sum with a special
   reduction op).

  See DiscountedCumSumElement and CombineCumSumOp for how we use a special operator to
  do this as an inclusive-sum.

  The tensors involved must be 2-dimensional with dimensions (N, T) where N is
  the batch size and T the time duration.

  Each thread-block is of (x,y,z) size (ThreadsPerBlock,1,1), and it processes N
  items.  It processes ThreadsPerBlock items at a time; and if T >
  ThreadsPerBlock it simply loops to cover the remaining items.

  The grid size (x,y,z) is (X,Y,1) where the X and Y together cover the "N"
  (batch) dimension.  (We can't cover it just in the X dimension because of
  limits on the size of each time).

    @param [in] N    The batch size, i.e. number of separate sequences.  We expect
               that N <= gridDim.x * gridDim.y.
    @param [in] T    The sequence length.  There is no constraint on the sequence
                     length; the kernel deals with ThreadsPerBlock items at a time,
                     and takes care of T > ThreadsPerBlock by looping.
    @param [in] x    Pointer to the x input data, which is an array of shape (N,T)
    @param [in] x_stride0  Stride along axis 0 of of the `x` data
    @param [in] gamma   Pointer to the gamma input data, which is an array of shape (N,T)
    @param [in] gamma_stride0  Stride along axis 0 of of the `gamma` data
    @param [in] y  Pointer to the y output data, which is an array of shape (N,T)
    @param [in] y_stride0  Stride along axis 0 of the `y` data
    @param [in] stride1  Stride along axis 1 of the three arrays (this is expected
                   to be identical, nonzero, and preferably -1 or 1.
*/
template <typename Real,
          int ThreadsPerBlock>
static __global__ void DiscountedCumSumKernel(int N, int T,
                                              const Real *x, int x_stride0,
                                              const Real *gamma, int gamma_stride0,
                                              Real *y, int y_stride0,
                                              int stride1) {
  int n_idx = blockIdx.y * gridDim.x + blockIdx.x;
  if (n_idx >= N)
    return;
  x += x_stride0 * n_idx;
  gamma += gamma_stride0 * n_idx;
  y += y_stride0 * n_idx;

  int thread_idx = threadIdx.x;
  using Elem = DiscountedCumSumElement<Real>;

  BlockPrefixCallbackOp<Real> prefix_callback;

  typedef cub::BlockScan<Elem, ThreadsPerBlock> BlockScan;
  // shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int base_t = 0; base_t < T; base_t += ThreadsPerBlock) {
    Elem elem;

    // Load x and gamma from memory.  These reads will be coalesced (which is
    // the advantage of having each thread process one element at this stage;
    // although we spend more time with raking reduction than we really
    // need to).
    if (base_t + thread_idx < T) {
      elem.y = x[(base_t + thread_idx) * stride1];
      elem.gamma = gamma[(base_t + thread_idx) * stride1];
    }
    CombineCumSumOp<Real> op;

    // the last arg is a callback functor that provides us the aggregate of this block
    // and which is expected to return the element that we want to add to
    BlockScan(temp_storage).InclusiveScan(elem, elem, op, prefix_callback);
    __syncthreads();

    if (base_t + thread_idx < T)
      y[(base_t + thread_idx) * stride1] = elem.y;
  }
}


template <typename Real, int ThreadsPerBlock>
void DiscountedCumSumCudaImpl(cudaStream_t stream,
                              int N, int T,
                              const Real *x, int x_stride0,
                              const Real *gamma, int gamma_stride0,
                              Real *y, int y_stride0, int stride1) {

  int32_t tot_grid_size = N;
  int32_t x_grid_size = (tot_grid_size < (1 << 20) ?
                         std::min<int32_t>(tot_grid_size, (1 << 10)) :
                         32768),
      y_grid_size = NumBlocks(tot_grid_size, x_grid_size);

  dim3 grid_dim(x_grid_size, y_grid_size, 1),
      block_dim(ThreadsPerBlock, 1, 1);
  K2_CUDA_SAFE_CALL(DiscountedCumSumKernel<Real, ThreadsPerBlock>
                    <<<grid_dim, block_dim, 0, stream>>>(N, T, x, x_stride0,
                                                         gamma, gamma_stride0,
                                                         y, y_stride0, stride1));
}


template <typename Real>
static void DiscountedCumSumCpuImpl(int N, int T,
                                    const Real *x, int x_stride0,
                                    const Real *gamma, int gamma_stride0,
                                    Real *y, int y_stride0,
                                    int stride1) {
  for (int32_t n = 0; n < N; n++,
           x += x_stride0, gamma += gamma_stride0, y += y_stride0) {
    Real cur_sum = 0.0;
    for (int32_t t = 0; t < T; t++) {
      cur_sum = x[t * stride1] + cur_sum * gamma[t * stride1];
      y[t * stride1] = cur_sum;
    }
  }
}


void DiscountedCumSum(const Tensor &src, const Tensor &gamma, Tensor *dest) {
  // check contexts compatible:
  if (!(IsCompatible(src, gamma) && IsCompatible(src, *dest))) {
    K2_LOG(FATAL) << "Tensors are on different devices";
  }
  if (!(src.NumAxes() == 2 && gamma.NumAxes() == 2 && dest->NumAxes() == 2)) {
    K2_LOG(FATAL) << "Expected all num-axes to equal 2.";
  }
  if (!(src.SameDims(gamma) && src.SameDims(*dest))) {
    K2_LOG(FATAL) << "Expected all args to have the same dim.";
  }
  if (!(src.Stride(1) == gamma.Stride(1) && src.Stride(1) == dest->Stride(1))) {
    K2_LOG(FATAL) << "Expected all strides on dim 1 to be the same.";
  }
  if (!(src.GetDtype() == gamma.GetDtype() && src.GetDtype() == dest->GetDtype())) {
    K2_LOG(FATAL) << "Expected all args to have the same dtype.";
  }
  int32_t N = src.Dim(0),
      T = src.Dim(1),
      src_stride0 = src.Stride(0),
      gamma_stride0 = gamma.Stride(0),
      dest_stride0 = dest->Stride(0),
      stride1 = src.Stride(1);  // these are all the same.
  ContextPtr c = src.Context();
  if (src.GetDtype() == kFloatDtype) {
    if (c->GetDeviceType() == kCuda) {
      DiscountedCumSumCudaImpl<float, 128>(c->GetCudaStream(), N, T,
                                           src.Data<float>(), src_stride0,
                                           gamma.Data<float>(), gamma_stride0,
                                           dest->Data<float>(), dest_stride0,
                                           stride1);
    } else {
      DiscountedCumSumCpuImpl<float>(N, T,
                                     src.Data<float>(), src_stride0,
                                     gamma.Data<float>(), gamma_stride0,
                                     dest->Data<float>(), dest_stride0,
                                     stride1);

    }
  } else if (src.GetDtype() == kDoubleDtype) {
    if (c->GetDeviceType() == kCuda) {
      DiscountedCumSumCudaImpl<double, 128>(c->GetCudaStream(), N, T,
                                            src.Data<double>(), src_stride0,
                                            gamma.Data<double>(), gamma_stride0,
                                            dest->Data<double>(), dest_stride0,
                                            stride1);
    } else {
      DiscountedCumSumCpuImpl<double>(N, T,
                                      src.Data<double>(), src_stride0,
                                      gamma.Data<double>(), gamma_stride0,
                                      dest->Data<double>(), dest_stride0,
                                      stride1);

    }
  } else {
    K2_LOG(FATAL) << "This algorithm only instantiated for float and double; type is "
                  << TraitsOf(src.GetDtype()).Name();
  }
}


Tensor Flip(Tensor &src, int32_t axis) {
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(axis, -num_axes);
  K2_CHECK_LT(axis, num_axes);
  if (axis < 0)
    axis += num_axes;
  int32_t old_dim = src.Dim(axis);
  if (old_dim <= 1)
    return src;  // No point copying it, it's a no-op.
  TensorImplPtr src_impl = src.Impl(),
      ans_impl = std::make_shared<TensorImpl>(*src_impl);
  int32_t old_stride = ans_impl->shape.Stride(axis);
  ans_impl->shape.SetStride(axis, -old_stride);
  int64_t byte_offset = old_stride * static_cast<int64_t>(old_dim - 1) *
      TraitsOf(ans_impl->dtype).NumBytes();
  ans_impl->byte_offset += byte_offset;
  return Tensor(ans_impl);
}


}  // namespace k2
