// k2/csrc/cuda/ops.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_OPS_H_
#define K2_CSRC_CUDA_OPS_H_

#include <cassert>
#include <type_traits>

#include "k2/csrc/cuda/array.h"
#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/error.h"

// Note, I'm not sure about the name of this file, they are not ops like in
// TensorFlow, but procedures..

namespace {
// TODO(haowen): manage/load block config with some classes? then we can get
// different configuration depending on num_elements and data type.
// block size for matrix transpose.
constexpr int kTransTileDim = 32;
constexpr int kTransBlockRows = 8;
}  // namespace

namespace k2 {
// TODO(haowen): move the implementations to file `op_inl.h` or
// `op.cu`(specialized on device and data type)?
template <typename T>
__global__ void TransposeKernel(int32_t rows, int32_t cols, const T *input,
                                T *output) {
  // TODO(haowen): here we need to handle different type of T to avoid bank
  // conflicts, the size of cache now is fine for type size with 32bit (e.g.
  // int32 or float).
  __shared__ T cache[kTransTileDim][kTransTileDim + 1];

  // input index, in a coalesced manner.
  int32_t x = threadIdx.x + blockIdx.x * kTransTileDim;
  int32_t y = threadIdx.y + blockIdx.y * kTransTileDim;

  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < cols && (y + i) < rows) {
      cache[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
    }
  }

  __syncthreads();

  // output index, in a coalesced manner
  x = threadIdx.x + blockIdx.y * kTransTileDim;
  y = threadIdx.y + blockIdx.x * kTransTileDim;
  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < rows && (y + i) < cols) {
      output[(y + i) * rows + x] = cache[threadIdx.x][threadIdx.y + i];
    }
  }
}

/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().  This is not the only way to transpose a matrix, you can also
  do: dest = Array2<T>(src.ToTensor().Transpose(0,1)), which will likely call
  this function

     @param [in] c   Context to use, must satisfy
                     `c.IsCompatible(src.Context())` and
                     `c.IsCompatible(dest->Context())`.
     @param [in] src  Source array to transpose
     @param [out] dest  Destination array; must satisfy
                        `dest->Size1() == src.Size0()` and
                        `dest->Size0() == src.Size1()`.
                        At exit, we'll have dest[i,j] == src[j,i].
 */
template <typename T>
void Transpose(ContextPtr &c, const Array2<T> &src, Array2<T> *dest) {
  assert(c.IsCompatible(src.Context()));
  assert(c.IsCompatible(dest->Context()));
  int32_t rows = src.Dim0();
  int32_t cols = src.Dim1();
  // TODO(haowen): limit the number of elements?
  assert(rows == dest->Dim1());
  assert(cols == dest->Dim0());
  const T *src_data = src.Data();
  T *dest_data = dest->Data();
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        dest_data[i * rows + j] = src_data[j * cols + i];
      }
    }
  } else {
    assert(d == kCuda);
    dim3 block_size(kTransTileDim, kTransBlockRows, 1);
    dim3 grid_size(NumBlocks(cols, kTransTileDim),
                   NumBlocks(rows, kTransTileDim));
    TransposeKernel<<<grid_size, block_size, 0, c->GetCudaStream()>>>(
        rows, cols, src_data, dest_data);
    CheckCudaError(cudaDeviceSynchronize());
  }
}

/*
  Sets 'dest' to exclusive prefix sum of 'src'.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data.  Must satisfy dest.Size() == src.Size()
                       or dest.Size() == src.Size() + 1, but in the latter case
                       we require that the memory region inside src be allocated
                       with at least one extra element, because the
                       exclusive-sum code may read from it even though it
                       doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].
                       Must be on same device as src.
 */
template <typename S, typename T>
void ExclusiveSum(ContextPtr &c, Array1<S> &src, Array1<T> *dest);

/*
  Sets 'dest' to exclusive prefix sum of 'src', along a specified axis.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data; allowed to be the same as src.
                       For axis==1, for example, at exit it will satisfy
                       dest[i][j] == sum_{k=0}^{j-1} src[i][k].
                       Must have the same size on the other axis; on the axis
                       being summed, must be either the same size as src,
                       or one greater. as src.
    @param [in] axis   Determines in what direction we sum, e.g. axis = 0 means
                       summation is over row axis (slower because we have to
                       transpose), axis = 1 means summation is over column axis.
 */
template <typename T>
void ExclusiveSum(ContextPtr &c, Array2<T> &src, Array2<T> *dest, int axis);



/*
  Return an array with dimension in.Dim0(), containing the maximum of each sub-list
  in 'in' (i.e. the max taken over axis 1), or T, whichever was larger.

  This is expected to be instantiated for, at least, float and int32_t.
 */
template <typename T>
Array1<T> MaxPerSublist(Ragged2<T> &in, T default);


}  // namespace k2

#endif  // K2_CSRC_CUDA_OPS_H_
