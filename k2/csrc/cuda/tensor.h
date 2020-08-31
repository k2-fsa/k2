// k2/csrc/cuda/tensor.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_TENSOR_H_
#define K2_CSRC_CUDA_TENSOR_H_

#include <vector>

#include "k2/csrc/cuda/context.h"
#include "k2/csrc/cuda/dtype.h"

namespace k2 {
class Shape {
 public:
  int32_t Ndim() const { return ndim_; }

  std::vector<int32_t> Dims() const { return dims_; }

  std::vector<int32_t> Strides() const { return strides_; }

  int32_t Dim(int32_t i) const {
    CHECK_LT(static_cast<uint32_t>(i), static_cast<uint32_t>(ndim_));
    return dims_[i];
  }

  int32_t Stride(int32_t i) const {
    CHECK_LT(static_cast<uint32_t>(i), static_cast<uint32_t>(ndim_));
    return strides_[i];
  }

  int32_t Nelement() const { return num_element_; }
  bool IsContiguous() const { return is_contiguous_; }

  Shape() : ndim_(0), num_element_(0), is_contiguous_(false) {}

  explicit Shape(const std::vector<int32_t> &dims);

  explicit Shape(const std::vector<int32_t> &dims,
                 const std::vector<int32_t> strides);

  Shape(const Shape &other) = default;

 private:
  static const int32_t kMaxDim = 4;  // Will increase this as needed

  int32_t ndim_;  // Must be >= 0
  int32_t num_element_;
  bool is_contiguous_;

  std::vector<int32_t> dims_;
  std::vector<int32_t> strides_;  // Strides in elements

  // compute the number of elements
  int32_t ComputeNumElement();
  bool CheckContiguous();
};

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/*
  Tensor is similar to PyTorch or TF Tensor.  Note, we don't use this that
  often: more often, we use templated types that make stronger assumptions about
  the dtype and layout, such as Array1, Array2 and Ragged.
 */
class Tensor {
 public:
  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const Shape &shape);

  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims);

  // Create Tensor backed by existing memory.
  Tensor(Dtype type, const Shape &shape, RegionPtr region, size_t bytes_offset);

  // Returns pointer to elem with index all-zeros... will check that the type
  // matches the correct one.
  template <typename T>
  T *data() {
    assert(dtype_ == DtypeOf<T>::dtype);
    return reinterpret_cast<T *>(reinterpret_cast<char *>(data_->data) +
                                 bytes_offset_);
  }

  // Return the result of indexing one of the axes, which will result in a
  // Tensor with one fewer axis.
  TensorPtr Index(int32_t axis, int32_t index) const;

  Dtype GetDtype() const { return dtype_; }
  const Shape &GetShape() const { return shape_; }
  std::shared_ptr<Region> &GetRegion() { return data_; }

 private:
  Shape shape_;
  Dtype dtype_;
  int32_t bytes_offset_;
  std::shared_ptr<Region> data_;

  void Init(ContextPtr c);
};

}  // namespace k2
#endif  // K2_CSRC_CUDA_TENSOR_H_
