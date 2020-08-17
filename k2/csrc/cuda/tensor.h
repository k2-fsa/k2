#include "k2/csrc/cuda/dtype.h"
#include "k2/csrc/cuda/context.h"


#define MAX_DIM 4  // Will increase this as needed

class Shape {
 public:
  int32_t Ndim() { return ndim_; }

  int32_t Dim(int32_t i) {
    CHECK_LT(dynamic_cast<uint32_t>(i), dynamic_cast<uint32_t>(ndim_));
    return dims_[i];
  }
  int32_t Stride(int32_t i) {
    CHECK_LT(dynamic_cast<uint32_t>(i), dynamic_cast<uint32_t>(ndim_));
    return strides_[i];
  }
  int32_t Nelement(); // compute, return number of elements..


  explicit Shape(const std::vector<int32_t> &dims,
                 int32_t bytes_per_elem);

  explicit Shape(const std::vector<int32_t> &dims,
                 const std::vector<int32_t> strides);

  Shape(const Shape &other);

  bool IsContiguous(int32_t bytes_per_elem);

 private:
  int32_t ndim_;  // Must be >= 0
  // elements of dims_ and strides_ >= ndim_ are currently not set;
  // in future we may change this.
  int32_t dims_[MAX_DIM];
  int32_t strides_[MAX_DIM];  // Strides in bytes
};



class Tensor {
 public:

  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const Shape &shape);

  // Creates Tensor backed by uninitialized memory
  Tensor(ContextPtr c, Dtype type, const std::vector<int32_t> &dims);

  // Create Tensor backed by existing memory.
  Tensor(const Shape &shape, Dtype dtype, RegionPtr region, size_t bytes_offset);

  template <typename T> T *data();  // Returns pointer to elem with index
                                    // all-zeros... will check that the type
                                    // matches the correct one.


  // Return the result of indexing one of the axes, which will result in a
  // Tensor with one fewer axis.
  TensorPtr Index(int32_t axis, int32_t index);

  Dtype GetDtype() { return dtype_; }
  const Shape &GetShape() { return shape_; }
  std::shared_ptr<Region> &GetRegion() { return data_; }

 private:

  Shape shape_;
  Dtype dtype_;
  std::shared_ptr<Region> data_;

};

using TensorPtr = std::shared_ptr<Tensor>;
