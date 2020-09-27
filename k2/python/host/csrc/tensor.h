// k2/python/host/csrc/tensor.h

// Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)

// See ../../../LICENSE for clarification regarding multiple authors

#ifndef K2_PYTHON_HOST_CSRC_TENSOR_H_
#define K2_PYTHON_HOST_CSRC_TENSOR_H_

#include "k2/python/host/csrc/dlpack.h"
#include "k2/python/host/csrc/k2.h"

namespace k2host {

enum class DataType : int8_t {
  kInt32Type = 0,
  kFloatType = 1,
  kDoubleType = 2,
  kUnknownType,
};

enum class DeviceType : int8_t {
  kCPU = 0,
  kGPU = 1,
  kUnknownDevice,
};

constexpr DataType kInt32Type = DataType::kInt32Type;
constexpr DataType kFloatType = DataType::kFloatType;
constexpr DataType kDoubleType = DataType::kDoubleType;
constexpr DataType kUnknownType = DataType::kUnknownType;

constexpr DeviceType kCPU = DeviceType::kCPU;
constexpr DeviceType kGPU = DeviceType::kGPU;
constexpr DeviceType kUnknownDevice = DeviceType::kUnknownDevice;

std::ostream &operator<<(std::ostream &os, DataType data_type);
std::ostream &operator<<(std::ostream &os, DeviceType data_type);

// It manages external memory passed from a PyCapsule
// via DLPack.
//
// To share memory with `torch::Tensor`, invoke `to_dlpack(torch::Tensor)`
// and then pass the capsule from Python to C++.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(py::capsule capsule);
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  ~Tensor();

  // Return true if the tensor is empty; false otherwise.
  bool Empty() const;

  // Return number of dimensions of the tensor.
  int32_t NumDim() const;

  // Get the i-th dimension.
  // `i` should be in [0, NumDim())
  int64_t Shape(int32_t i) const;

  // Return the stride for the i-th dimension.
  //
  // the returned result designates the number of elements, NOT number of bytes.
  // `i` should be in [0, NumDim())
  int64_t Stride(int32_t i) const;

  // The returned pointer is NOT owned by the caller.
  int64_t *Shape();

  // The returned pointer is NOT owned by the caller.
  int64_t *Stride();

  // The returned pointer is NOT owned by the caller.
  template <typename T>
  T *Data() {
    return reinterpret_cast<T *>(Data());
  }

  // The returned pointer is NOT owned by the caller.
  void *Data();

  // Return the number of bytes per element.
  int32_t BytesPerElement() const;

  DataType dtype() const { return dtype_; }
  DeviceType device_type() const { return device_type_; }

 private:
  void Check() const;

 private:
  DLManagedTensor *dl_managed_tensor_ = nullptr;
  DataType dtype_ = kUnknownType;
  DeviceType device_type_ = kUnknownDevice;
};

}  // namespace k2host

#endif  // K2_PYTHON_HOST_CSRC_TENSOR_H_
