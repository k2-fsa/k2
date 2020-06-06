// k2/python/csrc/tensor.h

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#ifndef K2_PYTHON_CSRC_TENSOR_H_
#define K2_PYTHON_CSRC_TENSOR_H_

#include <type_traits>

#include "k2/python/csrc/dlpack.h"
#include "k2/python/csrc/k2.h"

namespace k2 {

enum class DataType : int8_t {
  kInt32Type = 0,
  kFloatType = 1,
  kUnknownType,
};

enum class DeviceType : int8_t {
  kCPU = 0,
  kGPU = 1,
  kUnknownDevice,
};

constexpr DataType kInt32Type = DataType::kInt32Type;
constexpr DataType kFloatType = DataType::kFloatType;
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
  ~Tensor();

  bool Empty() const;

  int32_t NumDim() const;

  int64_t Shape(int32_t i) const;

  // the returned result designates the number of elements, NOT number of bytes
  int64_t Stride(int32_t i) const;

  int64_t *Shape();
  int64_t *Stride();

  template <typename T,
            typename std::enable_if<std::is_same<T, int32_t>::value ||
                                        std::is_same<T, float>::value,
                                    T>::type * = nullptr>
  T *Data() {
    return reinterpret_cast<T *>(Data());
  }

  void *Data();

  DataType dtype() const { return dtype_; }
  DeviceType device_type() const { return device_type_; }

 private:
  void Check() const;

 private:
  DLManagedTensor *dl_managed_tensor_ = nullptr;
  DataType dtype_ = kUnknownType;
  DeviceType device_type_ = kUnknownDevice;
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TENSOR_H_
