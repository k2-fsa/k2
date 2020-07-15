// k2/python/csrc/tensor.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/tensor.h"

#include "glog/logging.h"

namespace k2 {

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L375
// https://github.com/microsoft/onnxruntime-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L28
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L66
// PyTorch, TVM and CuPy name the created dltensor to be `dltensor`
static const char *kDLPackTensorName = "dltensor";

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L402
// https://github.com/apache/incubator-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L29
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L62
// PyTorch, TVM and CuPy name the used dltensor to be `used_dltensor`
static const char *kDLPackUsedTensorName = "used_dltensor";

static DataType DLDataTypeToK2DataType(DLDataTypeCode data_type) {
  switch (data_type) {
    case kDLInt:
      return kInt32Type;
    case kDLFloat:
      return kFloatType;
    default:
      LOG(FATAL) << "Unsupported DLDataTypeCode: " << data_type;
      return kUnknownType;
  }
}

static DeviceType DLDeviceTypeToK2DeviceType(DLDeviceType device_type) {
  switch (device_type) {
    case kDLCPU:
      return kCPU;
    case kDLGPU:
      return kGPU;
    default:
      LOG(FATAL) << "Unsupported DLDeviceType: " << device_type;
      return kUnknownDevice;
  }
}

std::ostream &operator<<(std::ostream &os, DataType data_type) {
  os << static_cast<int32_t>(data_type);
  return os;
}
std::ostream &operator<<(std::ostream &os, DeviceType device_type) {
  os << static_cast<int32_t>(device_type);
  return os;
}

Tensor::Tensor(py::capsule capsule) {
  // the following error message is modified from
  //     https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L384
  CHECK_EQ(strcmp(kDLPackTensorName, capsule.name()), 0)
      << "Expected capsule name: " << kDLPackTensorName << "\n"
      << "But got: " << capsule.name() << "\n"
      << "Note that DLTensor capsules can be consumed only once,\n"
      << "so you might have already constructed a tensor from it once.";

  // Change the name of the capsule so that it will not be used again.
  PyCapsule_SetName(capsule.ptr(), kDLPackUsedTensorName);

  dl_managed_tensor_ = capsule;  // either throw or succeed with a non-null ptr

  dtype_ = DLDataTypeToK2DataType(
      (DLDataTypeCode)dl_managed_tensor_->dl_tensor.dtype.code);

  device_type_ =
      DLDeviceTypeToK2DeviceType(dl_managed_tensor_->dl_tensor.ctx.device_type);

  Check();
}

Tensor::~Tensor() {
  if (dl_managed_tensor_ && dl_managed_tensor_->deleter)
    dl_managed_tensor_->deleter(dl_managed_tensor_);

  dl_managed_tensor_ = nullptr;
}

bool Tensor::Empty() const { return dl_managed_tensor_ != nullptr; }

int32_t Tensor::NumDim() const { return dl_managed_tensor_->dl_tensor.ndim; }

int64_t Tensor::Shape(int32_t i) const {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, NumDim());
  return dl_managed_tensor_->dl_tensor.shape[i];
}

int64_t Tensor::Stride(int32_t i) const {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, NumDim());
  return dl_managed_tensor_->dl_tensor.strides[i];
}

int64_t *Tensor::Shape() { return dl_managed_tensor_->dl_tensor.shape; }

int64_t *Tensor::Stride() { return dl_managed_tensor_->dl_tensor.strides; }

void *Tensor::Data() {
  return reinterpret_cast<char *>(dl_managed_tensor_->dl_tensor.data) +
         dl_managed_tensor_->dl_tensor.byte_offset;
}

int32_t Tensor::BytesPerElement() const {
  return dl_managed_tensor_->dl_tensor.dtype.bits / 8;
}

void Tensor::Check() const {
  CHECK(dtype_ == kInt32Type || dtype_ == kFloatType)
      << "We support only int32_t and float at present";

  CHECK_EQ(BytesPerElement(), 4) << "Only int32_t and float are supported";

  CHECK_EQ(dl_managed_tensor_->dl_tensor.dtype.lanes, 1u)
      << "We support only one lane";

  CHECK_EQ(device_type_, kCPU) << "We support only kCPU at present";
}

}  // namespace k2
