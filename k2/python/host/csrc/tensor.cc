// k2/python/host/csrc/tensor.cc

// Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
//                      Xiaomi Corporation (authors: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/tensor.h"

namespace k2host {

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

static DataType DLDataTypeToK2DataType(DLDataType data_type) {
  if (data_type.code == kDLInt && data_type.bits == 32) return kInt32Type;
  if (data_type.code == kDLFloat && data_type.bits == 32) return kFloatType;
  if (data_type.code == kDLFloat && data_type.bits == 64) return kDoubleType;
  K2_LOG(FATAL) << "Unsupported DLDataType: Code = " << data_type.code
                << ", Bits = " << data_type.bits;
  return kUnknownType;
}

static DeviceType DLDeviceTypeToK2DeviceType(DLDeviceType device_type) {
  switch (device_type) {
    case kDLCPU:
      return kCPU;
    case kDLGPU:
      return kGPU;
    default:
      K2_LOG(FATAL) << "Unsupported DLDeviceType: " << device_type;
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
  K2_CHECK_EQ(strcmp(kDLPackTensorName, capsule.name()), 0)
      << "Expected capsule name: " << kDLPackTensorName << "\n"
      << "But got: " << capsule.name() << "\n"
      << "Note that DLTensor capsules can be consumed only once,\n"
      << "so you might have already constructed a tensor from it once.";

  // Change the name of the capsule so that it will not be used again.
  PyCapsule_SetName(capsule.ptr(), kDLPackUsedTensorName);

  dl_managed_tensor_ = capsule;  // either throw or succeed with a non-null ptr

  dtype_ =
      DLDataTypeToK2DataType((DLDataType)dl_managed_tensor_->dl_tensor.dtype);

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
  K2_DCHECK_GE(i, 0);
  K2_DCHECK_LT(i, NumDim());
  return dl_managed_tensor_->dl_tensor.shape[i];
}

int64_t Tensor::Stride(int32_t i) const {
  K2_DCHECK_GE(i, 0);
  K2_DCHECK_LT(i, NumDim());
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
  K2_CHECK(dtype_ == kInt32Type || dtype_ == kFloatType ||
           dtype_ == kDoubleType)
      << "We support only int32_t, float and double at present";

  K2_CHECK(BytesPerElement() == 4 || BytesPerElement() == 8)
      << "Only int32_t, float and double are supported";

  K2_CHECK_EQ(dl_managed_tensor_->dl_tensor.dtype.lanes, 1u)
      << "We support only one lane";

  K2_CHECK_EQ(device_type_, kCPU) << "We support only kCPU at present";
}

}  // namespace k2host
