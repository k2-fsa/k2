/**
 * @brief A wrapper around Ragged<Any> and torch::Tensor
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/autograd/sum.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

RaggedAny::RaggedAny(const std::string &s, py::object dtype) {
  if (!dtype.is_none() && !THPDtype_Check(dtype.ptr())) {
    K2_LOG(FATAL) << "Expect an instance of torch.dtype. "
                  << "Given: " << py::str(dtype);
  }

  if (dtype.is_none()) {
    try {
      // we try int first, if it fails, use float
      any = Ragged<int32_t>(s).Generic();
      return;
    } catch (const std::exception &) {
      // we try int first, if it fails, use float
      any = Ragged<int32_t>(s).Generic();
    }
  }

  auto scalar_type = reinterpret_cast<THPDtype *>(dtype.ptr())->scalar_type;

  Dtype t = ScalarTypeToDtype(scalar_type);

  FOR_REAL_AND_INT32_TYPES(t, T, {
    any = Ragged<T>(s).Generic();
    return;
  });

  K2_LOG(FATAL) << "Unsupported dtype: " << scalar_type
                << ". Supported dtypes are: torch.int32, torch.float32, "
                << "and torch.float64";
}

RaggedAny::RaggedAny(py::list data, py::object dtype /*= py::none()*/) {
  if (!dtype.is_none() && !THPDtype_Check(dtype.ptr())) {
    K2_LOG(FATAL) << "Expect an instance of torch.dtype. "
                  << "Given: " << py::str(dtype);
  }

  if (dtype.is_none()) {
    try {
      // we try int first, if it fails, use float
      auto vecs = data.cast<std::vector<std::vector<int>>>();
      any = CreateRagged2(vecs).Generic();
      return;
    } catch (const std::exception &) {
      auto vecs = data.cast<std::vector<std::vector<float>>>();
      any = CreateRagged2(vecs).Generic();
      return;
    }
  }

  auto scalar_type = reinterpret_cast<THPDtype *>(dtype.ptr())->scalar_type;

  Dtype t = ScalarTypeToDtype(scalar_type);

  FOR_REAL_AND_INT32_TYPES(t, T, {
    auto vecs = data.cast<std::vector<std::vector<T>>>();
    any = CreateRagged2(vecs).Generic();
    return;
  });

  K2_LOG(FATAL) << "Unsupported dtype: " << scalar_type
                << ". Supported dtypes are: torch.int32, torch.float32, "
                << "and torch.float64";
}

const torch::Tensor &RaggedAny::Data() const {
  if (!data.defined()) {
    Dtype t = any.GetDtype();
    FOR_REAL_AND_INT32_TYPES(t, T, {
      const_cast<RaggedAny *>(this)->data =
          ToTorch((const_cast<RaggedAny *>(this)->any).Specialize<T>().values);
    });
  }
  return data;
}

std::string RaggedAny::ToString() const {
  std::ostringstream os;
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, { os << any.Specialize<T>(); });
  return os.str();
}

RaggedAny RaggedAny::To(torch::Device device) const {
  ContextPtr context = any.Context();
  if (device.is_cpu()) {
    // CPU -> CPU
    if (context->GetDeviceType() == kCpu) return *this;

    // CUDA -> CPU
    DeviceGuard guard(context);
    return RaggedAny(any.To(GetCpuContext()));
  }

  K2_CHECK(device.is_cuda()) << device.str();

  int32_t device_index = device.index();

  if (context->GetDeviceType() == kCuda &&
      context->GetDeviceId() == device_index)
    // CUDA to CUDA, and it's the same device
    return *this;

  // CPU to CUDA
  // or from one GPU to another GPU
  DeviceGuard guard(device_index);
  return RaggedAny(any.To(GetCudaContext(device_index)));
}

RaggedAny RaggedAny::To(torch::ScalarType scalar_type) const {
  Dtype d = any.GetDtype();

  switch (scalar_type) {
    case torch::kFloat:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return RaggedAny(any.Specialize<T>().ToFloat().Generic()); });
    case torch::kInt:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return RaggedAny(any.Specialize<T>().ToInt().Generic()); });
    case torch::kDouble:
      FOR_REAL_AND_INT32_TYPES(d, T, {
        return RaggedAny(any.Specialize<T>().ToDouble().Generic());
      });
    default:
      K2_LOG(FATAL) << "Unsupported scalar type: "
                    << torch::toString(scalar_type) << "\n";
  }
  // Unreachable code
  return {};
}

RaggedAny RaggedAny::Clone() const {
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(
      t, T, { return RaggedAny(any.Specialize<T>().Clone().Generic()); });

  // Unreachable code
  return {};
}

RaggedAny &RaggedAny::SetRequiresGrad(bool requires_grad /*=true*/) {
  // PyTorch will throw a RuntimeError exception if dtype is torch.int32
  // So no need to check it by us here
  Data().requires_grad_(requires_grad);
  return *this;
}

torch::Tensor RaggedAny::Sum(float initial_value /*=0*/) const {
  DeviceGuard guard(any.Context());
  return SumFunction::apply(*this, Data(), initial_value);
}

RaggedAny RaggedAny::Index(int32_t axis, int32_t i) const {
  K2_CHECK_EQ(axis, 0) << "Support only axis == 0 right now";

  return RaggedAny(any.Index(axis, i));
}

}  // namespace k2
