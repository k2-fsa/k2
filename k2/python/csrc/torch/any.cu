/**
 * @brief Wraps Ragged<Any>
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Daniel Povey)
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
#include <memory>
#include <string>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/any.h"
#include "k2/python/csrc/torch/torch_util.h"

namespace k2 {

using RaggedAny = Ragged<Any>;

/* Create a ragged tensor with two axes.

   @param data a list-of-list
   @param dtype An instance of torch.dtype. If it is None,
                the data type is inferred from the input `data`,
                which will either be torch.int32 or torch.float32.

   @TODO To support `data` with arbitrary number of axes.

   @CAUTION Currently supported dtypes are torch.float32, torch.float64,
   and torch.int32. To support torch.int64 and other dtypes, we can
   add a new macro to replace `FOR_REAL_AND_INT32_TYPES`.

   @return A ragged tensor with two axes.
 */
static RaggedAny CreateRagged2(py::list data, py::object dtype = py::none()) {
  if (!dtype.is_none() && !THPDtype_Check(dtype.ptr())) {
    K2_LOG(FATAL) << "Expect an instance of torch.dtype. "
                  << "Given: " << py::str(dtype);
  }

  if (dtype.is_none()) {
    try {
      // we try int first, if it fails, use float
      auto vecs = data.cast<std::vector<std::vector<int>>>();
      return CreateRagged2(vecs).Generic();
    } catch (const std::exception &) {
      auto vecs = data.cast<std::vector<std::vector<float>>>();
      return CreateRagged2(vecs).Generic();
    }
  }

  auto scalar_type = reinterpret_cast<THPDtype *>(dtype.ptr())->scalar_type;

  Dtype t = ScalarTypeToDtype(scalar_type);

  // TODO: To support kInt64Dtype
  FOR_REAL_AND_INT32_TYPES(t, T, {
    auto vecs = data.cast<std::vector<std::vector<T>>>();
    return CreateRagged2(vecs).Generic();
  });

  K2_LOG(FATAL) << "Unsupported dtype: " << scalar_type
                << ". Supported dtypes are: torch.int32, torch.float32, "
                << "and torch.float64";

  // Unreachable code
  return {};
}

/* Convert a ragged tensor to a string.

   @param any The input ragged tensor.
   @return Return a string representation of the ragged tensor.
 */
static std::string ToString(const RaggedAny &any) {
  std::ostringstream os;
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, { os << any.Specialize<T>(); });
  return os.str();
}

/* Move a ragged tensor to a given device.

   Note: If the input tensor is already on the given device, itself
   is returned. Otherwise, a copy of the input tensor moved to the given
   device is returned.

   @param any  The input ragged tensor.
   @param device  A torch device, which can be either a CPU device
                  or a CUDA device.
 */
static RaggedAny To(const RaggedAny &any, torch::Device device) {
  ContextPtr context = any.Context();
  if (device.is_cpu()) {
    // CPU -> CPU
    if (context->GetDeviceType() == kCpu) return any;

    // CUDA -> CPU
    DeviceGuard guard(context);
    return any.To(GetCpuContext());
  }

  K2_CHECK(device.is_cuda()) << device.str();

  int32_t device_index = device.index();

  if (context->GetDeviceType() == kCuda &&
      context->GetDeviceId() == device_index)
    // CUDA to CUDA, and it's the same device
    return any;

  // CPU to CUDA
  // or from one GPU to another GPU
  DeviceGuard guard(device_index);
  return any.To(GetCudaContext(device_index));
}

static RaggedAny To(const RaggedAny &any, torch::ScalarType scalar_type) {
  Dtype d = any.GetDtype();

  switch (scalar_type) {
    case torch::kFloat:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return any.Specialize<T>().ToFloat().Generic(); });
    case torch::kInt:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return any.Specialize<T>().ToInt().Generic(); });
    case torch::kDouble:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return any.Specialize<T>().ToDouble().Generic(); });
    case torch::kLong:
      FOR_REAL_AND_INT32_TYPES(
          d, T, { return any.Specialize<T>().ToLong().Generic(); });
    default:
      K2_LOG(FATAL) << "Unsupported scalar type: "
                    << torch::toString(scalar_type) << "\n";
  }
  // Unreachable code
  return {};
}

void PybindAny(py::module &m) {
  py::module ragged = m.def_submodule(
      "ragged", "Sub module containing operations for ragged tensors in k2");

  py::class_<RaggedAny> any(ragged, "Tensor");
  any.def(py::init<>());

  any.def(
      py::init([](py::list data,
                  py::object dtype = py::none()) -> std::unique_ptr<RaggedAny> {
        auto any = CreateRagged2(data, dtype);
        return std::make_unique<RaggedAny>(any.shape, any.values);
      }),
      py::arg("data"), py::arg("dtype") = py::none());

  any.def("__str__",
          [](const RaggedAny &self) -> std::string { return ToString(self); });
  // o is either torch.device or torch.dtype
  any.def("to", [](const RaggedAny &self, py::object o) -> RaggedAny {
    PyObject *ptr = o.ptr();
    if (THPDevice_Check(ptr)) {
      torch::Device device = reinterpret_cast<THPDevice *>(ptr)->device;
      return To(self, device);
    }

    if (THPDtype_Check(ptr)) {
      auto scalar_type = reinterpret_cast<THPDtype *>(ptr)->scalar_type;
      return To(self, scalar_type);
    }

    K2_LOG(FATAL)
        << "Expect an instance of torch.device or torch.dtype. Given: "
        << py::str(o);

    // Unreachable code
    return {};
  });

  any.def_property_readonly("dtype", [](const RaggedAny &self) -> py::object {
    Dtype t = self.GetDtype();
    auto torch = py::module::import("torch");
    switch (t) {
      case kFloatDtype:
        return torch.attr("float32");
      case kDoubleDtype:
        return torch.attr("float64");
      case kInt32Dtype:
        return torch.attr("int32");
      case kInt64Dtype:
        return torch.attr("int64");
      default:
        K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(t).Name();
    }

    // Unreachable code
    return py::none();
  });

  any.def_property_readonly("device", [](const RaggedAny &self) -> py::object {
    DeviceType d = self.Context()->GetDeviceType();
    torch::DeviceType device_type = ToTorchDeviceType(d);

    torch::Device device(device_type, self.Context()->GetDeviceId());

    PyObject *ptr = THPDevice_New(device);
    py::handle h(ptr);

    // takes ownership
    return py::reinterpret_steal<py::object>(h);
  });

  // Return the underlying memory of this tensor.
  // No data is copied. Memory is shared.
  any.def_property_readonly("data", [](RaggedAny &self) -> torch::Tensor {
    Dtype t = self.GetDtype();
    FOR_REAL_AND_INT32_TYPES(t, T,
                             { return ToTorch(self.values.Specialize<T>()); });
  });

  ragged.def(
      "tensor",
      [](py::list data, py::object dtype = py::none()) -> RaggedAny {
        return CreateRagged2(data, dtype);
      },
      py::arg("data"), py::arg("dtype") = py::none());
}

}  // namespace k2
