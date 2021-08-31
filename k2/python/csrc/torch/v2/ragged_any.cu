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

#include "k2/csrc/ragged_ops.h"
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
      any = Ragged<int32_t>(s, /*throw_on_failure*/ true).Generic();
      return;
    } catch (const std::runtime_error &) {
      // we try int first, if it fails, use float
      any = Ragged<float>(s).Generic();
      return;
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

  DeviceGuard guard(any.Context());
  return RaggedAny(any.Index(axis, i));
}

RaggedAny RaggedAny::RemoveAxis(int32_t axis) /*const*/ {
  DeviceGuard guard(any.Context());
  return RaggedAny(any.RemoveAxis(axis));
}

RaggedAny RaggedAny::Arange(int32_t axis, int32_t begin,
                            int32_t end) /*const*/ {
  DeviceGuard guard(any.Context());
  return RaggedAny(k2::Arange(any, axis, begin, end));
}

RaggedAny RaggedAny::RemoveValuesLeq(float cutoff) /*const*/ {
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return RaggedAny(
        k2::RemoveValuesLeq<T>(any.Specialize<T>(), cutoff).Generic());
  });
}

RaggedAny RaggedAny::RemoveValuesEq(float target) /*const*/ {
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return RaggedAny(
        k2::RemoveValuesEq<T>(any.Specialize<T>(), target).Generic());
  });
}

torch::Tensor RaggedAny::ArgMaxPerSublist(py::object initial_value) /*const*/ {
  K2_CHECK((bool)initial_value);
  K2_CHECK(!initial_value.is_none());

  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    k2::ArgMaxPerSublist<T>(any.Specialize<T>(), initial_value.cast<T>(),
                            &indexes);
  });

  return ToTorch(indexes);
}

torch::Tensor RaggedAny::MaxPerSublist(py::object initial_value) /*const*/ {
  K2_CHECK((bool)initial_value);
  K2_CHECK(!initial_value.is_none());

  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array1<T> max_values(any.Context(), num_rows);
    k2::MaxPerSublist<T>(any.Specialize<T>(), initial_value.cast<T>(),
                         &max_values);
    return ToTorch(max_values);
  });
  // Unreachable code
  return {};
}

torch::Tensor RaggedAny::MinPerSublist(py::object initial_value) /*const*/ {
  K2_CHECK((bool)initial_value);
  K2_CHECK(!initial_value.is_none());

  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array1<T> min_values(any.Context(), num_rows);
    k2::MinPerSublist<T>(any.Specialize<T>(), initial_value.cast<T>(),
                         &min_values);
    return ToTorch(min_values);
  });
  // Unreachable code
  return {};
}

RaggedAny RaggedAny::Cat(const std::vector<RaggedAny> &srcs, int32_t axis) {
  K2_CHECK_GT(srcs.size(), 0);
  DeviceGuard guard(srcs[0].any.Context());

  Dtype t = srcs[0].any.GetDtype();
  int32_t num_srcs = srcs.size();

  FOR_REAL_AND_INT32_TYPES(t, T, {
    std::vector<Ragged<T>> tmp;
    tmp.reserve(num_srcs);
    for (const auto &s : srcs) {
      tmp.push_back(s.any.Specialize<T>());
    }
    return RaggedAny(
        k2::Cat(axis, num_srcs, tmp.data(), /*merge_map*/ nullptr).Generic());
  });

  // Unreachable code
  return {};
}

RaggedAny RaggedAny::NormalizePerSublist(bool use_log) /*const*/ {
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();

  if (t == kFloatDtype) {
    return RaggedAny(
        k2::NormalizePerSublist(any.Specialize<float>(), use_log).Generic());
  }

  if (t == kDoubleDtype) {
    return RaggedAny(
        k2::NormalizePerSublist(any.Specialize<double>(), use_log).Generic());
  }

  K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(t).Name();
  return {};
}

torch::Tensor RaggedAny::Pad(const std::string &mode,
                             py::object padding_value) /*const*/ {
  K2_CHECK((bool)padding_value);
  K2_CHECK(!padding_value.is_none());

  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array2<T> arr =
        PadRagged(any.Specialize<T>(), mode, padding_value.cast<T>());
    return ToTorch(arr);
  });
  // Unreachable code
  return {};
}

template <typename T>
static py::list ToList(Ragged<T> &src, int32_t axis, int32_t begin,
                       int32_t end) {
  // assuming src is on CPU
  int32_t num_axes = src.NumAxes();

  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis, num_axes);
  K2_CHECK_LE(begin, end);
  K2_CHECK_LE(end, src.TotSize(axis));

  py::list ans(end - begin);
  if (axis == num_axes - 1) {
    const T *data = src.values.Data();
    // recursion ends here
    for (int32_t i = begin; i != end; ++i) {
      ans[i - begin] = data[i];
    }
  } else {
    const int32_t *data = src.RowSplits(axis + 1).Data();
    for (int32_t i = begin; i != end; ++i) {
      ans[i - begin] = ToList(src, axis + 1, data[i], data[i + 1]);
    }
  }
  return ans;
}

py::list RaggedAny::ToList() /*const*/ {
  if (any.Context()->GetDeviceType() != kCpu) {
    return RaggedAny(any.To(GetCpuContext())).ToList();
  }

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return k2::ToList(any.Specialize<T>(), /*axis*/ 0, /*begin*/ 0,
                      /*end*/ any.Dim0());
  });

  // Unreachable code
  return py::none();
}

torch::optional<torch::Tensor> RaggedAny::SortSublists(
    bool descending /*= false*/, bool need_new2old_indexes /*= false*/) {
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();

  Array1<int32_t> new2old;

  if (need_new2old_indexes) {
    new2old = Array1<int32_t>(any.Context(), any.NumElements());
  }

  FOR_REAL_AND_INT32_TYPES(t, T, {
    if (descending) {
      k2::SortSublists<T, GreaterThan<T>>(
          &any.Specialize<T>(), need_new2old_indexes ? &new2old : nullptr);
    } else {
      k2::SortSublists<T, LessThan<T>>(
          &any.Specialize<T>(), need_new2old_indexes ? &new2old : nullptr);
    }
  });

  torch::optional<torch::Tensor> ans;
  if (need_new2old_indexes) ans = ToTorch(new2old);
  return ans;
}

}  // namespace k2
