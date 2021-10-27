/**
 * @brief A wrapper around Ragged<Any> and torch::Tensor
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang,
 *                                              Wei Kang)
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

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/autograd/index_and_sum.h"
#include "k2/torch/csrc/autograd/normalize.h"
#include "k2/torch/csrc/autograd/sum.h"
#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

static void PrintSpaces(std::ostream &os, int32_t num_spaces) {
  K2_CHECK_GE(num_spaces, 0);
  for (int32_t i = 0; i != num_spaces; ++i) os << " ";
}

template <typename T>
void RaggedAnyToStringIter(std::ostream &os, const Ragged<T> ragged,
                           int32_t axis, int32_t begin_pos, int32_t end_pos,
                           int32_t num_indent, bool compact) {
  const auto &shape = ragged.shape;
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() && begin_pos >= 0 &&
           begin_pos <= end_pos && end_pos <= shape.TotSize(axis));
  std::string sep = "";
  bool is_first_row = true;
  for (int32_t d = begin_pos; d < end_pos; d++) {
    if (axis == shape.NumAxes() - 1) {
      os << sep << ragged.values[d];
      sep = ", ";
    } else {
      const int32_t *row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK_LE(d, shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d], row_end = row_splits[d + 1];

      if (!compact && !is_first_row) {
        PrintSpaces(os, num_indent + 1);
      }
      is_first_row = false;

      os << "[";

      RaggedAnyToStringIter(os, ragged, axis + 1, row_start, row_end,
                            num_indent + 1, compact);
      os << "]";
      if (d != end_pos - 1) {
        if (compact)
          os << ", ";
        else
          os << ",\n";
      }
    }
  }
}

RaggedAny::RaggedAny(const RaggedShape &shape, torch::Tensor value)
    : data(value) {
  ContextPtr context = GetContext(value);
  DeviceGuard guard(context);

  Dtype t = ScalarTypeToDtype(value.scalar_type());
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array1<T> array = FromTorch<T>(value);
    Ragged<T> r(shape, array);
    any = r.Generic();
    return;
  });
  // Unreachable code
  K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(t).Name();
}

RaggedAny::RaggedAny(const std::string &s,
                     torch::optional<torch::ScalarType> dtype /*={}*/,
                     torch::Device device /*=torch::kCPU*/) {
  ContextPtr context = GetContext(device);
  DeviceGuard guard(context);

  if (!dtype.has_value()) {
    try {
      // We try int first, if it fails, use float
      any = Ragged<int32_t>(s, /*throw_on_failure*/ true).To(context).Generic();
      return;
    } catch (const std::runtime_error &) {
      // Use float. If it fails again, another exception
      // is thrown and it is propagated to the user
      any = Ragged<float>(s).To(context).Generic();
      return;
    }
  }
  auto scalar_type = dtype.value();
  Dtype t = ScalarTypeToDtype(scalar_type);

  FOR_REAL_AND_INT32_TYPES(t, T, {
    any = Ragged<T>(s).To(context).Generic();
    return;
  });

  K2_LOG(FATAL) << "Unsupported dtype: " << scalar_type
                << ". Supported dtypes are: torch.int32, torch.float32, "
                << "and torch.float64";
}

RaggedAny::RaggedAny(torch::Tensor tensor) {
  int32_t ndim = tensor.dim();
  K2_CHECK_GE(ndim, 2) << "Expect a tensor with more than 1-D";
  ContextPtr context = GetContext(tensor);
  DeviceGuard guard(context);
  std::vector<RaggedShape> shapes;
  shapes.reserve(ndim - 1);
  int32_t dim0 = tensor.size(0);
  for (int32_t i = 1; i != ndim; ++i) {
    int32_t dim1 = tensor.size(i);
    shapes.push_back(RegularRaggedShape(context, dim0, dim1));
    dim0 *= dim1;
  }
  while (shapes.size() > 2u) {
    RaggedShape c = std::move(shapes.back());
    shapes.pop_back();

    RaggedShape b = std::move(shapes.back());
    shapes.pop_back();

    RaggedShape a = std::move(shapes.back());
    shapes.pop_back();

    RaggedShape abc = ComposeRaggedShapes3(a, b, c);
    shapes.push_back(std::move(abc));
  }

  if (shapes.size() > 1u) {
    RaggedShape b = std::move(shapes.back());
    shapes.pop_back();

    RaggedShape a = std::move(shapes.back());
    shapes.pop_back();

    RaggedShape ab = ComposeRaggedShapes(a, b);
    shapes.push_back(std::move(ab));
  }

  Dtype t = ScalarTypeToDtype(tensor.scalar_type());
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array1<T> values = FromTorch<T>(tensor.contiguous().view({-1}));
    any = Ragged<T>(shapes[0], values).Generic();
  });
}

const torch::Tensor &RaggedAny::Data() const {
  DeviceGuard guard(any.Context());
  if (!data.defined()) {
    Dtype t = any.GetDtype();
    FOR_REAL_AND_INT32_TYPES(t, T, {
      const_cast<RaggedAny *>(this)->data =
          ToTorch((const_cast<RaggedAny *>(this)->any).Specialize<T>().values);
    });
  }
  return data;
}

std::string RaggedAny::ToString(bool compact /*=false*/,
                                int32_t device_id /*=-1*/) const {
  ContextPtr context = any.Context();
  if (context->GetDeviceType() != kCpu) {
    return To("cpu").ToString(context->GetDeviceId());
  }

  std::ostringstream os;
  Dtype t = any.GetDtype();
  std::string dtype;
  if (t == kInt32Dtype)
    dtype = "torch.int32";
  else if (t == kFloatDtype)
    dtype = "torch.float32";
  else if (t == kDoubleDtype)
    dtype = "torch.float64";
  else
    K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(t).Name();

  FOR_REAL_AND_INT32_TYPES(t, T, {
    os << "RaggedTensor([";
    // 13 is strlen("RaggedTensor(")
    RaggedAnyToStringIter(os, any.Specialize<T>(), 0, 0, any.shape.Dim0(), 13,
                          compact);
    os << "]";
    if (device_id != -1) os << ", device='cuda:" << device_id << "'";
    os << ", dtype=" << dtype;
    os << ")";
  });
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

RaggedAny RaggedAny::To(const std::string &device) const {
  torch::Device d(device);
  return this->To(d);
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
  DeviceGuard guard(any.Context());
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

RaggedAny RaggedAny::RemoveValuesLeq(torch::IValue cutoff) /*const*/ {
  K2_CHECK(!cutoff.isNone());
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return RaggedAny(
        k2::RemoveValuesLeq<T>(any.Specialize<T>(), cutoff.to<T>()).Generic());
  });

  // Unreachable code
  return {};
}

RaggedAny RaggedAny::RemoveValuesEq(torch::IValue target) /*const*/ {
  K2_CHECK(!target.isNone());
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return RaggedAny(
        k2::RemoveValuesEq<T>(any.Specialize<T>(), target.to<T>()).Generic());
  });
  // Unreachable code
  return {};
}

torch::Tensor RaggedAny::ArgMax(
    torch::IValue initial_value /*= {}*/) /*const*/ {
  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    T v = initial_value.isNone() ? std::numeric_limits<T>::lowest()
                                 : initial_value.to<T>();
    ArgMaxPerSublist<T>(any.Specialize<T>(), v, &indexes);
  });

  return ToTorch(indexes);
}

torch::Tensor RaggedAny::Max(torch::IValue initial_value /*= {}*/) /*const*/ {
  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    T v = initial_value.isNone() ? std::numeric_limits<T>::lowest()
                                 : initial_value.to<T>();
    Array1<T> max_values(any.Context(), num_rows);
    MaxPerSublist<T>(any.Specialize<T>(), v, &max_values);
    return ToTorch(max_values);
  });
  // Unreachable code
  return {};
}

torch::Tensor RaggedAny::Min(torch::IValue initial_value /*= {}*/) /*const*/ {
  DeviceGuard guard(any.Context());
  int32_t last_axis = any.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = any.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;

  Array1<int32_t> indexes(any.Context(), num_rows);

  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    T v = initial_value.isNone() ? std::numeric_limits<T>::max()
                                 : initial_value.to<T>();
    Array1<T> min_values(any.Context(), num_rows);
    MinPerSublist<T>(any.Specialize<T>(), v, &min_values);
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

std::tuple<RaggedAny, torch::optional<RaggedAny>,
           torch::optional<torch::Tensor>>
RaggedAny::Unique(bool need_num_repeats /*= false*/,
                  bool need_new2old_indexes /*= false*/) {
  DeviceGuard guard(any.Context());

  Dtype t = any.GetDtype();
  K2_CHECK_EQ(t, kInt32Dtype) << "Unsupported dtype: " << TraitsOf(t).Name();

  Ragged<int32_t> num_repeats;
  Array1<int32_t> new2old_indexes;
  Ragged<int32_t> ans = UniqueSequences(
      any.Specialize<int32_t>(), need_num_repeats ? &num_repeats : nullptr,
      need_new2old_indexes ? &new2old_indexes : nullptr);

  torch::optional<RaggedAny> num_repeats_tensor;
  if (need_num_repeats) num_repeats_tensor = RaggedAny(num_repeats.Generic());

  torch::optional<torch::Tensor> new2old_indexes_tensor;
  if (need_new2old_indexes) new2old_indexes_tensor = ToTorch(new2old_indexes);

  return std::make_tuple(RaggedAny(ans.Generic()), num_repeats_tensor,
                         new2old_indexes_tensor);
}

RaggedAny RaggedAny::Normalize(bool use_log) /*const*/ {
  DeviceGuard guard(any.Context());
  RaggedAny out;
  NormalizeFunction::apply(*this, use_log, Data(), &out);
  return out;
}

torch::Tensor RaggedAny::Pad(const std::string &mode,
                             torch::IValue padding_value) /*const*/ {
  K2_CHECK(!padding_value.isNone());

  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    Array2<T> arr = PadRagged(any.Specialize<T>(), mode, padding_value.to<T>());
    return ToTorch(arr);
  });
  // Unreachable code
  return {};
}

torch::optional<torch::Tensor> RaggedAny::Sort(
    bool descending /*= false*/, bool need_new2old_indexes /*= false*/) {
  DeviceGuard guard(any.Context());
  Dtype t = any.GetDtype();

  Array1<int32_t> new2old;

  if (need_new2old_indexes) {
    new2old = Array1<int32_t>(any.Context(), any.NumElements());
  }

  FOR_REAL_AND_INT32_TYPES(t, T, {
    if (descending) {
      SortSublists<T, GreaterThan<T>>(
          &any.Specialize<T>(), need_new2old_indexes ? &new2old : nullptr);
    } else {
      SortSublists<T, LessThan<T>>(&any.Specialize<T>(),
                                   need_new2old_indexes ? &new2old : nullptr);
    }
  });

  torch::optional<torch::Tensor> ans;
  if (need_new2old_indexes) ans = ToTorch(new2old);
  return ans;
}

RaggedAny RaggedAny::Index(RaggedAny &indexes) /*const*/ {
  K2_CHECK_EQ(indexes.any.GetDtype(), kInt32Dtype)
      << "Unsupported dtype: " << TraitsOf(indexes.any.GetDtype()).Name();

  DeviceGuard guard(any.Context());

  bool remove_axis = false;
  Dtype t = any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return RaggedAny(k2::Index<T>(any.Specialize<T>(),
                                  indexes.any.Specialize<int32_t>(),
                                  remove_axis)
                         .Generic());
  });

  // Unreachable code
  return {};
}

std::pair<RaggedAny, torch::optional<torch::Tensor>> RaggedAny::Index(
    torch::Tensor indexes, int32_t axis,
    bool need_value_indexes /*= false*/) /*const*/ {
  DeviceGuard guard(any.Context());

  Array1<int32_t> indexes_array = FromTorch<int32_t>(indexes);

  Array1<int32_t> value_indexes;
  torch::optional<torch::Tensor> value_indexes_tensor;

  Dtype t = any.GetDtype();

  FOR_REAL_AND_INT32_TYPES(t, T, {
    Ragged<T> ans = k2::Index<T>(any.Specialize<T>(), axis, indexes_array,
                                 need_value_indexes ? &value_indexes : nullptr);

    if (need_value_indexes) value_indexes_tensor = ToTorch(value_indexes);

    return std::make_pair(RaggedAny(ans.Generic()), value_indexes_tensor);
  });

  // Unreachable code
  return {};
}

RaggedAny RaggedAny::Index(torch::Tensor src,
                           torch::IValue default_value /*= {}*/) /*const*/ {
  Dtype t = any.GetDtype();
  K2_CHECK_EQ(t, kInt32Dtype) << "Unsupported dtype: " << TraitsOf(t).Name();

  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();

  DeviceGuard guard(any.Context());
  Dtype dtype = ScalarTypeToDtype(src.scalar_type());
  FOR_REAL_AND_INT32_TYPES(dtype, T, {
    T value_for_minus_one =
        default_value.isNone() ? T() : default_value.to<T>();
    Array1<T> src_array = FromTorch<T>(src);
    return RaggedAny(
        k2::Index(src_array, any.Specialize<int32_t>(), value_for_minus_one)
            .Generic());
  });
  // Unreachable code
  return {};
}

torch::Tensor RaggedAny::IndexAndSum(torch::Tensor src) /*const*/ {
  DeviceGuard guard(any.Context());
  return IndexAndSumFunction::apply(src, *this);
}

}  // namespace k2
