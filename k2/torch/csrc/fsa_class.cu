/**
 * @brief A wrapper around FsaOrVec
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang, Fangjun Kuang)
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

#include <exception>
#include <string>
#include <vector>

#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

FsaClass FsaClass::FromUnaryFunctionTensor(FsaClass &src, const FsaOrVec &arcs,
                                           torch::Tensor arc_map) {
  FsaClass dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  dest.CopyTensorAttrs(src, arc_map);
  dest.CopyRaggedTensorAttrs(src, arc_map);
  return dest;
}

void FsaClass::CopyAttrs(FsaClass &src, torch::Tensor arc_map) {
  CopyTensorAttrs(src, arc_map);
  CopyRaggedTensorAttrs(src, arc_map);
}

void FsaClass::CopyTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  for (const auto &iter : src.tensor_attrs) {
    Dtype dtype = ConvertDtype(iter.second.scalar_type());
    FOR_REAL_AND_INT32_TYPES(dtype, T, {
      auto value = IndexSelect<T>(iter.second, arc_map, 0);
      SetTensorAttr(iter.first, value);
    });
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  Array1<int32_t> indexes_array = Array1FromTorch<int32_t>(arc_map);
  for (auto &iter : src.ragged_tensor_attrs) {
    Ragged<int32_t> ans =
        Index<int32_t>(iter.second, /*axis*/ 0, indexes_array, nullptr);
    SetRaggedTensorAttr(iter.first, ans);
  }
}

void FsaClass::SetScores(torch::Tensor scores) {
  K2_CHECK_EQ(scores.numel(), fsa.NumElements());
  K2_CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  K2_CHECK(ContextFromTensor(scores)->IsCompatible(*fsa.Context()));
  Scores().copy_(scores);
}

torch::Tensor FsaClass::Scores() {
  auto device = DeviceFromContext(fsa.Context());
  auto scalar_type = caffe2::TypeMeta::Make<float>();

  // an Arc has 4 members
  static_assert(sizeof(Arc) == 4 * sizeof(int32_t), "");

  std::vector<int64_t> sizes = {fsa.values.Dim(), 4};  // [num_rows, num_cols]
  std::vector<int64_t> strides = {4, 1};               // in number of elements
  auto options = torch::device(device).dtype(scalar_type);

  auto tmp_scores = torch::from_blob(
      fsa.values.Data(), sizes, strides,
      [saved_region = fsa.values.GetRegion()](void *) {}, options);
  return tmp_scores.index({"...", -1});
}

int32_t FsaClass::Properties() {
  if (properties == 0) {
    if (fsa.NumAxes() == 2) {
      properties = GetFsaBasicProperties(fsa);
    } else {
      GetFsaVecBasicProperties(fsa, nullptr, &properties);
    }
    if ((properties & kFsaPropertiesValid) != kFsaPropertiesValid) {
      K2_LOG(FATAL) << "Fsa is not valid, properties are : " << properties
                    << " = " << FsaPropertiesAsString(properties)
                    << ", arcs are : " << fsa;
    }
  }
  return properties;
}

torch::Tensor FsaClass::Labels() /*const*/ {
  auto device = DeviceFromContext(fsa.Context());
  auto scalar_type = caffe2::TypeMeta::Make<int32_t>();
  // an Arc has 4 members
  static_assert(sizeof(Arc) == 4 * sizeof(int32_t), "");

  std::vector<int64_t> sizes = {fsa.values.Dim(), 4};  // [num_rows, num_cols]
  std::vector<int64_t> strides = {4, 1};               // in number of elements
  auto options = torch::device(device).dtype(scalar_type);

  torch::Tensor arcs = torch::from_blob(
      fsa.values.Data(), sizes, strides,
      [saved_region = fsa.values.GetRegion()](void *) {}, options);

  return arcs.index({"...", 2});
}

void FsaClass::SetLabels(torch::Tensor labels) {
  K2_CHECK_EQ(labels.numel(), fsa.NumElements());
  K2_CHECK_EQ(labels.scalar_type(), torch::kInt32);
  K2_CHECK(ContextFromTensor(labels)->IsCompatible(*fsa.Context()));
  Labels().copy_(labels);
}

void FsaClass::SetAttr(const std::string &name, torch::IValue value) {
  if (name == "scores") {
    K2_CHECK(value.isTensor());
    SetScores(value.toTensor());
    return;
  }

  if (name == "labels") {
    K2_CHECK(value.isTensor());
    SetLabels(value.toTensor());
    return;
  }

  if (HasAttr(name)) DeleteAttr(name);

  if (value.isTensor()) {
    SetTensorAttr(name, value.toTensor());
  } else if (IsRaggedInt(value)) {
    SetRaggedTensorAttr(name, ToRaggedInt(value));
  } else {
    K2_LOG(FATAL) << "Unsupported type: " << value.tagKind()
                  << " for attribute '" << name
                  << "'.\nExpect torch::Tensor or Ragged<int32_t>";
  }
  all_attr_names.insert(name);
}

torch::IValue FsaClass::GetAttr(const std::string &name) /*const*/ {
  if (name == "scores") {
    return torch::IValue(Scores());
  }

  if (name == "labels") {
    return torch::IValue(Labels());
  }

  if (!HasAttr(name)) {
    std::ostringstream os;
    os << "No such attribute '" << name << "'";
    throw std::runtime_error(os.str().c_str());
  }

  {
    auto it = tensor_attrs.find(name);
    if (it != tensor_attrs.end()) {
      return torch::IValue(it->second);
    }
  }

  {
    auto it = ragged_tensor_attrs.find(name);
    if (it != ragged_tensor_attrs.end()) {
      return ToIValue(it->second);
    }
  }
  // Unreachable code
  K2_LOG(FATAL) << "Attribute not found, name : " << name;
  return torch::IValue();
}

void FsaClass::DeleteAttr(const std::string &name) {
  {
    auto it = all_attr_names.find(name);
    if (it != all_attr_names.end()) {
      all_attr_names.erase(it);
    } else {
      std::ostringstream os;
      os << "No such attribute '" << name << "'";
      throw std::runtime_error(os.str().c_str());
    }
  }

  {
    // Were we allowed to use C++ 17, could we use the following statement:
    // if (auto it = tensor_attrs.find(name); it != tensor_attrs.end()) {
    auto it = tensor_attrs.find(name);
    if (it != tensor_attrs.end()) {
      tensor_attrs.erase(it);
      return;
    }
  }

  {
    auto it = ragged_tensor_attrs.find(name);
    if (it != ragged_tensor_attrs.end()) {
      ragged_tensor_attrs.erase(it);
      return;
    }
  }
}

bool FsaClass::HasAttr(const std::string &name) const {
  // we treat labels & scores as attributes, though they don't store in
  // attribute containers.
  if (name == "scores" || name == "labels") return true;
  return all_attr_names.count(name) > 0;
}

FsaClass FsaClass::ToOtherContext(const ContextPtr &context) const {
  K2_CHECK(!context->IsCompatible(*fsa.Context()));
  FsaClass dest(fsa.To(context));
  auto device = DeviceFromContext(context);
  for (const auto &iter : tensor_attrs) {
    dest.SetTensorAttr(iter.first, iter.second.to(device));
  }
  for (const auto &iter : ragged_tensor_attrs) {
    dest.SetRaggedTensorAttr(iter.first, iter.second.To(context));
  }
  return dest;
}

FsaClass FsaClass::To(torch::Device device) const {
  ContextPtr context = fsa.Context();
  if (device.is_cpu()) {
    // CPU -> CPU
    if (context->GetDeviceType() == kCpu) return *this;

    // CUDA -> CPU
    DeviceGuard guard(context);
    return this->ToOtherContext(GetCpuContext());
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
  return this->ToOtherContext(GetCudaContext(device_index));
}

FsaClass FsaClass::To(const std::string &device) const {
  torch::Device d(device);
  return this->To(d);
}

}  // namespace k2
