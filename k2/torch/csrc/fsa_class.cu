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

FsaClass FsaClass::FromUnaryFunctionRagged(FsaClass &src, const FsaOrVec &arcs,
                                           Ragged<int32_t> &arc_map) {
  FsaClass dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  for (const auto &iter : src.tensor_attrs) {
    if (iter.second.scalar_type() == torch::kInt32) {
      torch::Tensor value = iter.second;
      auto masking =
          torch::logical_or(torch::ne(src.Labels(), -1), torch::ne(value, -1));
      // we need a int32_t scalar, so we have to use tensor.
      auto filler_scalar =
          torch::tensor(0, torch::dtype(torch::kInt32).device(value.device()));
      value = torch::where(masking, value, filler_scalar);
      Array1<int32_t> value_array = Array1FromTorch<int32_t>(value);
      auto new_value = Index(value_array, arc_map, /*default_value*/ 0);
      dest.SetRaggedTensorAttr(iter.first, RemoveValuesEq(new_value, 0));
    } else {
      K2_CHECK(iter.second.dtype() == torch::kFloat32 ||
               iter.second.dtype() == torch::kFloat64);

      Dtype dtype = ConvertDtype(iter.second.scalar_type());
      FOR_REAL_AND_INT32_TYPES(dtype, T, {
        Array1<T> src_array = Array1FromTorch<T>(iter.second);
        Ragged<T> s = Index<T>(src_array, arc_map, /*default_value*/ 0);
        Array1<T> ans_array(s.Context(), s.Dim0());
        SumPerSublist<T>(s, 0, &ans_array);
        dest.SetTensorAttr(iter.first, Array1ToTorch(ans_array));
      });
    }
  }
  dest.CopyRaggedTensorAttrs(src, arc_map);
  return dest;
}

FsaClass FsaClass::FromBinaryFunctionTensor(FsaClass &a_src, FsaClass &b_src,
                                            const FsaOrVec &arcs,
                                            torch::Tensor a_arc_map,
                                            torch::Tensor b_arc_map) {
  FsaClass dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  for (const auto &iter : a_src.tensor_attrs) {
    if (b_src.HasAttr(iter.first)) {
      if (iter.second.scalar_type() != torch::kFloat32) {
        std::ostringstream oss;
        oss << "We don't support propagating two "
            << "attributes with the same name that are "
            << "not real-valued, in intersection: " << iter.first;
        throw std::runtime_error(oss.str().c_str());
      }
      auto b_value = b_src.GetAttr(iter.first).toTensor();
      K2_CHECK_EQ(b_value.scalar_type(), torch::kFloat32);
      auto new_value =
          IndexSelect<float>(iter.second, a_arc_map, /*default_value*/ 0) +
          IndexSelect<float>(b_value, b_arc_map, /*default_value*/ 0);
      dest.SetTensorAttr(iter.first, new_value);
    } else {
      Dtype dtype = ConvertDtype(iter.second.scalar_type());
      FOR_REAL_AND_INT32_TYPES(dtype, T, {
        auto value =
            IndexSelect<T>(iter.second, a_arc_map, /*default_value*/ 0);
        dest.SetTensorAttr(iter.first, value);
      });
    }
  }
  dest.CopyRaggedTensorAttrs(a_src, a_arc_map);
  dest.CopyTensorAttrs(b_src, b_arc_map);
  dest.CopyRaggedTensorAttrs(b_src, b_arc_map);
  return dest;
}

void FsaClass::CopyTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  for (const auto &iter : src.tensor_attrs) {
    if (!HasAttr(iter.first)) {
      Dtype dtype = ConvertDtype(iter.second.scalar_type());
      FOR_REAL_AND_INT32_TYPES(dtype, T, {
        auto value = IndexSelect<T>(iter.second, arc_map, 0);
        SetTensorAttr(iter.first, value);
      });
    }
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  for (auto &iter : src.ragged_tensor_attrs) {
    if (!HasAttr(iter.first)) {
      Array1<int32_t> indexes_array = Array1FromTorch<int32_t>(arc_map);
      Ragged<int32_t> ans =
          Index<int32_t>(iter.second, /*axis*/ 0, indexes_array, nullptr);
      SetRaggedTensorAttr(iter.first, ans);
    }
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, Ragged<int32_t> &arc_map) {
  for (auto &iter : src.ragged_tensor_attrs) {
    if (!HasAttr(iter.first)) {
      Ragged<int32_t> new_value =
          Index<int32_t>(iter.second, arc_map, /*remove_axis*/ true);
      SetRaggedTensorAttr(iter.first, new_value);
    }
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
    if (properties & kFsaPropertiesValid != kFsaPropertiesValid) {
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

  all_attr_names.insert(name);

  if (value.isTensor()) {
    SetTensorAttr(name, value.toTensor());
    return;
  }

  if (IsRaggedInt(value)) {
    SetRaggedTensorAttr(name, ToRaggedInt(value));
    return;
  }

  K2_LOG(FATAL) << "Attribute type is not supported, name : " << name;
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

}  // namespace k2
