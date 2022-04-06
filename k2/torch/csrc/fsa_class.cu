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
  dest.CopyAttrs(src, arc_map);
  return dest;
}

void FsaClass::CopyAttrs(FsaClass &src, torch::Tensor arc_map) {
  CopyTensorAttrs(src, arc_map);
  CopyRaggedTensorAttrs(src, arc_map);
}

void FsaClass::CopyAttrs(std::vector<FsaClass> &srcs,
                         Ragged<int32_t> &arc_map) {
  K2_CHECK_EQ(fsa.NumAxes(), 3);
  CopyTensorAttrs(srcs, arc_map);
  CopyRaggedTensorAttrs(srcs, arc_map);
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

void FsaClass::CopyTensorAttrs(std::vector<FsaClass> &srcs,
                               Ragged<int32_t> &arc_map) {
  K2_CHECK_EQ(arc_map.NumAxes(), 2);
  K2_CHECK_EQ(arc_map.Dim0(), static_cast<int32_t>(srcs.size()));
  // Gather attributes info of all source fsas.
  std::unordered_map<std::string, Dtype> attrs_info;
  for (const auto &fsa : srcs) {
    for (const auto &iter : fsa.tensor_attrs) {
      Dtype dtype = ConvertDtype(iter.second.scalar_type());
      attrs_info.insert(std::make_pair(iter.first, dtype));
    }
  }
  std::vector<torch::Tensor> values;
  auto row_splits = arc_map.RowSplits(1).To(GetCpuContext());
  for (const auto &iter : attrs_info) {
    for (int32_t i = 0; i < static_cast<int32_t>(srcs.size()); ++i) {
      auto this_arc_map_array =
          arc_map.values.Arange(row_splits[i], row_splits[i + 1]);
      auto this_arc_map = Array1ToTorch<int32_t>(this_arc_map_array);
      if (srcs[i].HasTensorAttr(iter.first)) {
        auto attr = srcs[i].GetTensorAttr(iter.first);
        FOR_REAL_AND_INT32_TYPES(iter.second, T, {
          auto value = IndexSelect<T>(attr, this_arc_map, 0);
          values.emplace_back(value);
        });
      } else {
        FOR_REAL_AND_INT32_TYPES(iter.second, T, {
          auto opts = torch::dtype(ConvertDtype(iter.second))
                          .device(this_arc_map.device());
          auto value = torch::zeros(this_arc_map.numel(), opts);
          values.emplace_back(value);
        });
      }
    }
    SetTensorAttr(iter.first, torch::cat(values));
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  Array1<int32_t> indexes_array = Array1FromTorch<int32_t>(arc_map);
  for (auto &iter : src.ragged_tensor_attrs) {
    auto value = Index<int32_t>(iter.second, 0, indexes_array, nullptr);
    SetRaggedTensorAttr(iter.first, value);
  }
}

void FsaClass::CopyRaggedTensorAttrs(std::vector<FsaClass> &srcs,
                                     Ragged<int32_t> &arc_map) {
  K2_CHECK_EQ(arc_map.NumAxes(), 2);
  K2_CHECK_EQ(arc_map.Dim0(), static_cast<int32_t>(srcs.size()));
  std::unordered_set<std::string> attrs_name;
  for (const auto &fsa : srcs) {
    for (const auto &iter : fsa.ragged_tensor_attrs) {
      attrs_name.insert(iter.first);
    }
  }
  std::vector<Ragged<int32_t>> values;
  auto row_splits = arc_map.RowSplits(1).To(GetCpuContext());
  for (const auto &name : attrs_name) {
    for (int32_t i = 0; i < static_cast<int32_t>(srcs.size()); ++i) {
      auto this_arc_map =
          arc_map.values.Arange(row_splits[i], row_splits[i + 1]);
      if (srcs[i].HasRaggedTensorAttr(name)) {
        auto attr = srcs[i].GetRaggedTensorAttr(name);
        auto value = Index<int32_t>(attr, 0 /*axis*/, this_arc_map);
        values.emplace_back(value);
      } else {
        auto empty_shape =
            RegularRaggedShape(this_arc_map.Context(), this_arc_map.Dim(), 0);
        auto value = Ragged<int32_t>(empty_shape);
        values.emplace_back(value);
      }
    }
    SetRaggedTensorAttr(name, Cat(0 /*axis*/, values.size(), values.data()));
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
                    << " = " << FsaPropertiesAsString(properties);
    }
  }
  return properties;
}

torch::Tensor FsaClass::Labels() {
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
  properties = 0;  // Clear cached properties as we changed the labels
}

}  // namespace k2
