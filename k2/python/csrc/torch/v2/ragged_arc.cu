/**
 * @brief A wrapper around Ragged<Arc> and torch::Tensor
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

#include <exception>
#include <string>
#include <vector>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/autograd/arc_sort.h"
#include "k2/python/csrc/torch/v2/autograd/get_forward_scores.h"
#include "k2/python/csrc/torch/v2/autograd/index_select.h"
#include "k2/python/csrc/torch/v2/autograd/utils.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {

RaggedArc::RaggedArc(
    const std::string &s,
    const std::vector<std::string> &extra_label_names /*= {}*/) {
  // TODO: pass following options from arguments
  bool openfst = false;
  int32_t num_extra_labels = 0;
  Array2<int32_t> extra_labels;
  Array2<int32_t> *p_extra_labels;
  int32_t num_ragged_labels = 0;
  Ragged<int32_t> *ragged_labels = nullptr;

  if (!extra_label_names.empty()) {
    num_extra_labels = extra_label_names.size();
    p_extra_labels = &extra_labels;
  }

  fsa = FsaFromString(s, openfst, num_extra_labels, p_extra_labels,
                      num_ragged_labels, ragged_labels);

  if (num_extra_labels) {
    for (int32_t i = 0; i != num_extra_labels; ++i) {
      const auto &name = extra_label_names[i];
      Array1<int32_t> row = extra_labels.Row(i);
      tensor_attrs[name] = ToTorch(row);
      all_attr_names.insert(name);
    }
  }
  // Check the validation of this fsa, will trigger a fatal error if this fsa
  // is not valid.
  Properties();

  // TODO: we also need to pass the name of extra_labels and ragged_labels.
}

RaggedArc RaggedArc::FromUnaryFunctionTensor(const RaggedArc &src,
                                             const Ragged<Arc> &arcs,
                                             torch::Tensor arc_map) {
  RaggedArc dest(arcs);

  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();

  dest.CopyTensorAttrs(src, arc_map);

  dest.CopyRaggedTensorAttrs(src, arc_map);

  dest.CopyOtherAttrs(src);

  PhantomIndexSelectScoresFunction::apply(dest, src.Scores(), arc_map);
  return dest;
}

RaggedArc RaggedArc::FromUnaryFunctionRagged(RaggedArc &src,
                                             const Ragged<Arc> &arcs,
                                             Ragged<int32_t> &arc_map,
                                             bool remove_filler /*= true*/) {
  RaggedArc dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();

  RaggedAny arc_map_any = RaggedAny(arc_map.Generic());

  for (const auto &iter : src.tensor_attrs) {
    if (remove_filler && iter.second.scalar_type() == torch::kInt32) {
      int32_t filler = (int32_t)src.GetFiller(iter.first);
      if (filler != -1) {
        torch::Tensor value = iter.second.clone();
        auto masking = torch::logical_or(torch::ne(src.Labels(), -1),
                                         torch::ne(value, -1));
        auto filler_scalar = torch::tensor(
            filler, torch::dtype(torch::kInt32).device(value.device()));
        value = torch::where(masking, value, filler_scalar);
        auto new_value = arc_map_any.Index(value, py::int_(filler));
        dest.SetAttr(iter.first, new_value.RemoveValuesEq(py::int_(filler)));
      }
    } else {
      K2_CHECK(iter.second.dtype() == torch::kFloat32 ||
               iter.second.dtype() == torch::kFloat64);
      torch::Tensor new_value = arc_map_any.IndexAndSum(iter.second);
      dest.SetAttr(iter.first, new_value);
    }
  }

  dest.CopyRaggedTensorAttrs(src, arc_map_any);

  dest.CopyOtherAttrs(src);

  PhantomIndexAndSumScoresFunction::apply(dest, src.Scores(), arc_map);
  return dest;
}

RaggedArc RaggedArc::FromBinaryFunctionTensor(const RaggedArc &a_src,
                                              const RaggedArc &b_src,
                                              const Ragged<Arc> &arcs,
                                              torch::Tensor a_arc_map,
                                              torch::Tensor b_arc_map) {
  RaggedArc dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  for (const auto &iter : a_src.tensor_attrs) {
    float filler = a_src.GetFiller(iter.first);
    if (b_src.HasAttr(iter.first)) {
      if (iter.second.scalar_type() != torch::kFloat32) {
        // TODO: raise an exception
        K2_LOG(WARNING) << "We don't support propagating two "
                        << "attributes with the same name that are "
                        << "not real-valued, in intersection: " << iter.first;
        continue;
      }
      auto b_value = b_src.GetAttr(iter.first).cast<torch::Tensor>();
      K2_CHECK_EQ(b_value.scalar_type(), torch::kFloat32);
      auto new_value =
          IndexSelectFunction::apply(iter.second, a_arc_map, filler) +
          IndexSelectFunction::apply(b_value, b_arc_map, filler);
      dest.SetAttr(iter.first, new_value);
    } else {
      auto new_value =
          IndexSelectFunction::apply(iter.second, a_arc_map, filler);
      dest.SetAttr(iter.first, new_value);
    }
  }

  dest.CopyRaggedTensorAttrs(a_src, a_arc_map);

  dest.CopyOtherAttrs(a_src);

  dest.CopyTensorAttrs(b_src, b_arc_map, false);

  dest.CopyRaggedTensorAttrs(b_src, b_arc_map, false);

  dest.CopyOtherAttrs(b_src, false);

  // The following will actually overwrite `scores` with the same
  // value it had before; but this enables the autograd to work since
  // we do it using torch mechanisms.
  dest.SetScores(IndexSelectFunction::apply(a_src.Scores(), a_arc_map, 0) +
                 IndexSelectFunction::apply(b_src.Scores(), b_arc_map, 0));
  return dest;
}

void RaggedArc::CopyTensorAttrs(const RaggedArc &src, torch::Tensor arc_map,
                                bool over_write /*= true*/) {
  for (const auto &iter : src.tensor_attrs) {
    if (over_write || !HasAttr(iter.first)) {
      float filler = GetFiller(iter.first);
      auto value = IndexSelectFunction::apply(iter.second, arc_map, filler);
      SetAttr(iter.first, value);
    }
  }
}

void RaggedArc::CopyTensorAttrs(std::vector<RaggedArc> &srcs) {
  std::unordered_set<std::string> tensor_attr_names;
  for (const auto &fsa : srcs)
    for (const auto &attr : fsa.tensor_attrs)
      tensor_attr_names.insert(attr.first);

  std::vector<torch::Tensor> values;
  for (const auto &name : tensor_attr_names) {
    for (const auto &fsa : srcs) {
      auto iter = fsa.tensor_attrs.find(name);
      K2_CHECK(iter != fsa.tensor_attrs.end());
      values.emplace_back(iter->second);
    }
    torch::Tensor value = torch::cat(values, 0);
    SetAttr(name, value);
  }
}

void RaggedArc::CopyOtherAttrs(const RaggedArc &src,
                               bool over_write /*= true*/) {
  for (const auto &iter : src.other_attrs) {
    if (over_write || !HasAttr(iter.first)) SetAttr(iter.first, iter.second);
  }
}

void RaggedArc::CopyOtherAttrs(std::vector<RaggedArc> &srcs) {
  std::unordered_set<std::string> other_attr_names;
  for (const auto &fsa : srcs) {
    for (const auto &attr : fsa.other_attrs) {
      other_attr_names.insert(attr.first);
    }
  }
  for (const auto &name : other_attr_names) {
    for (const auto &fsa : srcs) {
      auto iter = fsa.other_attrs.find(name);
      if (iter != fsa.other_attrs.end()) {
        auto self_iter = other_attrs.find(name);
        if (self_iter != other_attrs.end()) {
          // TODO: Check the values iter & self_iter pointing to are identical.
        } else {
          SetAttr(name, iter->second);
        }
      }
    }
  }
}

void RaggedArc::CopyRaggedTensorAttrs(const RaggedArc &src,
                                      torch::Tensor arc_map,
                                      bool over_write /*= true*/) {
  for (const auto &iter : src.ragged_tensor_attrs) {
    if (over_write || !HasAttr(iter.first)) {
      // Only integer types ragged attributes are supported now
      K2_CHECK_EQ(iter.second.any.GetDtype(), kInt32Dtype);
      auto new_value =
          const_cast<RaggedAny &>(iter.second).Index(arc_map, 0, false);
      SetAttr(iter.first, new_value.first);
    }
  }
}

void RaggedArc::CopyRaggedTensorAttrs(const RaggedArc &src, RaggedAny &arc_map,
                                      bool over_write /*= true*/) {
  for (const auto &iter : src.ragged_tensor_attrs) {
    if (over_write || !HasAttr(iter.first)) {
      // We currently don't support float ragged attributes
      K2_CHECK_EQ(iter.second.any.GetDtype(), kInt32Dtype);
      RaggedAny new_value = const_cast<RaggedAny &>(iter.second).Index(arc_map);
      new_value = new_value.RemoveAxis(new_value.any.NumAxes() - 2);
      SetAttr(iter.first, new_value);
    }
  }
}

void RaggedArc::CopyRaggedTensorAttrs(std::vector<RaggedArc> &srcs) {
  std::unordered_set<std::string> tensor_attr_names;
  for (const auto &fsa : srcs)
    for (const auto &attr : fsa.ragged_tensor_attrs)
      tensor_attr_names.insert(attr.first);

  std::vector<RaggedAny> values;
  for (const auto &name : tensor_attr_names) {
    for (const auto &fsa : srcs) {
      auto iter = fsa.ragged_tensor_attrs.find(name);
      K2_CHECK(iter != fsa.ragged_tensor_attrs.end());
      values.emplace_back(iter->second);
    }
    RaggedAny value = RaggedAny::Cat(values, 0);
    SetAttr(name, value);
  }
}

void RaggedArc::SetScores(torch::Tensor scores) {
  K2_CHECK_EQ(scores.numel(), fsa.NumElements());
  Scores().copy_(scores.detach());
  PhantomSetScoresFunction::apply(*this, scores);
}

torch::Tensor &RaggedArc::Scores() {
  if (!scores.defined()) {
    auto device_type = ToTorchDeviceType(fsa.Context()->GetDeviceType());
    int32_t device_id = fsa.Context()->GetDeviceId();
    auto device = torch::Device(device_type, device_id);
    auto scalar_type = ToScalarType<float>::value;
    // an Arc has 4 members
    static_assert(sizeof(Arc) == 4 * sizeof(int32_t), "");

    std::vector<int64_t> sizes = {fsa.values.Dim(), 4};  // [num_rows, num_cols]
    std::vector<int64_t> strides = {4, 1};  // in number of elements
    auto options = torch::device(device).dtype(scalar_type);

    auto tmp_scores = torch::from_blob(
        fsa.values.Data(), sizes, strides,
        [saved_region = fsa.values.GetRegion()](void *) {}, options);
    scores = tmp_scores.index({"...", -1});
  }
  return scores;
}

const torch::Tensor &RaggedArc::Scores() const {
  return const_cast<RaggedArc *>(this)->Scores();
}

int32_t RaggedArc::Properties() {
  if (properties == 0) {
    if (fsa.NumAxes() == 2) {
      properties = GetFsaBasicProperties(fsa);
    } else {
      GetFsaVecBasicProperties(fsa, nullptr, &properties);
    }
    if (properties & 1 != 1) {
      K2_LOG(FATAL) << "Fsa is not valid, properties are : " << properties
                    << " = " << PropertiesStr() << ", arcs are : " << fsa;
    }
  }
  return properties;
}

std::string RaggedArc::PropertiesStr() const {
  return FsaPropertiesAsString(const_cast<RaggedArc *>(this)->Properties());
}

torch::Tensor RaggedArc::Arcs() {
  auto device_type = ToTorchDeviceType(fsa.Context()->GetDeviceType());
  int32_t device_id = fsa.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<int32_t>::value;
  // an Arc has 4 members
  static_assert(sizeof(Arc) == 4 * sizeof(int32_t), "");

  std::vector<int64_t> sizes = {fsa.values.Dim(), 4};  // [num_rows, num_cols]
  std::vector<int64_t> strides = {4, 1};               // in number of elements
  auto options = torch::device(device).dtype(scalar_type);

  return torch::from_blob(
      fsa.values.Data(), sizes, strides,
      [saved_region = fsa.values.GetRegion()](void *) {}, options);
}

torch::Tensor RaggedArc::Labels() /*const*/ { return Arcs().index({"...", 2}); }

void RaggedArc::SetLabels(torch::Tensor labels) {
  K2_CHECK_EQ(labels.numel(), fsa.NumElements());
  Arcs().index({"...", 2}).copy_(labels);
}

RaggedArc &RaggedArc::SetRequiresGrad(bool requires_grad /*=true*/) {
  Scores().requires_grad_(requires_grad);
  return *this;
}

RaggedArc RaggedArc::To(const ContextPtr &context) const {
  RaggedArc dest(fsa.To(context));
  auto device = GetDevice(context);
  for (const auto &iter : tensor_attrs) {
    dest.SetAttr(iter.first, (iter.second).to(device));
  }
  for (const auto &iter : ragged_tensor_attrs) {
    dest.SetAttr(iter.first, (iter.second).To(device));
  }
  for (const auto &iter : other_attrs) {
    dest.SetAttr(iter.first, iter.second);
  }
  // The following is a magic invocation to make sure
  // the backprop happens.
  PhantomSetScoresFunction::apply(dest, Scores().to(device));
  return dest;
}

RaggedArc RaggedArc::To(torch::Device device) const {
  ContextPtr context = fsa.Context();
  if (device.is_cpu()) {
    // CPU -> CPU
    if (context->GetDeviceType() == kCpu) return *this;

    // CUDA -> CPU
    DeviceGuard guard(context);
    return this->To(GetCpuContext());
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
  return this->To(GetCudaContext(device_index));
}

RaggedArc RaggedArc::To(const std::string &device) const {
  torch::Device d(device);
  return this->To(d);
}

std::string RaggedArc::ToStringSimple() const {
  std::ostringstream os;
  // TODO: Handle aux_label
  if (fsa.NumAxes() == 2) {
    os << FsaToString(fsa, /*openfst*/ false,
                      /*num_extra_labels*/ 0,
                      /*extra_labels*/ nullptr,
                      /*num_ragged_labels*/ 0,
                      /*ragged_labels*/ nullptr);
  } else {
    for (int32_t i = 0; i < fsa.Dim0(); ++i) {
      Ragged<Arc> sub_fsa = fsa.Index(0, i);
      os << FsaToString(sub_fsa, /*openfst*/ false,
                        /*num_extra_labels*/ 0,
                        /*extra_labels*/ nullptr,
                        /*num_ragged_labels*/ 0,
                        /*ragged_labels*/ nullptr);
    }
  }
  return os.str();
}

std::string RaggedArc::ToString() const {
  std::ostringstream os;
  std::vector<Array1<int32_t>> extra_labels;
  std::vector<Ragged<int32_t>> ragged_labels;
  for (auto &p : tensor_attrs) {
    if (p.second.scalar_type() == torch::kInt) {
      extra_labels.push_back(
          FromTorch<int32_t>(const_cast<torch::Tensor &>(p.second)));
    }
  }
  for (const auto &p : ragged_tensor_attrs) {
    if (p.second.any.GetDtype() == kInt32Dtype) {
      ragged_labels.push_back(p.second.any.Specialize<int32_t>());
    }
  }
  if (fsa.NumAxes() == 2) {
    os << "k2.Fsa: "
       << FsaToString(fsa, /*openfst*/ false,
                      /*num_extra_labels*/ extra_labels.size(),
                      /*extra_labels*/ extra_labels.data(),
                      /*num_ragged_labels*/ ragged_labels.size(),
                      /*ragged_labels*/ ragged_labels.data());
  } else {
    os << "k2.FsaVec: \n";
    for (int32_t i = 0; i < fsa.Dim0(); ++i) {
      Ragged<Arc> sub_fsa = fsa.Index(0, i);
      int32_t start = sub_fsa.values.Data() - fsa.values.Data(),
              end = start + sub_fsa.values.Dim();

      std::vector<Array1<int32_t>> sub_extra_labels;
      for (auto &v : extra_labels)
        sub_extra_labels.emplace_back(v.Arange(start, end));

      std::vector<Ragged<int32_t>> sub_ragged_labels;
      for (auto &v : ragged_labels)
        sub_ragged_labels.emplace_back(Arange(v, 0, start, end));

      os << "FsaVec[ " << i << " ]: "
         << FsaToString(sub_fsa, /*openfst*/ false,
                        /*num_extra_labels*/ sub_extra_labels.size(),
                        /*extra_labels*/ sub_extra_labels.data(),
                        /*num_ragged_labels*/ sub_ragged_labels.size(),
                        /*ragged_labels*/ sub_ragged_labels.data());
    }
  }

  os << "properties_str = " << PropertiesStr() << ".";
  for (const auto &v : tensor_attrs) os << "\n" << v.first << ": " << v.second;
  for (const auto &v : ragged_tensor_attrs) {
    os << "\n"
       << v.first << ": " << const_cast<RaggedAny &>(v.second).ToString();
  }
  for (const auto &v : other_attrs) os << "\n" << v.first << ": " << v.second;

  return os.str();
}

RaggedArc RaggedArc::CreateFsaVec(std::vector<RaggedArc> &fsas) {
  DeviceGuard guard(fsas[0].fsa.Context());
  std::vector<Fsa *> tmp_fsas;
  std::vector<torch::Tensor> tmp_scores;

  tmp_fsas.reserve(fsas.size());
  for (auto &f : fsas) {
    K2_CHECK_EQ(f.fsa.NumAxes(), 2);
    tmp_fsas.push_back(&f.fsa);
    tmp_scores.push_back(f.Scores());
  }
  FsaVec fsa_vec = k2::CreateFsaVec(tmp_fsas.size(), tmp_fsas.data());

  // TODO(fangjun): Don't handle scores specially, treat it
  // like other tensor attributes
  torch::Tensor scores = torch::cat(tmp_scores, 0);

  RaggedArc dest = RaggedArc(fsa_vec, scores);

  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();

  dest.CopyTensorAttrs(fsas);

  dest.CopyRaggedTensorAttrs(fsas);

  dest.CopyOtherAttrs(fsas);
  return dest;
}

RaggedArc RaggedArc::AddEpsilonSelfLoops() /*const*/ {
  DeviceGuard guard(fsa.Context());
  Array1<int32_t> arc_map;
  Ragged<Arc> arcs;
  k2::AddEpsilonSelfLoops(fsa, &arcs, &arc_map);
  return FromUnaryFunctionTensor(*this, arcs, ToTorch<int32_t>(arc_map));
}

RaggedArc RaggedArc::ArcSort() /*const*/ {
  if ((Properties() & kFsaPropertiesArcSorted) != 0) return *this;

  DeviceGuard guard(fsa.Context());
  Array1<int32_t> arc_map;
  Ragged<Arc> arcs;
  k2::ArcSort(fsa, &arcs, &arc_map);
  return FromUnaryFunctionTensor(*this, arcs, ToTorch<int32_t>(arc_map));
}

RaggedArc RaggedArc::Connect() /*const*/ {
  if ((Properties() & kFsaPropertiesMaybeAccessible) != 0 &&
      (Properties() & kFsaPropertiesMaybeCoaccessible) != 0)
    return *this;

  DeviceGuard guard(fsa.Context());
  Array1<int32_t> arc_map;
  Ragged<Arc> out;
  k2::Connect(fsa, &out, &arc_map);
  RaggedArc dest =
      FromUnaryFunctionTensor(*this, out, ToTorch<int32_t>(arc_map));
  return dest;
}

RaggedArc RaggedArc::TopSort() /*const*/ {
  if ((Properties() & kFsaPropertiesTopSorted) != 0) return *this;

  DeviceGuard guard(fsa.Context());
  Array1<int32_t> arc_map;
  Ragged<Arc> out;
  k2::TopSort(fsa, &out, &arc_map);
  RaggedArc dest =
      FromUnaryFunctionTensor(*this, out, ToTorch<int32_t>(arc_map));
  return dest;
}

void RaggedArc::SetAttr(const std::string &name, py::object value) {
  if (name == "grad") {
    // Note we don't use pybind11's def_property since it does not allow
    // to use argument annotions, which means it is not possible to
    // run: fsa.grad = None
    if (value.is_none()) {
      Scores().mutable_grad() = {};
    } else {
      Scores().mutable_grad() = value.cast<torch::Tensor>();
    }
    return;
  }

  if (HasAttr(name)) DeleteAttr(name);

  all_attr_names.insert(name);

  if (THPVariable_Check(value.ptr())) {
    torch::Tensor tensor = value.cast<torch::Tensor>();
    SetAttr(name, tensor);
    return;
  }

  try {
    RaggedAny ragged_tensor = value.cast<RaggedAny>();
    SetAttr(name, ragged_tensor);
    return;
  } catch (const py::cast_error &) {
    // do nothing.
  }

  all_attr_names.insert(name);
  other_attrs[name] = value;
}

py::object RaggedArc::GetAttr(const std::string &name) const {
  if (name == "grad") {
    return py::cast(Scores().grad());
  }

  if (!HasAttr(name)) {
    std::ostringstream os;
    os << "No such attribute '" << name << "'";
    // It's safe to use c_str() here as it is copied inside PyErr_SetString()
    //
    // See https://github.com/python/cpython/blob/main/Python/errors.c#L234
    PyErr_SetString(PyExc_AttributeError, os.str().c_str());
    throw py::error_already_set();
  }

  {
    auto it = tensor_attrs.find(name);
    if (it != tensor_attrs.end()) {
      return py::cast(it->second);
    }
  }

  {
    auto it = ragged_tensor_attrs.find(name);
    if (it != ragged_tensor_attrs.end()) {
      return py::cast(it->second);
    }
  }

  return other_attrs.at(name);
}

void RaggedArc::DeleteAttr(const std::string &name) {
  {
    auto it = all_attr_names.find(name);
    if (it != all_attr_names.end()) {
      all_attr_names.erase(it);
    } else {
      std::ostringstream os;
      os << "No such attribute '" << name << "'";
      // It's safe to use c_str() here as it is copied inside
      // PyErr_SetString()
      //
      // See https://github.com/python/cpython/blob/main/Python/errors.c#L234
      PyErr_SetString(PyExc_AttributeError, os.str().c_str());
      throw py::error_already_set();
    }
  }

  {
    // Erase the filler
    auto it = fillers.find(name);
    if (it != fillers.end()) {
      fillers.erase(it);
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

  {
    auto it = other_attrs.find(name);
    if (it != other_attrs.end()) {
      other_attrs.erase(it);
      return;
    }
  }
}

bool RaggedArc::HasAttr(const std::string &name) const {
  return all_attr_names.count(name) > 0;
}

void RaggedArc::SetFiller(const std::string &name, float filler) {
  fillers[name] = filler;
}

float RaggedArc::GetFiller(const std::string &name) const {
  auto iter = fillers.find(name);
  if (iter != fillers.end()) {
    return iter->second;
  } else {
    return 0;
  }
}

Ragged<int32_t> RaggedArc::GetStateBatches(bool transpose /*= true*/) {
  std::string name;
  if (transpose) {
    name = "state_batches_true";
  } else {
    name = "state_batches_false";
  }
  auto it = cached_ragged_tensor.find(name);
  if (it != cached_ragged_tensor.end()) {
    return it->second;
  }

  Ragged<int32_t> value = k2::GetStateBatches(fsa, transpose);
  cached_ragged_tensor[name] = value;
  return value;
}

Array1<int32_t> RaggedArc::GetDestStates(bool as_idx01) {
  std::string name;
  if (as_idx01) {
    name = "dest_states_true";
  } else {
    name = "dest_states_false";
  }
  auto it = cached_tensor.find(name);
  if (it != cached_tensor.end()) {
    return it->second;
  }

  Array1<int32_t> value = k2::GetDestStates(fsa, as_idx01);
  cached_tensor[name] = value;
  return value;
}

Ragged<int32_t> RaggedArc::GetIncomingArcs() {
  std::string name = "incoming_arcs";
  auto it = cached_ragged_tensor.find(name);
  if (it != cached_ragged_tensor.end()) {
    return it->second;
  }

  Array1<int32_t> dest_states = GetDestStates(/*as_idx01*/ true);
  Ragged<int32_t> value = k2::GetIncomingArcs(fsa, dest_states);
  cached_ragged_tensor[name] = value;
  return value;
}

Ragged<int32_t> RaggedArc::GetEnteringArcIndexBatches() {
  std::string name = "entering_arc_index_batches";
  auto it = cached_ragged_tensor.find(name);
  if (it != cached_ragged_tensor.end()) {
    return it->second;
  }

  Ragged<int32_t> incoming_arcs = GetIncomingArcs();
  Ragged<int32_t> state_batches = GetStateBatches(/*transpose*/ true);
  Ragged<int32_t> value =
      k2::GetEnteringArcIndexBatches(fsa, incoming_arcs, state_batches);
  cached_ragged_tensor[name] = value;
  return value;
}

Array1<int32_t> RaggedArc::GetEnteringArcs(bool use_double_scores) {
  std::string name = "entering_arcs";
  auto it = cached_tensor.find(name);
  if (it != cached_tensor.end()) {
    return it->second;
  }

  // This function will compute and fill in cached_tensor[name]
  (void)GetForwardScores(use_double_scores, /*log_semiring*/ false);

  return cached_tensor.at(name);
}

Ragged<int32_t> RaggedArc::GetLeavingArcIndexBatches() {
  std::string name = "leaving_arc_index_batches";
  auto it = cached_ragged_tensor.find(name);
  if (it != cached_ragged_tensor.end()) {
    return it->second;
  }

  Ragged<int32_t> state_batches = GetStateBatches(/*transpose*/ true);
  Ragged<int32_t> value = k2::GetLeavingArcIndexBatches(fsa, state_batches);
  cached_ragged_tensor[name] = value;
  return value;
}

// TODO(fangjun): Implement autograd for get forward scores
torch::Tensor RaggedArc::GetForwardScoresImpl(bool use_double_scores,
                                              bool log_semiring) {
  Array1<int32_t> entering_arcs;

  Array1<int32_t> *p_entering_arcs = nullptr;
  if (!log_semiring) {
    p_entering_arcs = &entering_arcs;
  }

  Ragged<int32_t> state_batches = GetStateBatches(/*transpose*/ true);
  Ragged<int32_t> entering_arc_batches = GetEnteringArcIndexBatches();
  Dtype t = kFloatDtype;
  if (use_double_scores) {
    t = kDoubleDtype;
  }

  FOR_REAL_TYPES(t, T, {
    Array1<T> forward_scores =
        k2::GetForwardScores<T>(fsa, state_batches, entering_arc_batches,
                                log_semiring, p_entering_arcs);
    if (!log_semiring) {
      cached_tensor["entering_arcs"] = entering_arcs;
    }
    return ToTorch(forward_scores);
  });

  // Unreachable code
  return {};
}

torch::Tensor RaggedArc::GetForwardScores(bool use_double_scores,
                                          bool log_semiring) {
  return GetForwardScoresFunction::apply(*this, use_double_scores, log_semiring,
                                         Scores());
};

}  // namespace k2
