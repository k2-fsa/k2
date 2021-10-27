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
#include "k2/torch/csrc/autograd/get_forward_scores.h"
#include "k2/torch/csrc/autograd/index_select.h"
#include "k2/torch/csrc/autograd/utils.h"
#include "k2/torch/csrc/fsa_class.h"

namespace k2 {

FsaClass::FsaClass(const std::string &s,
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

  // TODO: we also need to pass the name of ragged_labels.
}

FsaClass::FsaClass(const Ragged<Arc> &fsa, torch::Tensor aux_labels)
    : fsa(fsa) {
  K2_CHECK_EQ(fsa.NumElements(), aux_labels.numel());
  K2_CHECK_EQ(aux_labels.scalar_type(), torch::kInt32);
  SetTensorAttr("aux_labels", aux_labels);
}

FsaClass::FsaClass(const Ragged<Arc> &fsa, RaggedAny &aux_labels) : fsa(fsa) {
  K2_CHECK_EQ(fsa.NumElements(), aux_labels.any.Dim0());
  K2_CHECK_EQ(aux_labels.any.GetDtype(), kInt32Dtype);
  SetRaggedTensorAttr("aux_labels", aux_labels);
}

FsaClass FsaClass::FromUnaryFunctionTensor(FsaClass &src,
                                           const Ragged<Arc> &arcs,
                                           torch::Tensor arc_map) {
  FsaClass dest(arcs);

  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();

  dest.CopyTensorAttrs(src, arc_map);

  dest.CopyRaggedTensorAttrs(src, arc_map);

  dest.CopyOtherAttrs(src);

  PhantomIndexSelectScoresFunction::apply(dest, src.Scores(), arc_map);
  return dest;
}

FsaClass FsaClass::FromUnaryFunctionRagged(FsaClass &src,
                                           const Ragged<Arc> &arcs,
                                           Ragged<int32_t> &arc_map,
                                           bool remove_filler /*= true*/) {
  FsaClass dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();

  RaggedAny arc_map_any = RaggedAny(arc_map.Generic());

  for (const auto &iter : src.tensor_attrs) {
    if (remove_filler && iter.second.scalar_type() == torch::kInt32) {
      auto filler = src.GetFiller(iter.first);
      K2_CHECK(filler.isInt());
      if (filler.toInt() != -1) {
        torch::Tensor value = iter.second.clone();
        auto masking = torch::logical_or(torch::ne(src.Labels(), -1),
                                         torch::ne(value, -1));
        // we need a int32_t scalar, so we have to use tensor.
        auto filler_scalar = torch::tensor(
            filler.toInt(), torch::dtype(torch::kInt32).device(value.device()));
        value = torch::where(masking, value, filler_scalar);
        auto new_value = arc_map_any.Index(value, filler);
        dest.SetRaggedTensorAttr(iter.first, new_value.RemoveValuesEq(filler));
      }
    } else {
      K2_CHECK(iter.second.dtype() == torch::kFloat32 ||
               iter.second.dtype() == torch::kFloat64);
      torch::Tensor new_value = arc_map_any.IndexAndSum(iter.second);
      dest.SetTensorAttr(iter.first, new_value);
    }
  }

  dest.CopyRaggedTensorAttrs(src, arc_map_any);

  dest.CopyOtherAttrs(src);

  PhantomIndexAndSumScoresFunction::apply(dest, src.Scores(), arc_map);
  return dest;
}

FsaClass FsaClass::FromBinaryFunctionTensor(FsaClass &a_src, FsaClass &b_src,
                                            const Ragged<Arc> &arcs,
                                            torch::Tensor a_arc_map,
                                            torch::Tensor b_arc_map) {
  FsaClass dest(arcs);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  for (const auto &iter : a_src.tensor_attrs) {
    auto filler_ivalue = a_src.GetFiller(iter.first);
    float filler = filler_ivalue.isInt() ? filler_ivalue.toInt()
                                         : filler_ivalue.toDouble();
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
          IndexSelectFunction::apply(iter.second, a_arc_map, filler) +
          IndexSelectFunction::apply(b_value, b_arc_map, filler);
      dest.SetTensorAttr(iter.first, new_value);
    } else {
      auto new_value =
          IndexSelectFunction::apply(iter.second, a_arc_map, filler);
      dest.SetTensorAttr(iter.first, new_value);
    }
  }

  dest.CopyRaggedTensorAttrs(a_src, a_arc_map);

  dest.CopyOtherAttrs(a_src);

  dest.CopyTensorAttrs(b_src, b_arc_map);

  dest.CopyRaggedTensorAttrs(b_src, b_arc_map);

  dest.CopyOtherAttrs(b_src);

  // The following will actually overwrite `scores` with the same
  // value it had before; but this enables the autograd to work since
  // we do it using torch mechanisms.
  dest.SetScores(IndexSelectFunction::apply(a_src.Scores(), a_arc_map, 0) +
                 IndexSelectFunction::apply(b_src.Scores(), b_arc_map, 0));
  return dest;
}

void FsaClass::CopyTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  for (const auto &iter : src.tensor_attrs) {
    if (!HasAttr(iter.first)) {
      auto filler_ivalue = GetFiller(iter.first);
      float filler = filler_ivalue.isInt() ? filler_ivalue.toInt()
                                           : filler_ivalue.toDouble();
      auto value = IndexSelectFunction::apply(iter.second, arc_map, filler);
      SetTensorAttr(iter.first, value);
    }
  }
}

void FsaClass::CopyTensorAttrs(FsaClass &src, int32_t start, int32_t end) {
  K2_CHECK_EQ(fsa.NumAxes(), 3);
  K2_CHECK_GE(start, 0);
  K2_CHECK_GE(end, start);
  K2_CHECK_LT(end, fsa.NumElements());
  for (const auto &iter : src.tensor_attrs) {
    auto value = (iter.second).index({torch::indexing::Slice(start, end)});
    SetTensorAttr(iter.first, value);
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, int32_t start,
                                     int32_t end) {
  K2_CHECK_EQ(fsa.NumAxes(), 3);
  K2_CHECK_GE(start, 0);
  K2_CHECK_GE(end, start);
  K2_CHECK_LT(end, fsa.NumElements());
  for (auto &iter : src.ragged_tensor_attrs) {
    auto value = (iter.second).Arange(0, start, end);
    SetRaggedTensorAttr(iter.first, value);
  }
}

void FsaClass::CopyOtherAttrs(FsaClass &src) {
  for (const auto &iter : src.other_attrs) {
    if (!HasAttr(iter.first)) SetAttr(iter.first, iter.second);
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map) {
  for (auto &iter : src.ragged_tensor_attrs) {
    if (!HasAttr(iter.first)) {
      // Only integer types ragged attributes are supported now
      K2_CHECK_EQ(iter.second.any.GetDtype(), kInt32Dtype);
      auto new_value = (iter.second).Index(arc_map, 0, false);
      SetRaggedTensorAttr(iter.first, new_value.first);
    }
  }
}

void FsaClass::CopyRaggedTensorAttrs(FsaClass &src, RaggedAny &arc_map) {
  for (auto &iter : src.ragged_tensor_attrs) {
    if (!HasAttr(iter.first)) {
      // We currently don't support float ragged attributes
      K2_CHECK_EQ(iter.second.any.GetDtype(), kInt32Dtype);
      RaggedAny new_value = (iter.second).Index(arc_map);
      new_value = new_value.RemoveAxis(new_value.any.NumAxes() - 2);
      SetRaggedTensorAttr(iter.first, new_value);
    }
  }
}

void FsaClass::SetScoresStochastic(torch::Tensor scores) {
  K2_CHECK_EQ(scores.sizes().size(), 1);
  K2_CHECK_EQ(scores.dtype(), torch::kFloat32);
  K2_CHECK_EQ(scores.numel(), fsa.NumElements());

  auto ragged_scores = RaggedAny(fsa.shape.To(GetContext(scores)), scores);
  RaggedAny norm_scores = ragged_scores.Normalize(true).To(Scores().device());
  SetScores(norm_scores.Data());
}

void FsaClass::SetScores(torch::Tensor scores) {
  K2_CHECK_EQ(scores.numel(), fsa.NumElements());
  Scores().copy_(scores.detach());
  PhantomSetScoresFunction::apply(*this, scores);
}

torch::Tensor &FsaClass::Scores() {
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

const torch::Tensor &FsaClass::Scores() const {
  return const_cast<FsaClass *>(this)->Scores();
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
                    << " = " << PropertiesStr() << ", arcs are : " << fsa;
    }
  }
  return properties;
}

std::string FsaClass::PropertiesStr() /*const*/ {
  return FsaPropertiesAsString(Properties());
}

torch::Tensor FsaClass::Arcs() {
  auto device = GetDevice(fsa.Context());
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

torch::Tensor FsaClass::Labels() /*const*/ { return Arcs().index({"...", 2}); }

void FsaClass::SetLabels(torch::Tensor labels) {
  K2_CHECK_EQ(labels.numel(), fsa.NumElements());
  K2_CHECK_EQ(labels.scalar_type(), torch::kInt32);
  Labels().copy_(labels);
}

FsaClass &FsaClass::SetRequiresGrad(bool requires_grad /*=true*/) {
  Scores().requires_grad_(requires_grad);
  return *this;
}

FsaClass FsaClass::ToOtherContext(const ContextPtr &context) const {
  FsaClass dest(fsa.To(context));
  auto device = GetDevice(context);
  for (const auto &iter : tensor_attrs) {
    dest.SetTensorAttr(iter.first, (iter.second).to(device));
  }
  for (const auto &iter : ragged_tensor_attrs) {
    dest.SetRaggedTensorAttr(iter.first, (iter.second).To(device));
  }
  for (const auto &iter : other_attrs) {
    dest.SetAttr(iter.first, iter.second);
  }
  // The following is a magic invocation to make sure
  // the backprop happens.
  PhantomSetScoresFunction::apply(dest, Scores().to(device));
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

std::string FsaClass::ToString() const {
  std::ostringstream os;
  if (fsa.NumAxes() == 2)
    os << "k2.Fsa: ";
  else
    os << "k2.FsaVec: ";

  // TODO: support fsa.NumAxes() == 3
  K2_CHECK_EQ(fsa.NumAxes(), 2);

  std::vector<Array1<int32_t>> extra_labels;
  for (auto &p : tensor_attrs) {
    if (p.second.scalar_type() == torch::kInt) {
      extra_labels.push_back(
          FromTorch<int32_t>(const_cast<torch::Tensor &>(p.second)));
    }
  }

  os << FsaToString(fsa, /*openfst*/ false,
                    /*num_extra_labels*/ extra_labels.size(),
                    /*extra_labels*/ extra_labels.data(),
                    /*num_ragged_labels*/ 0,
                    /*ragged_labels*/ nullptr);
  return os.str();
}

FsaClass FsaClass::CreateFsaVec(std::vector<FsaClass> &fsas) {
  DeviceGuard guard(fsas[0].fsa.Context());
  std::vector<Fsa *> tmp_fsas;
  std::vector<torch::Tensor> tmp_scores;

  tmp_fsas.reserve(fsas.size());
  for (auto &f : fsas) {
    tmp_fsas.push_back(&f.fsa);
    tmp_scores.push_back(f.Scores());
  }
  FsaVec fsa_vec = k2::CreateFsaVec(tmp_fsas.size(), tmp_fsas.data());

  // TODO(fangjun): Don't handle scores specially, treat it
  // like other tensor attributes
  torch::Tensor scores = torch::cat(tmp_scores, 0);

  // TODO(fangjun): support propagating attributes
  FsaClass dest(fsa_vec);
  dest.SetScores(scores);
  return dest;
}

FsaClass FsaClass::ArcSort() /*const*/ {
  Array1<int32_t> arc_map;
  Ragged<Arc> arcs;
  k2::ArcSort(fsa, &arcs, &arc_map);
  return FromUnaryFunctionTensor(*this, arcs, ToTorch<int32_t>(arc_map));
}

void FsaClass::SetAttr(const std::string &name, torch::IValue value) {
  if (name == "grad") {
    // Note we don't use pybind11's def_property since it does not allow
    // to use argument annotations, which means it is not possible to
    // run: fsa.grad = None
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR > 6)
    if (value.isNone()) {
      Scores().mutable_grad() = {};
    } else {
      Scores().mutable_grad() = value.toTensor();
    }
#else
    if (value.isNone()) {
      Scores().grad() = {};
    } else {
      Scores().grad() = value.toTensor();
    }
#endif
    return;
  }

  if (name == "scores") {
    // Note we don't use pybind11's def_property to set scores since it will go
    // into __setattr__ function when running fsa.scores = tensor.
    K2_CHECK(value.isTensor());
    SetScores(value.toTensor());
    return;
  }

  if (HasAttr(name)) DeleteAttr(name);

  all_attr_names.insert(name);

  if (value.isTensor()) {
    SetTensorAttr(name, value.toTensor());
    return;
  }

  if (GetCustomClassName(value) == "_k2.RaggedAnyHolder") {
    SetRaggedTensorAttr(name, ToRaggedAny(value));
    return;
  }

  other_attrs[name] = value;
}

torch::IValue FsaClass::GetAttr(const std::string &name) const {
  if (name == "grad") {
    return torch::IValue(Scores().grad());
  }

  if (name == "scores") {
    return Scores();
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

  return other_attrs.at(name);
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

      // Erase the filler
      std::string filler_name = name + "_filler";
      auto it = other_attrs.find(filler_name);
      if (it != other_attrs.end()) {
        other_attrs.erase(it);
      }
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

bool FsaClass::HasAttr(const std::string &name) const {
  // we treat grad & scores as attributes, though they don't store in attribute
  // containers.
  if (name == "grad" || name == "scores") return true;
  return all_attr_names.count(name) > 0;
}

FsaClass FsaClass::Index(int32_t index) {
  K2_CHECK_EQ(fsa.NumAxes(), 3);
  K2_CHECK_GE(index, 0);
  K2_CHECK_LT(index, fsa.Dim0());
  Ragged<Arc> sub_fsa = fsa.Index(0, index);
  int32_t start = sub_fsa.values.Data() - fsa.values.Data(),
          end = start + sub_fsa.values.Dim();

  FsaClass dest(sub_fsa);
  // Check the validation of the fsa, will trigger a fatal error if the fsa
  // is not valid.
  dest.Properties();
  dest.CopyTensorAttrs(*this, start, end);
  dest.CopyRaggedTensorAttrs(*this, start, end);
  dest.CopyOtherAttrs(*this);
  PhantomSetScoresFunction::apply(
      dest, Scores().index({torch::indexing::Slice(start, end)}));
  return dest;
}

torch::IValue FsaClass::GetFiller(const std::string &name) const {
  std::string filler_name = name + "_filler";
  auto iter = other_attrs.find(filler_name);
  if (iter != other_attrs.end()) {
    return iter->second;
  } else {
    return torch::IValue(0);
  }
}

Ragged<int32_t> FsaClass::GetStateBatches(bool transpose /*= true*/) {
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

Array1<int32_t> FsaClass::GetDestStates(bool as_idx01) {
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

Ragged<int32_t> FsaClass::GetIncomingArcs() {
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

Ragged<int32_t> FsaClass::GetEnteringArcIndexBatches() {
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

Array1<int32_t> FsaClass::GetEnteringArcs(bool use_double_scores) {
  std::string name = "entering_arcs";
  auto it = cached_tensor.find(name);
  if (it != cached_tensor.end()) {
    return it->second;
  }

  // This function will compute and fill in cached_tensor[name]
  (void)GetForwardScores(use_double_scores, /*log_semiring*/ false);

  return cached_tensor.at(name);
}

Ragged<int32_t> FsaClass::GetLeavingArcIndexBatches() {
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
torch::Tensor FsaClass::GetForwardScoresImpl(bool use_double_scores,
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

torch::Tensor FsaClass::GetForwardScores(bool use_double_scores,
                                         bool log_semiring) {
  return GetForwardScoresFunction::apply(*this, use_double_scores, log_semiring,
                                         Scores());
};

}  // namespace k2
