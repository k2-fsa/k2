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

#include "k2/csrc/fsa_utils.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/autograd/arc_sort.h"
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

  // TODO: we also need to pass the name of extra_labels and ragged_labels.
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
        fsa.values.Data(), sizes, strides, [](void *) {}, options);
    scores = tmp_scores.index({"...", -1});
  }
  return scores;
}

const torch::Tensor &RaggedArc::Scores() const {
  return const_cast<RaggedArc *>(this)->Scores();
}

torch::Tensor RaggedArc::Arcs() /*const*/ {
  auto device_type = ToTorchDeviceType(fsa.Context()->GetDeviceType());
  int32_t device_id = fsa.Context()->GetDeviceId();
  auto device = torch::Device(device_type, device_id);
  auto scalar_type = ToScalarType<int32_t>::value;
  // an Arc has 4 members
  static_assert(sizeof(Arc) == 4 * sizeof(int32_t), "");

  std::vector<int64_t> sizes = {fsa.values.Dim(), 4};  // [num_rows, num_cols]
  std::vector<int64_t> strides = {4, 1};               // in number of elements
  auto options = torch::device(device).dtype(scalar_type);

  // NOTE: We are accessing it from python as a property of an FSA,
  // so it is alive as long as the underlying FSA is alive.
  return torch::from_blob(
      fsa.values.Data(), sizes, strides, [](void *) {}, options);
}

RaggedArc &RaggedArc::SetRequiresGrad(bool requires_grad /*=true*/) {
  Scores().requires_grad_(requires_grad);
  return *this;
}

std::string RaggedArc::ToString() const {
  std::ostringstream os;
  if (fsa.NumAxes() == 2)
    os << "k2.Fsa: ";
  else
    os << "k2.FsaVec: ";

  // TODO: support fsa.NumAxes() == 3
  K2_CHECK_EQ(fsa.NumAxes(), 2);

  int32_t num_extra_labels = 0;
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

RaggedArc RaggedArc::ArcSort() /*const*/ {
  RaggedArc out;
  (void)ArcSortFunction::apply(*this, Scores(), &out);
  return out;
}

void RaggedArc::SetAttr(const std::string &name, py::object value) {
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
      // It's safe to use c_str() here as it is copied inside PyErr_SetString()
      //
      // See https://github.com/python/cpython/blob/main/Python/errors.c#L234
      PyErr_SetString(PyExc_AttributeError, os.str().c_str());
      throw py::error_already_set();
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

}  // namespace k2
