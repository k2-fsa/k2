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

RaggedArc::RaggedArc(const std::string &s,
                     py::list extra_label_names /*= py::none()*/) {
  // TODO: pass following options from arguments
  bool openfst = false;
  int32_t num_extra_labels = 0;
  Array2<int32_t> extra_labels;
  Array2<int32_t> *p_extra_labels;
  int32_t num_ragged_labels = 0;
  Ragged<int32_t> *ragged_labels = nullptr;

  if (extra_label_names) {
    num_extra_labels = extra_label_names.size();
    p_extra_labels = &extra_labels;
  }

  fsa = FsaFromString(s, openfst, num_extra_labels, p_extra_labels,
                      num_ragged_labels, ragged_labels);

  if (num_extra_labels) {
    for (int32_t i = 0; i != num_extra_labels; ++i) {
      std::string name = py::str(extra_label_names[i]);
      Array1<int32_t> row = extra_labels.Row(i);
      tensor_attrs[name] = ToTorch(row);
    }
  }

  // TODO: we also need to pass the name of extra_labels and ragged_labels.
}  // namespace k2

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

}  // namespace k2
