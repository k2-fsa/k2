/**
 * @brief python wrapper for Ragged<Arc>
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC__H_
#define K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC__H_

#include <unordered_map>

#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

// It is a wrapper of Ragged<Arc> to support backward props in PyTorch
struct RaggedArc {
  Ragged<Arc> fsa;
  torch::Tensor scores;  // shares the same memory with fsa.values

  std::unordered_map<std::string, torch::Tensor> tensor_attrs;
  std::unordered_map<std::string, RaggedAny> ragged_tensor_attrs;

  std::unordered_map<std::string, py::object> other_attrs;
  std::unordered_set<std::string> all_attr_names;

  // The default constructor initializes an invalid ragged tensor.
  RaggedArc() = default;

  explicit RaggedArc(const Ragged<Arc> &fsa) : fsa(fsa) {}

  // TODO: support more options, e.g.,
  /* Construct a RaggedArc from a string.

     @param s  The input string that can be passed to FsaFromString
     @param extra_label_names A list of strings specifying the names of
                extra labels. If it is empty, then the string represents
                an acceptor.
   */
  RaggedArc(const std::string &s, py::list extra_label_names = py::none());

  RaggedArc(const RaggedArc &other) = default;

  RaggedArc &operator=(const RaggedArc &other) = default;

  RaggedArc(RaggedArc &&other) = default;

  RaggedArc &operator=(RaggedArc &&other) = default;

  // Populate `this->scores` and return it
  const torch::Tensor &Scores() const;
  torch::Tensor &Scores();

  /* Enable/Disable requires_grad of this tensor

     @param requires_grad True to requires grad for this tensors.
                          False to not require grad.

     @note If this is NOT a float tenors and requires_grad is True,
     it throws a RuntimeError exception.
   */
  RaggedArc &SetRequiresGrad(bool requires_grad = true);

  /* Convert a ragged arc to a string.

     @return Return a string representation of the ragged arc.
   */
  std::string ToString() const;

  RaggedArc ArcSort() /*const*/;
};

}  // namespace k2
#endif  // K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC_H_
