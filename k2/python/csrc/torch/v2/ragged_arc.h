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

#ifndef K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC_H_
#define K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

// It is a wrapper of Ragged<Arc> to support backward props in PyTorch
struct RaggedArc {
  Ragged<Arc> fsa;
  torch::Tensor scores;  // shares the same memory with fsa.values

  /// It contains all tensor attributes of this FSA
  std::unordered_map<std::string, torch::Tensor> tensor_attrs;

  /// It contains all ragged tensor attributes of this FSA
  /// It supports only ragged tensor attributes with dtype==torch.int32
  std::unordered_map<std::string, RaggedAny> ragged_tensor_attrs;

  /// All attributes of other types of this FSA
  std::unordered_map<std::string, py::object> other_attrs;

  /// The name of all attributes of this FSA
  std::unordered_set<std::string> all_attr_names;

  // The default constructor initializes an invalid ragged tensor.
  RaggedArc() = default;

  explicit RaggedArc(const Ragged<Arc> &fsa) : fsa(fsa) {}
  RaggedArc(const Ragged<Arc> &fsa, torch::Tensor scores)
      : fsa(fsa), scores(scores) {}

  // TODO: support more options, e.g.,
  /* Construct a RaggedArc from a string.

     @param s  The input string that can be passed to FsaFromString
     @param extra_label_names A list of strings specifying the names of
                extra labels. If it is empty, then the string represents
                an acceptor.
   */
  RaggedArc(const std::string &s,
            const std::vector<std::string> &extra_label_names = {});

  RaggedArc(const RaggedArc &other) = default;

  RaggedArc &operator=(const RaggedArc &other) = default;

  RaggedArc(RaggedArc &&other) = default;

  RaggedArc &operator=(RaggedArc &&other) = default;

  // Populate `this->scores` and return it
  const torch::Tensor &Scores() const;
  torch::Tensor &Scores();

  /* Return a 2-D int32 torch tensor.
     Each row represents an arc, where:
      - column 0 is the source state
      - column 1 is the dest state
      - column 2 is the label
      - column 3 is the score, reinterpreted cast from a float.

    @caution You should not modify the returned tensor since it shares
    the underlying memory with this FSA.
   */
  torch::Tensor Arcs() /*const*/;

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

  static RaggedArc CreateFsaVec(std::vector<RaggedArc> &fsas);

  RaggedArc ArcSort() /*const*/;

  /** Associate an attribute with a value.

    If there is no attribute with the given `name`,
      - If `value` is an instance of `torch::Tensor`, add it to `tensor_attrs`
      - If `value` is an instance of `RaggedAny`, add it to
        `ragged_tensor_attrs`
      - Otherwise, add it to `other_attrs`.

    If there is already an attribute with the given `name`, we first
    remove this attribute and then add it using the above logic.

    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetAttr(const std::string &name, py::object value);

  /** Get an attribute by its name.

    Raise an Python exception "AttributeError" if there is no such attribute.

    @param name The attribute name.
    @return Return the value of the attribute.
   */
  py::object GetAttr(const std::string &name) const;

  /** Delete an attribute by its name.

    Raise an Python exception "AttributeError" if there is no such attribute.

    @param name The attribute name.
   */
  void DeleteAttr(const std::string &name);

  /** Query if an attribute exists.

    @param name The attribute name.
    @return Return `true` if the given attribute exists.
            Return `false` otherwise.
   */
  bool HasAttr(const std::string &name) const;

  /** Wrapper for k2::GetForwardScores

    @param use_double_scores True to use double for computation.
                             False to use float.
    @param log_semiring   True to use log semiring.
                          False to use tropical semiring.

    @return Return a 1-D tensor containing the forward scores of each state.
            If use_double_scores is True, the dtype of the returned tensor is
            torch.float64; otherwise, the dtype is torch.float32.
   */
  torch::Tensor GetForwardScoresImpl(bool use_double_scores, bool log_semiring);

  /// GetForwardScores() supporting autograd
  torch::Tensor GetForwardScores(bool use_double_scores, bool log_semiring);

 private:
  void SetAttr(const std::string &name, torch::Tensor value) {
    K2_CHECK_EQ(value.size(0), fsa.NumElements())
        << "shape[0] of the tensor MUST be equal to number of arcs";
    tensor_attrs[name] = value;
  }
  void SetAttr(const std::string &name, const RaggedAny &value) {
    K2_CHECK_EQ(value.any.Dim0(), fsa.NumElements())
        << "dim0 of the tensor MUST be equal to number of arcs";
    ragged_tensor_attrs[name] = value;
  }

 public:  // we make these functions public since they are called in autograd
          // related functions
  /** Wrapper for k2::GetStateBatches.

     If `cached_ragged_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_ragged_tensor` does not contain the value, it
     computes the state batches, saves it into `cached_ragged_tensor`,
     and returns it.
   */
  Ragged<int32_t> GetStateBatches(bool transpose = true);

  /** Wrapper for k2::GetDestStates.

     If `cached_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_tensor` does not contain the value, it
     computes the dest states, saves it into `cached_tensor`,
     and returns it.
   */
  Array1<int32_t> GetDestStates(bool as_idx01);

  /** Wrapper for k2::GetIncomingArcs.

     If `cached_ragged_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_ragged_tensor` does not contain the value, it
     computes the incoming arcs, saves it into `cached_ragged_tensor`,
     and returns it.
   */
  Ragged<int32_t> GetIncomingArcs();

  /** Wrapper for k2::GetIncomingArcIndexBatches.

     If `cached_ragged_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_ragged_tensor` does not contain the value, it
     computes the incoming arcs, saves it into `cached_ragged_tensor`,
     and returns it.
   */
  Ragged<int32_t> GetEnteringArcIndexBatches();

  /** It uses k2::GetForwardScores() to compute `entering_arcs`.

     If `cached_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_tensor` does not contain the value, it
     computes the entering arcs, saves it into `cached_tensor`,
     and returns it.
   */
  Array1<int32_t> GetEnteringArcs(bool use_double_scores);

  /** Wrapper for k2::GetLeavingArcIndexBatches.

     If `cached_ragged_tensor` already contains the value, no
     computation is performed and the value is return directly

     If `cached_ragged_tensor` does not contain the value, it
     computes the leaving arc batches, saves it into `cached_ragged_tensor`,
     and returns it.
   */
  Ragged<int32_t> GetLeavingArcIndexBatches();

 private:
  /// It saves intermediate results for various FSA operations
  std::unordered_map<std::string, Ragged<int32_t>> cached_ragged_tensor;

  /// It saves intermediate results for various FSA operations
  std::unordered_map<std::string, Array1<int32_t>> cached_tensor;
};

}  // namespace k2
#endif  // K2_PYTHON_CSRC_TORCH_V2_RAGGED_ARC_H_
