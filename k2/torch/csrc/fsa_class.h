/**
 * @brief python wrapper for FsaOrVec
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

#ifndef K2_TORCH_CSRC_FSA_CLASS_H_
#define K2_TORCH_CSRC_FSA_CLASS_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/torch/csrc/utils.h"
#include "torch/script.h"

namespace k2 {

// It is a wrapper of FsaOrVec to support attributes propagation
struct FsaClass {
  FsaOrVec fsa;
  int32_t properties = 0;

  /// It contains all tensor attributes of this FSA
  std::unordered_map<std::string, torch::Tensor> tensor_attrs;

  /// It contains all ragged tensor attributes of this FSA
  std::unordered_map<std::string, Ragged<int32_t>> ragged_tensor_attrs;

  /// The name of all attributes of this FSA
  std::unordered_set<std::string> all_attr_names;

  // The default constructor initializes an invalid FSA.
  FsaClass() = default;

  explicit FsaClass(const FsaOrVec &fsa) : fsa(fsa) {
    // Check the validation of the fsa, will trigger a fatal error if the fsa
    // is not valid.
    Properties();
  }

  FsaClass(const FsaClass &other) = default;

  FsaClass &operator=(const FsaClass &other) = default;

  FsaClass(FsaClass &&other) = default;

  FsaClass &operator=(FsaClass &&other) = default;

  /**
    Create an Fsa object, including propagating properties from the source FSA.
    This is intended to be called from unary functions on FSAs where the arc_map
    is a Tensor of int32 (i.e. not ragged).
    @param src The source Fsa, i.e. the arg to the unary function.
    @param arcs The raw output of the unary function, as output by whatever C++
                algorithm we used.
    @param arc_map A map from arcs in `arcs` to the corresponding arc-index in
                   `src`, or -1 if the arc had no source arc
                   (e.g. added epsilon self-loops).
   */
  static FsaClass FromUnaryFunctionTensor(FsaClass &src, const FsaOrVec &arcs,
                                          torch::Tensor arc_map);

  /* Return a 1-D float32 torch tensor.
      @caution You should not modify the returned tensor since it shares
      the underlying memory with this FSA.
     */
  torch::Tensor Scores();
  /* Set scores, will modify scores in fsa.arcs

     @param scores The given scores.
   */
  void SetScores(torch::Tensor scores);

  /* Return a 1-D int32 torch tensor.
      @caution You should not modify the returned tensor since it shares
      the underlying memory with this FSA.
     */
  torch::Tensor Labels() /*const*/;
  /* Set labels, will modify labels in fsa.arcs

     @param The given labels.
   */
  void SetLabels(torch::Tensor labels);

  // Get fsa properties.
  int32_t Properties();

  /** Associate an attribute with a value.
    If there is no attribute with the given `name`,
      - If `value` is an instance of `torch::Tensor`, add it to `tensor_attrs`
      - If `value` is an instance of `Ragged<int32_t>`, add it to
        `ragged_tensor_attrs`
    If there is already an attribute with the given `name`, we first
    remove this attribute and then add it using the above logic.
    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetAttr(const std::string &name, torch::IValue value);

  /** Get an attribute by its name.
    Raise a RuntimeError exception if there is no such attribute.
    @param name The attribute name.
    @return Return the value of the attribute.
   */
  torch::IValue GetAttr(const std::string &name) /*const*/;

  torch::Tensor GetTensorAttr(const std::string &name) const {
    return tensor_attrs.at(name);
  }

  Ragged<int32_t> GetRaggedTensorAttr(const std::string &name) const {
    return ragged_tensor_attrs.at(name);
  }

  /** Delete an attribute by its name.
    Raise a RuntimeError exception if there is no such attribute.
    @param name The attribute name.
   */
  void DeleteAttr(const std::string &name);

  /** Query if an attribute exists.
    @param name The attribute name.
    @return Return `true` if the given attribute exists.
            Return `false` otherwise.
   */
  bool HasAttr(const std::string &name) const;

  /** Propagate attributes from source FsaClass via tensor arc_map.

    Caution: If there are attributes in source FsaClass with the name
    conflicting with current FsaClass, we will skip the attributes in source
    FsaClass and keep the current one.

    @param src  The source FsaClass.
    @param arc_map  The arc_map (as idx012) to select items in attributes.
   */
  void CopyAttrs(FsaClass &src, torch::Tensor arc_map);

  // Transfer current fsa to another device.
  FsaClass To(torch::Device device) const;
  FsaClass To(const std::string &device) const;

 private:
  /** Associate an tensor attribute with a value directly.

    Caution: This function assumes that there is no other type of attribute
    named the given `name`. And the tensor type attribute named the given `name`
    will be overwritten.

    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetTensorAttr(const std::string &name, torch::Tensor value) {
    K2_CHECK_EQ(value.size(0), fsa.NumElements())
        << "'" << name
        << "': shape[0] of the tensor MUST be equal to number of arcs";
    K2_CHECK(ContextFromTensor(value)->IsCompatible(*fsa.Context()));
    all_attr_names.insert(name);
    tensor_attrs[name] = value;
  }

  /** Associate a ragged tensor attribute with a value directly.

    Caution: This function assumes that there is no other type of attribute
    named the given `name`. And the ragged tensor type attribute named the given
    `name` will be overwritten.

    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetRaggedTensorAttr(const std::string &name,
                           const Ragged<int32_t> &value) {
    K2_CHECK_EQ(value.Dim0(), fsa.NumElements())
        << "'" << name
        << "': dim0 of the tensor MUST be equal to number of arcs";
    K2_CHECK(value.Context()->IsCompatible(*fsa.Context()));
    all_attr_names.insert(name);
    ragged_tensor_attrs[name] = value;
  }

  /** Propagate tensor attributes from source FsaClass via tensor arc_map.

      Caution: If there are attributes in source FsaClass with the name
      conflicting with current FsaClass, we will skip the attributes in source
      FsaClass and keep the current one.

      @param src  The source FsaClass.
      @param arc_map  The arc_map (as idx012) to select items in attributes.
     */
  void CopyTensorAttrs(FsaClass &src, torch::Tensor arc_map);

  /** Propagate ragged tensor attributes from source FsaClass via tensor
    arc_map.

    Caution: If there are attributes in source FsaClass with the name
    conflicting with current FsaClass, we will skip the attributes in source
    FsaClass and keep the current one.

    @param src  The source FsaClass.
    @param arc_map  The arc_map (as idx012) to select items in attributes.
   */
  void CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map);

  /** Transfer current FsaClass to another devices.

    Note: This function assumes that the target context is different from
    current context. It crashes if you call this function with the context the
    same as current one.

    @param context  The target context.
   */
  FsaClass ToOtherContext(const ContextPtr &context) const;
};

}  // namespace k2
#endif  // K2_TORCH_CSRC_FSA_CLASS_H_
