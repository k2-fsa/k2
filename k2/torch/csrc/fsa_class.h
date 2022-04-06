/**
 * @brief Wrapper for k2::Fsa to support attribute propagation.
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
#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/torch/csrc/utils.h"
#include "torch/script.h"

namespace k2 {

// It is a wrapper of FsaOrVec to support attributes propagation
struct FsaClass {
  // TODO(fangjun): Make it a class and set its data members to private
  FsaOrVec fsa;
  int32_t properties = 0;

  // TODO(fangjun): Use two arrays to represent tensor_attrs
  // as there are usually only one or two attributes associated
  // with an FSA in decoding.
  //
  /// It contains all tensor attributes of this FSA
  std::unordered_map<std::string, torch::Tensor> tensor_attrs;

  /// It contains all ragged tensor attributes of this FSA
  std::unordered_map<std::string, Ragged<int32_t>> ragged_tensor_attrs;

  // The default constructor initializes an invalid FSA.
  FsaClass() = default;

  explicit FsaClass(const FsaOrVec &fsa) : fsa(fsa) {
    // Check the validation of the fsa, will trigger a fatal error if the fsa
    // is not valid.
    Properties();
  }

  FsaClass(const FsaClass &) = default;
  FsaClass &operator=(const FsaClass &) = default;
  FsaClass(FsaClass &&) = default;
  FsaClass &operator=(FsaClass &&) = default;

  /// Returns the number of attributes contained in this FSA
  int32_t NumAttrs() const {
    return tensor_attrs.size() + ragged_tensor_attrs.size();
  }

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

  /* Return a 1-D torch.float32 torch tensor.

     @caution It shares the underlying memory with this FSA.
     */
  torch::Tensor Scores();

  /** Set scores, will modify scores in fsa.arcs

     @param scores A 1-D tensor of dtype torch.float32.
   */
  void SetScores(torch::Tensor scores);

  /** Return a 1-D int32 torch tensor.
     @caution It shares the underlying memory with this FSA.
   */
  torch::Tensor Labels();

  /** Set labels, will modify labels in fsa.arcs

     @param labels  A 1-D tensor of dtype torch.int32.
   */
  void SetLabels(torch::Tensor labels);

  // Get fsa properties.
  int32_t Properties();

  /// Return the given tensor attribute by its name
  const torch::Tensor &GetTensorAttr(const std::string &name) const {
    return tensor_attrs.at(name);
  }

  /// Return the given tensor attribute by its name
  torch::Tensor &GetTensorAttr(const std::string &name) {
    return tensor_attrs.at(name);
  }

  /// Return the given ragged tensor attribute by its name
  const Ragged<int32_t> &GetRaggedTensorAttr(const std::string &name) const {
    return ragged_tensor_attrs.at(name);
  }

  /// Return the given ragged tensor attribute by its name
  Ragged<int32_t> &GetRaggedTensorAttr(const std::string &name) {
    return ragged_tensor_attrs.at(name);
  }

  /// Return true if this FSA has a tensor attribute with such a name.
  /// Return false otherwise.
  bool HasTensorAttr(const std::string &name) const {
    return tensor_attrs.count(name) > 0;
  }

  /// Return true if this FSA has a ragged tensor attribute with such a name.
  /// Return false otherwise.
  bool HasRaggedTensorAttr(const std::string &name) const {
    return ragged_tensor_attrs.count(name) > 0;
  }

  /** Delete a tensor attribute by its name.
   *
    Raise a RuntimeError exception if there is no such attribute.

    @param name The attribute name.
   */
  void DeleteTensorAttr(const std::string &name) {
    auto it = tensor_attrs.find(name);
    if (it == tensor_attrs.end()) {
      K2_LOG(FATAL) << "No such tensor attribute: " << name;
    }
    tensor_attrs.erase(it);
  }

  /** Delete a ragged attribute by its name.

      Raise a RuntimeError exception if there is no such attribute.

      @param name The attribute name.
   */
  void DeleteRaggedTensorAttr(const std::string &name) {
    auto it = ragged_tensor_attrs.find(name);
    if (it == ragged_tensor_attrs.end()) {
      K2_LOG(FATAL) << "No such ragged tensor attribute: " << name;
    }
    ragged_tensor_attrs.erase(it);
  }

  /** Propagate attributes from source FsaClass via tensor arc_map.

    @param src  The source FsaClass.
    @param arc_map  The arc_map (as idx012) to select items in attributes.
   */
  void CopyAttrs(FsaClass &src, torch::Tensor arc_map);

  /** Propagate attributes from a list of source FsaClasses via ragged tensor
      arc_map. We assume that each sublist in arc_map contains the indexes into
      arcs (as idx01) of corresponding Fsa in the list of source FsaClasses.
      And we propagate the attributes from the source FsaClass to the
      corresponding Fsa(i.e. sub Fsa of raw FsaVec in current FsaClass object)
      via the indexes in the corresponding sublist of arc_map.

      Caution: The raw fsa in current object MUST be an 3 axes FsaVec, and it
               MUST satisfy `fsa.Numelements() == arc_map.Numelements()` and
               `fsa.Dim0() == arc_map.Dim0()`.

      Note: The attributes of current object is a union of the attributes
            of all the source FsaClasses. For example, srcs[0] has attributes
            attr1, attr2; srcs[1] has attributes attr1, attr3; srcs[2] has
            attributes attr3, attr4; then current FsaClass object has attributes
            attr1, attr2, attr3, attr4 after propagation.

    @param srcs  A vector of the source FsaClasses.
    @param arc_map  The arc_map (as idx01) to select items in attributes.
   */
  void CopyAttrs(std::vector<FsaClass> &srcs, Ragged<int32_t> &arc_map);

  /** Associate an tensor attribute with a value directly.

    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetTensorAttr(const std::string &name, torch::Tensor value) {
    K2_CHECK_EQ(value.size(0), fsa.NumElements())
        << "'" << name
        << "': shape[0] of the tensor MUST be equal to number of arcs";
    K2_CHECK(ContextFromTensor(value)->IsCompatible(*fsa.Context()));
    tensor_attrs[name] = value;
  }

  /** Associate a ragged tensor attribute with a value directly.

    @param name  The attribute name.
    @param value  The attribute value.
   */
  void SetRaggedTensorAttr(const std::string &name,
                           const Ragged<int32_t> &value) {
    K2_CHECK_EQ(value.Dim0(), fsa.NumElements())
        << "'" << name
        << "': dim0 of the tensor MUST be equal to number of arcs";
    K2_CHECK(value.Context()->IsCompatible(*fsa.Context()));
    ragged_tensor_attrs[name] = value;
  }

 private:
  /** Propagate tensor attributes from source FsaClass via tensor arc_map.

      @param src  The source FsaClass.
      @param arc_map  The arc_map (as idx012) to select items in attributes.
     */
  void CopyTensorAttrs(FsaClass &src, torch::Tensor arc_map);


  /** Propagate tensor attributes from a list of source FsaClasses via ragged
      tensor arc_map.
      See docs in CopyAttrs that has same arguments for more details.

      @param srcs  A vector of the source FsaClasses.
      @param arc_map  The arc_map (as idx01) to select items in attributes.
   */
  void CopyTensorAttrs(std::vector<FsaClass> &srcs, Ragged<int32_t> &arc_map);

  /** Propagate ragged tensor attributes from source FsaClass via tensor
    arc_map.

    @param src  The source FsaClass.
    @param arc_map  The arc_map (as idx012) to select items in attributes.
   */
  void CopyRaggedTensorAttrs(FsaClass &src, torch::Tensor arc_map);

  /** Propagate ragged tensor attributes from a list of source FsaClasses via
      ragged tensor arc_map.
      See docs in CopyAttrs that has same arguments for more details.

      @param srcs  A vector of the source FsaClasses.
      @param arc_map  The arc_map (as idx01) to select items in attributes.
   */
  void CopyRaggedTensorAttrs(std::vector<FsaClass> &srcs,
                             Ragged<int32_t> &arc_map);
};

}  // namespace k2
#endif  // K2_TORCH_CSRC_FSA_CLASS_H_
