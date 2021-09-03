/**
 * @brief A wrapper around Ragged<Any> and torch::Tensor
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

#ifndef K2_PYTHON_CSRC_TORCH_RAGGED_V2_ANY_H
#define K2_PYTHON_CSRC_TORCH_RAGGED_V2_ANY_H

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch.h"

namespace k2 {

/** RaggedAny is introduced to support backward propagation on
 Ragged<Any> since there has to be a tensor involved during backward
 propagations.

 Ragged<Any> unifies Ragged<int32_t>, Ragged<float>, and Ragged<double>
 so that Python only sees Ragged<Any>.
*/
struct RaggedAny {
  Ragged<Any> any;
  torch::Tensor data;  //!< shares the underlying memory with any.values

  /// The default constructor initializes an invalid ragged tensor.
  RaggedAny() = default;

  RaggedAny(const RaggedAny &) = default;
  RaggedAny &operator=(const RaggedAny &) = default;
  RaggedAny(RaggedAny &&) = default;
  RaggedAny &operator=(RaggedAny &&) = default;

  explicit RaggedAny(const Ragged<Any> &any) : any(any) {}

  /** Create a ragged tensor from its string representation.

      An example string with 3 axes is::

      [ [[1 2] [3] []]   [[1] [10] [20 30]] ]

     @param s  The string representation of a ragged tensor.
     @param dtype  An instance of `torch.dtype`. Supported dtypes are:
                   `torch.float32`, `torch.float64`, and
                   `torch.int32`. If it is `None`, the dtype is
                   inferred from the given string.  It first tries
                   to use `torch.int32`. If it fails, it will switch
                   to `torch.float32`.

     @note We can support other dtypes if needed.
   */
  RaggedAny(const std::string &s, py::object dtype = py::none());

  /** Create a ragged tensor from a list of sublist(s).

     @param data A python list-of lists.
     @param dtype An instance of `torch.dtype`. If it is `None`,
                  the data type is inferred from the input `data`,
                  which will either be torch.int32 or torch.float32.
                  Supported dtypes are: `torch.int32`, torch.`float32`,
                  and `torch.float64`.

     @note It supports `data` with number of axes >= 2.
   */
  RaggedAny(py::list data, py::object dtype = py::none());

  /// Populate `this->data` and return it
  const torch::Tensor &Data() const;

  /** Convert a ragged tensor to a string.

     @return Return a string representation of this tensor.
   */
  std::string ToString() const;

  /* Move a ragged tensor to a given device.

     Note: If the this tensor is already on the given device, itself
     is returned. Otherwise, a copy of the this tensor moved to the given
     device is returned.

     @param device  A torch device, which can be either a CPU device
                    or a CUDA device.

     @return Return a ragged tensor on the given device.
   */
  RaggedAny To(torch::Device device) const;

  /** Move this tensor to a given device.
*
    Note: If the this tensor is already on the given device, itself
    is returned. Otherwise, a copy of the this tensor moved to the given
    device is returned.

    @param device A string representation of a device, e.g.,
                  "cpu", "cuda:0", "cuda:1", etc.

    @return Return a ragged tensor on the given device.
   */
  RaggedAny To(const std::string &device) const;

  /* Convert a ragged tensor to given scalar type.

     Note: If the this tensor is already of the given type, itself
     is returned. Otherwise, a copy of the this tensor converted to the given
     type is returned.

     @param scalar_type The type this tensor should be converted to.

     @return Return a ragged tensor with the specified type.
   */
  RaggedAny To(torch::ScalarType scalar_type) const;

  /// Return a copy of this ragged tensor
  RaggedAny Clone() const;

  /** Enable/Disable requires_grad of this tensor

     @param requires_grad True to require grad for this tensors.
                          False to not require grad.

     @note If this is NOT a float tenor and requires_grad is `True`,
     PyTorch will throw a RuntimeError exception.
   */
  RaggedAny &SetRequiresGrad(bool requires_grad = true);

  /** Compute the sum over the last axis of the ragged tensor.

     It is a wrapper around k2::SumPerSublist.

     @note It supports autograd if the dtype of this tensor is
     `torch.float32` or `torch.float64`.

     @param initial_value  This value is added to the sum of each
     sub-list. If a sublist is empty, the sum of it is just initial_value.

     @return Return the sum of each sublist as a 1-D tensor.
   */
  torch::Tensor Sum(float initial_value = 0) const;

  /** Index a ragged tensor (supporting only axis==0 at present).

     It requires that the ragged tensor has at least 3 axes.

     @TODO: To add autograd support.

     @param axis  The axis to index. Must be 0 at present.
     @param i  The i-th sublist of the specified axis.

     @return Return a ragged tensor with one fewer axis.
     It shares data with "this" tensor.
   */
  RaggedAny Index(int32_t axis, int32_t i) const;

  /** A wrapper around k2::RemoveAxis. See its doc in
     k2/csrc/ragged_ops.h
   */
  RaggedAny RemoveAxis(int32_t axis) /*const*/;

  /** A wrapper for k2::RaggedArange. See its doc for help.
   */
  RaggedAny Arange(int32_t axis, int32_t begin, int32_t end) /*const*/;

  /// Wrapper for k2::RemoveValuesLeq()
  RaggedAny RemoveValuesLeq(py::object cutoff) /*const*/;

  /// Wrapper for k2::RemoveValuesEq()
  RaggedAny RemoveValuesEq(py::object target) /*const*/;

  /// Wrapper for k2::ArgMaxPerSublist
  torch::Tensor ArgMax(py::object initial_value = py::none()) /*const*/;

  // Wrapper for k2::MaxPerSublist
  torch::Tensor Max(py::object initial_value = py::none()) /*const*/;

  // Wrapper for k2::MinPerSublist
  torch::Tensor Min(py::object initial_value) /*const*/;

  /// Wrapper for k2::Cat
  static RaggedAny Cat(const std::vector<RaggedAny> &srcs, int32_t axis);

  /// Wrapper for k2::UniqueSequences
  std::tuple<RaggedAny, torch::optional<RaggedAny>,
             torch::optional<torch::Tensor>>
  Unique(bool need_num_repeats = false, bool need_new2old_indexes = false);

  /// Wrapper for k2::NormalizePerSublist
  RaggedAny Normalize(bool use_log) /*const*/;

  /// Wrapper for k2::PadRagged
  torch::Tensor Pad(const std::string &mode,
                    py::object padding_value) /*const*/;

  /// Convert a ragged tensor to a list of lists [of lists ...]
  /// Note: You can use the return list to construct a ragged tensor.
  py::list ToList() /*const*/;

  /// Wrapper for k2::SortSublists
  torch::optional<torch::Tensor> Sort(bool descending = false,
                                      bool need_new2old_indexes = false);

  /// Wrapper for k2::Index
  RaggedAny Index(RaggedAny &indexes, bool remove_axis = true) /*const*/;

  /// Wrapper for k2::Index
  std::pair<RaggedAny, torch::optional<torch::Tensor>> Index(
      torch::Tensor indexes, int32_t axis,
      bool need_value_indexes = false) /*const*/;

  /// Wrapper for k2::Index
  RaggedAny Index(torch::Tensor src,
                  py::object default_value = py::none()) /*const*/;

  /// Wrapper for k2::Index
  torch::Tensor IndexAndSum(torch::Tensor src,
                            py::object default_value = py::none()) /*const*/;
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_RAGGED_V2_ANY_H
