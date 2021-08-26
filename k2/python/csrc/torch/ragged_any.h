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

#ifndef K2_PYTHON_CSRC_TORCH_RAGGED_ANY_H
#define K2_PYTHON_CSRC_TORCH_RAGGED_ANY_H

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/k2.h"
#include "torch/extension.h"

namespace k2 {

// RaggedAny is introduced to support backward propagations on
// Ragged<Any> since there has to be a tensor involved during backprob
struct RaggedAny {
  Ragged<Any> any_;
  torch::Tensor data_;  // shares the underlying memory with any_.values

  // The default constructor initializes an invalid ragged tensor.
  RaggedAny() = default;

  explicit RaggedAny(const Ragged<Any> &any) : any_(any) {}

  // share Ragged<Any> with other
  RaggedAny(const RaggedAny &other) : any_(other.any_) {}

  // share Ragged<Any> with other
  RaggedAny &operator=(const RaggedAny &other) { any_ = other.any_; }

  RaggedAny(RaggedAny &&other) { any_ = std::move(other.any_); }

  RaggedAny &operator=(RaggedAny &&other) {
    if (&other != this) any_ = std::move(other.any_);
    return *this;
  }

  /* Create a ragged tensor with two axes.

     @param data a list-of-list
     @param dtype An instance of torch.dtype. If it is None,
                  the data type is inferred from the input `data`,
                  which will either be torch.int32 or torch.float32.

     @TODO To support `data` with arbitrary number of axes.

     @CAUTION Currently supported dtypes are torch.float32, torch.float64,
     and torch.int32. To support torch.int64 and other dtypes, we can
     add a new macro to replace `FOR_REAL_AND_INT32_TYPES`.
   */
  RaggedAny(py::list data, py::object dtype = py::none());

  /* Convert a ragged tensor to a string.

     @param any The input ragged tensor.
     @return Return a string representation of the ragged tensor.
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

  /* Convert a ragged tensor to given scalar type.

     Note: If the this tensor is already of the given type, itself
     is returned. Otherwise, a copy of the this tensor converted to the given
     type is returned.

     @param scalar_type The type this tensor should be converted to.

     @return Return a ragged tensor with the specified type.
   */
  RaggedAny To(torch::ScalarType scalar_type) const;

  // Return a copy of this ragged tensor
  RaggedAny Clone() const;

  /* Enable/Disable requires_grad of this tensor

     @param requires_grad True to requires grad for this tensors.
                          False to not require grad.
   */
  void SetRequiresGrad(bool requires_grad);

  // TODO: Add const
  // TODO: Return a RaggedAny
  torch::Tensor Sum(float initial_value = 0) /*const*/;
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_RAGGED_ANY_H
