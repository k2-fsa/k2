/**
 * @brief python wrappers for array_ops.h
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.       (authors: Wei Kang)
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

#include "k2/csrc/array_ops.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/array_ops.h"

namespace k2 {

static void PybindMonotonicLowerBound(py::module &m) {
  m.def(
      "monotonic_lower_bound",
      [](torch::Tensor src, bool inplace = false) -> torch::Tensor {
        Dtype t = ScalarTypeToDtype(src.scalar_type());
        ContextPtr c = GetContext(src);
        DeviceGuard guard(c);
        FOR_REAL_AND_INT_TYPES(t, T, {
          if (src.dim() == 1) {
            Array1<T> src_array = FromTorch<T>(src);
            Array1<T> dest_array = src_array;
            if (!inplace) {
              dest_array = Array1<T>(c, src_array.Dim());
            }
            MonotonicLowerBound(src_array, &dest_array);
            return ToTorch<T>(dest_array);
          } else if (src.dim() == 2) {
            Array2<T> src_array = FromTorch<T>(src, Array2Tag{});
            Array2<T> dest_array = src_array;
            if (!inplace) {
              dest_array = Array2<T>(c, src_array.Dim0(), src_array.Dim1());
            }
            for (int32_t i = 0; i < src_array.Dim0(); ++i) {
              Array1<T> row = dest_array.Row(i);
              MonotonicLowerBound(src_array.Row(i), &row);
            }
            return ToTorch<T>(dest_array);
          } else {
            K2_LOG(FATAL)
                << "Only support 1 dimension and 2 dimensions tensor, given "
                   "dimension : "
                << src.dim();
            return torch::Tensor();
          }
        });
        // Unreachable code, to make compiler happy
        return torch::Tensor();
      },
      py::arg("src"), py::arg("inplace") = false,
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace k2

void PybindArrayOps(py::module &m) { k2::PybindMonotonicLowerBound(m); }
