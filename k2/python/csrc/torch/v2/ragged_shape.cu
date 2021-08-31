/**
 * @brief python wrapper for RaggedShape
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

#include "k2/csrc/device_guard.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/doc/ragged_shape.h"
#include "k2/python/csrc/torch/v2/ragged_shape.h"

namespace k2 {

void PybindRaggedShape(py::module &m) {
  py::class_<RaggedShape> shape(m, "RaggedShape");

  // Construct a ragged shape from a string, e.g.,
  // [ [x x] [x] [] ]
  shape.def(py::init<const std::string &>(), py::arg("s"), kRaggedShapeInitStr);

  shape.def_property_readonly("dim0", &RaggedShape::Dim0, kRaggedShapeDim0Doc);

  shape.def(
      "max_size",
      [](RaggedShape &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.MaxSize(axis);
      },
      py::arg("axis"), kRaggedShapeMaxSizeDoc);

  shape.def(
      "numel",
      [](RaggedShape &self) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.NumElements();
      },
      kRaggedShapeNumelDoc);

  shape.def(
      "tot_size",
      [](RaggedShape &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.TotSize(axis);
      },
      py::arg("axis"), kRaggedShapeTotSizeDoc);

  shape.def(
      "to",
      [](const RaggedShape &self, torch::Device device) -> RaggedShape {
        DeviceGuard guard(self.Context());

        if (device.type() == torch::kCPU) return self.To(GetCpuContext());

        K2_CHECK(device.is_cuda());
        return self.To(GetCudaContext(device.index()));
      },
      py::arg("device"), kRaggedShapeToDeviceDoc);

  shape.def(
      "row_ids",
      [](RaggedShape &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTorch(row_ids);
      },
      py::arg("axis"), kRaggedShapeRowIdsDoc);

  shape.def(
      "row_splits",
      [](RaggedShape &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTorch(row_splits);
      },
      py::arg("axis"), kRaggedShapeRowSplitsDoc);

  shape.def(
      "tot_sizes",
      [](const RaggedShape &self) -> py::tuple {
        DeviceGuard guard(self.Context());
        int32_t num_axes = self.NumAxes();
        py::tuple ans(num_axes);
        for (int32_t i = 0; i != num_axes; ++i) ans[i] = self.TotSize(i);
        return ans;
      },
      kRaggedShapeTotSizesDoc);

  shape.def(
      "__eq__",
      [](const RaggedShape &self, const RaggedShape &other) -> bool {
        DeviceGuard guard(self.Context());
        return Equal(self, other);
      },
      py::arg("other"), kRaggedShapeEqDoc);

  shape.def(
      "__ne__",
      [](const RaggedShape &self, const RaggedShape &other) -> bool {
        DeviceGuard guard(self.Context());
        return !Equal(self, other);
      },
      py::arg("other"), kRaggedShapeNeDoc);

  shape.def(
      "__str__",
      [](const RaggedShape &self) -> std::string {
        DeviceGuard guard(self.Context());
        std::ostringstream os;
        os << self;
        return os.str();
      },
      kRaggedShapeStrDoc);

  shape.def(
      "__repr__",
      [](const RaggedShape &self) -> std::string {
        DeviceGuard guard(self.Context());
        std::ostringstream os;
        os << self;
        return os.str();
      },
      kRaggedShapeStrDoc);

  shape.def(
      "__getitem__",
      [](RaggedShape &self, int32_t i) -> RaggedShape {
        DeviceGuard guard(self.Context());
        RaggedShape ans = self.Index(/*axis*/ 0, i, /*value_offset*/ nullptr);
        return ans;
      },
      py::arg("i"), kRaggedShapeGetItemDoc);

  shape.def_property_readonly(
      "num_axes",
      [](const RaggedShape &self) -> int32_t { return self.NumAxes(); },
      kRaggedShapeNumAxesDoc);

  shape.def_property_readonly(
      "device",
      [](const RaggedShape &self) -> py::object {
        DeviceType d = self.Context()->GetDeviceType();
        torch::DeviceType device_type = ToTorchDeviceType(d);

        torch::Device device(device_type, self.Context()->GetDeviceId());

        PyObject *ptr = THPDevice_New(device);

        // takes ownership
        return py::reinterpret_steal<py::object>(ptr);
      },
      kRaggedShapeDeviceDoc);

  shape.def_static(
      "regular_ragged_shape",
      [](int32_t dim0, int32_t dim1) -> RaggedShape {
        ContextPtr c = GetCpuContext();
        return RegularRaggedShape(c, dim0, dim1);
      },
      py::arg("dim0"), py::arg("dim1"));

  m.attr("regular_ragged_shape") = shape.attr("regular_ragged_shape");

  shape.def("get_layer",
            [](const RaggedShape &self, int32_t layer) -> RaggedShape {
              return GetLayer(self, layer);
            });

  // return a pair:
  //  - ans (RaggedShape)
  //  - value_indexes (optional)
  //
  shape.def(
      "index",
      [](RaggedShape &self, int32_t axis, torch::Tensor indexes,
         bool need_value_indexes =
             true) -> std::pair<RaggedShape, torch::optional<torch::Tensor>> {
        DeviceGuard guard(self.Context());
        Array1<int32_t> indexes_array = FromTorch<int32_t>(indexes);
        Array1<int32_t> value_indexes;
        RaggedShape ans = Index(self, axis, indexes_array,
                                need_value_indexes ? &value_indexes : nullptr);

        torch::optional<torch::Tensor> value_indexes_tensor;
        if (need_value_indexes) value_indexes_tensor = ToTorch(value_indexes);

        return std::make_pair(ans, value_indexes_tensor);
      },
      py::arg("axis"), py::arg("indexes"),
      py::arg("need_value_indexes") = true);
}

}  // namespace k2
