/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Wei Kang)
 *
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

#include <string>

#include "gtest/gtest.h"
#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/pybind_utils.h"
#include "pybind11/embed.h"

namespace k2 {

TEST(PybindTest, ToPyObject) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    torch::IValue ivalue("hello k2");
    auto py_value = ToPyObject(ivalue);
    EXPECT_EQ(py_value.cast<std::string>(), "hello k2");

    ivalue = torch::IValue(10);
    py_value = ToPyObject(ivalue);
    EXPECT_EQ(py_value.cast<int>(), 10);

    ivalue = torch::IValue(10.1);
    py_value = ToPyObject(ivalue);
    EXPECT_LT(std::fabs(py_value.cast<float>() - 10.1), 1e-6);

    RaggedAny ragged("[[1 2] [] [3 4]]", torch::kInt32, device);
    ivalue = torch::make_custom_class<k2::RaggedAnyHolder>(
        std::make_shared<RaggedAny>(ragged));
    py_value = ToPyObject(ivalue);
    EXPECT_EQ(py_value.cast<RaggedAny>().ToString(), ragged.ToString());
  }
}

TEST(PybindTest, ToIValue) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    py::object py_value = py::str("hello k2");
    auto ivalue = ToIValue(py_value);
    EXPECT_EQ(ivalue.toStringRef(), "hello k2");

    py_value = py::int_(10);
    ivalue = ToIValue(py_value);
    EXPECT_EQ(ivalue.toInt(), 10);

    py_value = py::float_(10.1);
    ivalue = ToIValue(py_value);
    EXPECT_LT(std::fabs(ivalue.toDouble() - 10.1), 1e-6);

    RaggedAny ragged("[[1 2] [] [3 4]]", torch::kInt32, device);
    py_value = py::cast(ragged);
    ivalue = ToIValue(py_value);
    torch::intrusive_ptr<RaggedAnyHolder> ragged_any_holder =
        ivalue.toCustomClass<RaggedAnyHolder>();
    EXPECT_EQ((*(ragged_any_holder->ragged)).ToString(), ragged.ToString());
  }
}

}  // namespace k2

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  py::scoped_interpreter guard{};
  py::module_::import("torch");
  py::module_::import("_k2");
  return RUN_ALL_TESTS();
}
