/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "gtest/gtest.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/test_deserialization_data.h"

#ifdef K2_WITH_CUDA
#include "torch/cuda.h"
#endif

namespace k2 {

// defined in k2/torch/csrc/deserialization.cu
Ragged<int32_t> ToRaggedInt(torch::IValue value);

static void TestDictOfTensorIntStr(const std::string &dir_name) {
  std::string filename = dir_name + "/d1.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData1),
             sizeof(kTestLoadData1));
  }
  // d1.pt contains
  // {"a": torch.tensor([1., 2.]), "b": 10, "c": "k2"}
  torch::IValue ivalue = Load(filename);
  EXPECT_TRUE(ivalue.isGenericDict());
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();

  EXPECT_TRUE(dict.contains("a"));
  EXPECT_TRUE(dict.contains("b"));
  EXPECT_TRUE(dict.contains("c"));

  torch::Tensor a = dict.at("a").toTensor();
  int32_t b = dict.at("b").toInt();
  const std::string &c = dict.at("c").toStringRef();

  EXPECT_TRUE(a.allclose(torch::tensor({1, 2}, a.options())));
  EXPECT_EQ(b, 10);
  EXPECT_EQ(c, "k2");

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

static void TestDictOfTensorAndRaggedTensor(const std::string &dir_name) {
  std::string filename = dir_name + "/d2.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData2),
             sizeof(kTestLoadData2));
  }
  // d2.pt contains
  // {"a": torch.tensor([1.0, 2.0]), "b": k2.RaggedTensor([[1.5, 2], [3], []])}
  torch::IValue ivalue = Load(filename);
  EXPECT_TRUE(ivalue.isGenericDict());
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();

  EXPECT_TRUE(dict.contains("a"));
  EXPECT_TRUE(dict.contains("b"));

  torch::Tensor a = dict.at("a").toTensor();
  Ragged<int32_t> b = ToRaggedInt(dict.at("b"));

  EXPECT_TRUE(a.allclose(torch::tensor({1, 2}, a.options())));
  EXPECT_TRUE(Equal(b, Ragged<int32_t>("[[15 2] [3] []]")));

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

#ifdef K2_WITH_CUDA
static void TestDictOfCudaTensorAndCudaRaggedTensor(
    const std::string &dir_name) {
  std::string filename = dir_name + "/d3.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData3),
             sizeof(kTestLoadData3));
  }
  // d3.pt contains:
  // {
  //  "a": torch.tensor([1, 2], device=torch.device("cuda:0")),
  //  "b": k2.RaggedTensor([[15, 2], [3], []], device="cuda:0"),
  // }
  torch::IValue ivalue = Load(filename);
  EXPECT_TRUE(ivalue.isGenericDict());
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();

  EXPECT_TRUE(dict.contains("a"));
  EXPECT_TRUE(dict.contains("b"));

  torch::Tensor a = dict.at("a").toTensor();
  Ragged<int32_t> b = ToRaggedInt(dict.at("b"));

  EXPECT_TRUE(a.is_cuda());

  EXPECT_TRUE(a.allclose(torch::tensor({1, 2}, a.options())));
  EXPECT_TRUE(Equal(b, Ragged<int32_t>("[[15 2] [3] []]").To(b.Context())));

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

static void TestLoadFsaCuda(const std::string &dir_name) {
  std::string filename = dir_name + "/d4.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData4),
             sizeof(kTestLoadData4));
  }
  // d4.pt contains:
  // { 'arcs': tensor([[0,1,-1,1036831949]],device='cuda:0',dtype=torch.int32),
  //  'aux_labels': RaggedTensor([[1, 2]], device='cuda:0', dtype=torch.int32),
  //  'attr': tensor([1.5000], device='cuda:0')
  // }
  FsaClass fsa = LoadFsa(filename);
  auto device = DeviceFromContext(fsa.fsa.Context());
  EXPECT_EQ(device, torch::Device("cuda:0"));

  auto attr = fsa.GetTensorAttr("attr");
  EXPECT_TRUE(attr.allclose(torch::tensor({1.5}, attr.options())));
  EXPECT_TRUE(Equal(fsa.GetRaggedTensorAttr("aux_labels"),
                    Ragged<int32_t>("[[1 2]]").To(fsa.fsa.Context())));

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

#endif

static void TestDictOfTensorAndRaggedTensorMapToCpu(
    const std::string &dir_name) {
  std::string filename = dir_name + "/d3.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData3),
             sizeof(kTestLoadData3));
  }
  // d3.pt contains:
  // {
  //  "a": torch.tensor([1, 2], device=torch.device("cuda:0")),
  //  "b": k2.RaggedTensor([[15, 2], [3], []], device="cuda:0"),
  // }
  torch::IValue ivalue = Load(filename, /*map_location*/ torch::kCPU);
  EXPECT_TRUE(ivalue.isGenericDict());
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();

  EXPECT_TRUE(dict.contains("a"));
  EXPECT_TRUE(dict.contains("b"));

  torch::Tensor a = dict.at("a").toTensor();
  Ragged<int32_t> b = ToRaggedInt(dict.at("b"));
  EXPECT_FALSE(a.is_cuda());

  EXPECT_TRUE(a.allclose(torch::tensor({1, 2}, a.options())));
  EXPECT_TRUE(Equal(b, Ragged<int32_t>("[[15 2] [3] []]")));

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

static void TestLoadFsaMapToCpu(const std::string &dir_name) {
  std::string filename = dir_name + "/d4.pt";
  {
    std::ofstream os(filename, std::ofstream::binary);
    os.write(reinterpret_cast<const char *>(kTestLoadData4),
             sizeof(kTestLoadData4));
  }
  // d4.pt contains:
  // { 'arcs': tensor([[0,1,-1,1036831949]],device='cuda:0',dtype=torch.int32),
  //  'aux_labels': RaggedTensor([[1, 2]], device='cuda:0', dtype=torch.int32),
  //  'attr': tensor([1.5000], device='cuda:0')
  // }
  FsaClass fsa = LoadFsa(filename, torch::kCPU);
  auto device = DeviceFromContext(fsa.fsa.Context());
  EXPECT_EQ(device, torch::Device(torch::kCPU));

  auto attr = fsa.GetTensorAttr("attr");
  EXPECT_TRUE(attr.allclose(torch::tensor({1.5}, attr.options())));
  EXPECT_TRUE(Equal(fsa.GetRaggedTensorAttr("aux_labels"),
                    Ragged<int32_t>("[[1 2]]").To(fsa.fsa.Context())));

  int32_t ret = remove(filename.c_str());
  assert(ret == 0);
}

TEST(Deserialization, Test) {
  char pattern[] = "/tmp/k2_test.XXXXXX";
#ifndef _MSC_VER
  char *dir_name = mkdtemp(pattern);
#else
  char *dir_name = "./";
#endif
  assert(dir_name != nullptr);

  TestDictOfTensorIntStr(dir_name);
  TestDictOfTensorAndRaggedTensor(dir_name);
  TestDictOfTensorAndRaggedTensorMapToCpu(dir_name);
  TestLoadFsaMapToCpu(dir_name);

#ifdef K2_WITH_CUDA
  if (torch::cuda::is_available()) {
    TestDictOfCudaTensorAndCudaRaggedTensor(dir_name);
    TestLoadFsaCuda(dir_name);
  }
#endif

#ifndef _MSC_VER
  int ret = rmdir(dir_name);
  assert(ret == 0);
#endif
}

}  // namespace k2
