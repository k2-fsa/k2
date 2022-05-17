/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <fstream>

#include "gtest/gtest.h"
#include "k2/torch/csrc/parse_options.h"

namespace k2 {

struct MyOptions {
  bool b = false;
  int32_t i32 = 1;
  uint32_t u32 = 2;
  float f = 3;
  double d = 4;
  std::string s;

  void Register(ParseOptions *po) {
    po->Register("my-bool", &b, "A bool variable in MyOptions.");
    po->Register("my-i32", &i32, "An int32 variable in MyOptions.");

    po->Register("my-u32", &u32, "An uint32 variable in MyOptions.");

    po->Register("my-f", &f, "A float variable in MyOptions.");

    po->Register("my-d", &d, "A double variable in MyOptions.");

    po->Register("my-s", &s, "A string variable in MyOptions.");
  }
};

TEST(ParseOptions, FromCommandline) {
  int32_t a;
  double d;
  const char *const argv[] = {"./a.out",      "--my-bool=1", "--my-i32=100",
                              "--my-u32=8",   "--my-f=0.5",  "--my-d=1.5",
                              "--my-s=hello", "--a=3",       "--d=-1.25",
                              "--print-args", "foo",         "bar"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  ParseOptions po("Test parsing from the commandline");
  MyOptions opts;
  opts.Register(&po);
  po.Register("a", &a, "An integer variable");
  po.Register("d", &d, "A double variable");
  po.Read(argc, argv);

  EXPECT_EQ(a, 3);
  EXPECT_EQ(d, -1.25);
  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, 100);
  EXPECT_EQ(opts.u32, 8);
  EXPECT_EQ(opts.f, 0.5);
  EXPECT_EQ(opts.d, 1.5);
  EXPECT_EQ(opts.s, "hello");

  EXPECT_EQ(po.NumArgs(), 2);
  EXPECT_EQ(po.GetArg(1), "foo");
  EXPECT_EQ(po.GetArg(2), "bar");
}

TEST(ParseOptions, FromCommandlineWithPrefix) {
  int32_t a;
  double d;
  const char *const argv[] = {"./a.out",
                              "--print-args",
                              "--k2.my-bool=1",
                              "--k2.my-i32=100",
                              "--k2.my-u32=8",
                              "--k2.my-f=0.5",
                              "--k2.my-d=1.5",
                              "--k2.my-s=hello",
                              "--a=3",
                              "--d=-1.25",
                              "foo",
                              "bar"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  ParseOptions po("Test parsing from the commandline with prefix");
  ParseOptions po2("k2", &po);
  MyOptions opts;
  opts.Register(&po2);
  po.Register("a", &a, "An integer variable");
  po.Register("d", &d, "A double variable");
  po.Read(argc, argv);

  EXPECT_EQ(a, 3);
  EXPECT_EQ(d, -1.25);
  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, 100);
  EXPECT_EQ(opts.u32, 8);
  EXPECT_EQ(opts.f, 0.5);
  EXPECT_EQ(opts.d, 1.5);
  EXPECT_EQ(opts.s, "hello");

  EXPECT_EQ(po.NumArgs(), 2);
  EXPECT_EQ(po.GetArg(1), "foo");
  EXPECT_EQ(po.GetArg(2), "bar");
}

TEST(ParseOptions, FromCommandlineWithTwoPrefixes) {
  int32_t a;
  double d;
  const char *const argv[] = {"./a.out",
                              "--print-args",
                              "--k2.torch.my-bool=1",
                              "--k2.torch.my-i32=100",
                              "--k2.torch.my-u32=8",
                              "--k2.torch.my-f=0.5",
                              "--k2.torch.my-d=1.5",
                              "--k2.torch.my-s=hello",
                              "--a=3",
                              "--d=-1.25",
                              "foo",
                              "bar"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  ParseOptions po("Test parsing from the commandline with two prefixes");
  ParseOptions po2("k2", &po);
  ParseOptions po3("torch", &po2);
  MyOptions opts;
  opts.Register(&po3);
  po.Register("a", &a, "An integer variable");
  po.Register("d", &d, "A double variable");
  po.Read(argc, argv);

  EXPECT_EQ(a, 3);
  EXPECT_EQ(d, -1.25);
  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, 100);
  EXPECT_EQ(opts.u32, 8);
  EXPECT_EQ(opts.f, 0.5);
  EXPECT_EQ(opts.d, 1.5);
  EXPECT_EQ(opts.s, "hello");

  EXPECT_EQ(po.NumArgs(), 2);
  EXPECT_EQ(po.GetArg(1), "foo");
  EXPECT_EQ(po.GetArg(2), "bar");
}

TEST(ParseOptions, ParseHelp) {
  const char *const argv[] = {"./a.out", "--help"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);

  ParseOptions po("Parse help");
  MyOptions opts;
  opts.Register(&po);

  EXPECT_EXIT(po.Read(argc, argv), testing::ExitedWithCode(0), "");
}

TEST(ParseOptions, ParseFromFile) {
  std::string filename = "my-options-for-parse-options.txt";
  {
    std::ofstream of(filename);

    of << "--my-bool=1\n";
    of << "--my-i32=-100\n";
    of << "--my-s=hello\n";
  }

  const char *const argv[] = {
      "./a.out",      "--config=my-options-for-parse-options.txt",
      "--my-u32=8",   "--my-f=0.5",
      "--my-d=1.5",   "--my-s=world",
      "--print-args", "foo",
      "bar"};

  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  ParseOptions po("Test parsing from the commandline and config file");

  MyOptions opts;
  opts.Register(&po);

  po.Read(argc, argv);

  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, -100);
  EXPECT_EQ(opts.u32, 8);
  EXPECT_EQ(opts.f, 0.5);
  EXPECT_EQ(opts.d, 1.5);
  EXPECT_EQ(opts.s, "world");  // commandline options have a higher priority

  remove(filename.c_str());
}

TEST(ParseOptions, ParseFromMultipleFiles) {
  std::string filename1 = "my-options-for-parse-options1.txt";
  std::string filename2 = "my-options-for-parse-options2.txt";
  {
    std::ofstream of(filename1);

    of << "--my-bool=1\n";
    of << "--my-i32=-100\n";
  }

  {
    std::ofstream of(filename2);

    of << "--my-s=hello\n";
  }

  const char *const argv[] = {"./a.out",
                              "--config=my-options-for-parse-options1.txt",
                              "--config=my-options-for-parse-options2.txt",
                              "--my-u32=8",
                              "--my-f=0.5",
                              "--my-d=1.5",
                              "--print-args",
                              "foo",
                              "bar"};

  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  ParseOptions po("Test parsing from the commandline and config files");

  MyOptions opts;
  opts.Register(&po);

  po.Read(argc, argv);

  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, -100);
  EXPECT_EQ(opts.u32, 8);
  EXPECT_EQ(opts.f, 0.5);
  EXPECT_EQ(opts.d, 1.5);
  EXPECT_EQ(opts.s, "hello");

  remove(filename1.c_str());
  remove(filename2.c_str());
}

TEST(ParseOptions, Duplicates) {
  int32_t a = 10;
  int32_t b = 20;
  ParseOptions po("Test duplicates");
  po.Register("i", &a, "My integer option");
  po.Register("i", &b, "My integer option");
  // The second one is ignored
  const char *const argv[] = {"./a.out", "--i=3"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);
  po.Read(argc, argv);

  EXPECT_EQ(a, 3);
  EXPECT_EQ(b, 20);
  EXPECT_EQ(po.NumArgs(), 0);
}

TEST(ParseOptions, DoubleDash) {
  int32_t a = 10;

  const char *const argv[] = {"./a.out", "--i=3", "--", "--foo=bar", "baz"};
  int32_t argc = sizeof(argv) / sizeof(argv[0]);

  ParseOptions po("Test double dash");
  po.Register("i", &a, "My integer option");
  po.Read(argc, argv);

  EXPECT_EQ(a, 3);
  EXPECT_EQ(po.NumArgs(), 2);
  EXPECT_EQ(po.GetArg(1), "--foo=bar");
  EXPECT_EQ(po.GetArg(2), "baz");
}

TEST(ReadConfigFromFile, OneOption) {
  std::string filename = "my-options-for-parse-options.txt";
  {
    std::ofstream of(filename);

    of << "--my-bool=1\n";
    of << "--my-i32=-100\n";
    of << "--my-u32=1000\n";
    of << "--my-f=-0.5\n";
    of << "--my-d=3.5\n";
    of << "--my-s=hello world\n";
  }
  MyOptions opts;
  ReadConfigFromFile(filename, &opts);

  EXPECT_EQ(opts.b, true);
  EXPECT_EQ(opts.i32, -100);
  EXPECT_EQ(opts.u32, 1000);
  EXPECT_EQ(opts.f, -0.5);
  EXPECT_EQ(opts.d, 3.5);
  EXPECT_EQ(opts.s, "hello world");

  remove(filename.c_str());
}

}  // namespace k2
