// k2/csrc/fsa_renderer_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_renderer.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace k2 {

// NOTE(fangjun): this test always passes.
// Its purpose is to get a Graphviz representation
// of the fsa and convert it to a viewable format **offline**.
//
// For example, you can run
//
//  ./k2/csrc/fsa_renderer_test 2>&1 >/dev/null | dot -Tpdf > test.pdf
//
// and then open the generated "test.pdf" to verify FsaRenderer works
// as expected.
TEST(FsaRenderer, Render) {
  std::vector<Arc> arcs = {
      {0, 1, 2}, {0, 2, 1}, {1, 2, 0}, {1, 3, 5}, {2, 3, 6},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 4, 5};

  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);

  FsaRenderer renderer(fsa);
  std::cerr << renderer.Render();
}

}  // namespace k2
