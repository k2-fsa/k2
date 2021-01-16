/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/fsa_renderer.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "k2/csrc/host/fsa_util.h"

namespace k2host {

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
      {0, 1, 2, 1}, {0, 2, 1, 2}, {1, 2, 0, 3}, {1, 3, 5, 4}, {2, 3, 6, 5},
  };

  FsaCreator fsa_creator(arcs, 3);
  const auto &fsa = fsa_creator.GetFsa();

  FsaRenderer renderer(fsa);
  std::cerr << renderer.Render();
}

}  // namespace k2host
