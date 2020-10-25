/**
 * @brief
 * compose_test
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/compose.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"

namespace k2host {

TEST(ComposeTest, Compose) {
  std::vector<Arc> arcs_a = {{0, 1, 1, 0.1}, {1, 2, 2, 0.2}, {2, 3, -1, 0.3}};
  FsaCreator fsa_creator_a(arcs_a, 3);
  const auto &a = fsa_creator_a.GetFsa();
  std::vector<int32_t> a_aux_labels = {0, 1, -1};

  std::vector<Arc> arcs_b = {{0, 1, 1, 10}, {1, 2, -1, 20}};
  FsaCreator fsa_creator_b(arcs_b, 2);
  const auto &b = fsa_creator_b.GetFsa();
  std::vector<int32_t> b_aux_labels = {10, 0};

  Compose compose(a, b, a_aux_labels.data(), b_aux_labels.data());
  Array2Size<int32_t> fsa_size;
  compose.GetSizes(&fsa_size);

  FsaCreator fsa_creator_out(fsa_size);
  auto &c = fsa_creator_out.GetFsa();

  std::vector<int32_t> arc_map_a(fsa_size.size2);
  std::vector<int32_t> arc_map_b(fsa_size.size2);
  std::vector<int32_t> c_aux_labels;
  bool status =
      compose.GetOutput(&c, &c_aux_labels, arc_map_a.data(), arc_map_b.data());

  EXPECT_TRUE(status);

  std::vector<Arc> arcs(c.data, c.data + c.size2);
  std::vector<Arc> arcs_c = {{0, 1, 1, 0.1}, {1, 2, 2, 10.2}, {2, 3, -1, 20.3}};
  for (std::size_t i = 0; i != arcs_c.size(); ++i)
    EXPECT_EQ(arcs[i], arcs_c[i]);

  EXPECT_THAT(c_aux_labels, ::testing::ElementsAre(0, 10, 0));
  EXPECT_THAT(arc_map_a, ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(arc_map_b, ::testing::ElementsAre(-1, 0, 1));
}

}  // namespace k2host
