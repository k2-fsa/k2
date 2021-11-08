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

#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/nbest.h"
namespace k2 {

Nbest::Nbest(const FsaClass &fsa, const RaggedShape &shape)
    : fsa(fsa), shape(shape) {
  K2_CHECK_EQ(fsa.fsa.NumAxes(), 3) << "Expect an FsaVec";
  K2_CHECK_EQ(shape.NumAxes(), 2) << "Expect a shape with axes [utt][path]";
  K2_CHECK_EQ(fsa.fsa.Dim0(), shape.NumElements());
}

Nbest Nbest::FromLattice(FsaClass &lattice, int32_t num_paths,
                         float nbest_scale /*= 0.5*/) {
  K2_CHECK_EQ(lattice.fsa.NumAxes(), 3);
  K2_CHECK_GT(num_paths, 1);

  torch::Tensor scores = lattice.Scores();
  torch::Tensor saved_scores = scores.clone();

  scores = scores * nbest_scale;
  lattice.SetScores(scores);
  Nbest ans = RandomPaths(lattice, num_paths);
  lattice.SetScores(saved_scores);
  return ans;
}

void Nbest::Intersect(FsaClass *lattice) {
  K2_CHECK_EQ(lattice->fsa.NumAxes(), 3);
  Invert(&fsa);
  // Now fsa contains word IDs as labels and aux_labels as token IDs.

  fsa.Scores().zero_();  // Just in case it has scores set

  K2_CHECK(lattice->HasTensorAttr("aux_labels") ||
           lattice->HasRaggedTensorAttr("aux_labels"));

  // We don't need the aux labels for this->fsa,
  // as we are going to use the one from lattice.
  Fsa word_fsa_with_epsilon_self_loops;
  RemoveEpsilonAndAddSelfLoops(fsa.fsa, fsa.Properties(),
                               &word_fsa_with_epsilon_self_loops);

  auto &path_to_utt_map = shape.RowIds(1);

  // The following Invert() and ArcSort() change lattice in-place
  Invert(lattice);
  // Now lattice has word IDs as labels and token IDs as aux_labels
  ArcSort(lattice);

  Array1<int32_t> arc_map_a, arc_map_b;

  Fsa path_lattice = k2::IntersectDevice(
      lattice->fsa, lattice->Properties(), word_fsa_with_epsilon_self_loops,
      FsaClass(word_fsa_with_epsilon_self_loops).Properties(), path_to_utt_map,
      &arc_map_a, &arc_map_b, true);

  FsaClass ans(path_lattice);
  ans.CopyAttrs(*lattice, k2::Array1ToTorch(arc_map_a));
  Connect(ans);
  TopSort(ans);
  ans = ShortestPath(ans);
  Invert(*ans);
  // now ans.fsa has token IDs as labels and word IDs as aux_labels.

  this->fsa = ans;
}

}  // namespace k2
