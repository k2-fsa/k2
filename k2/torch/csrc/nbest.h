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

#ifndef K2_TORCH_CSRC_NBEST_H_
#define K2_TORCH_CSRC_NBEST_H_

#include <sstream>
#include <string>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/torch/csrc/fsa_class.h"

namespace k2 {

/*

An Nbest object contains two fields:

    (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
             Its axes are [path][state][arc]
    (2) shape. Its type is :class:`k2::RaggedShape`.
               Its axes are [utt][path]

The field `shape` has two axes [utt][path]. `shape.Dim0` contains
the number of utterances, which is also the number of rows in the
supervision_segments. `shape.tot_size(1)` contains the number
of paths, which is also the number of FSAs in `fsa`.

Caution:
  Don't be confused by the name `Nbest`. The best in the name `Nbest`
  has nothing to do with `best scores`. The important part is
  `N` in `Nbest`, not `best`.
 */
struct Nbest {
  FsaClass fsa;
  RaggedShape shape;

  Nbest(const FsaClass &fsa, const RaggedShape &shape);

  // Return a string representation of this object
  // in the form
  // Nbest(num_utteraces=xxx, num_paths=xxx)
  std::string ToString() const {
    std::ostringstream os;
    os << "Nbest(num_utterances=" << shape.Dim0()
       << ", num_paths=" << shape.NumElements() << ")";
    return os.str();
  }
  /** Construct an Nbest object by sampling num_paths from a lattice.

      @param lattice The input/output lattice to be sampled.
      @param num_paths  Number of paths to sample.
      @param nbest_scale  Scale lattice.scores by this value before
                          sampling.
      @return Return an Nbest object containing the sampled paths, with
              duplicated paths being removed.
   */
  static Nbest FromLattice(FsaClass &lattice, int32_t num_paths,
                           float nbest_scale = 0.5);

  /// Intersect this object with a lattice to assign scores
  /// `this` nbest.
  ///
  ///  @param lattice The lattice to intersect. Note it is modified in-place.
  ///                 You should not use it after invoking this function.
  ///
  /// Note: The scores for the return value of FromLattice() are
  ///  all 0s.
  void Intersect(FsaClass *lattice);

  /// Compute the AM scores of each path
  /// Return a 1-D torch.float32 tensor with dim equal to fsa.Dim0()
  torch::Tensor ComputeAmScores() /*const*/;

  /// Compute the LM scores of each path
  /// Return a 1-D torch.float32 tensor with dim equal to fsa.Dim0()
  torch::Tensor ComputeLmScores() /*const*/;
};

}  // namespace k2

#endif  // K2_TORCH_CSRC_NBEST_H_
