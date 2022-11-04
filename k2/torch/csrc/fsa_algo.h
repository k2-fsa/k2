/**
 * Copyright      2021  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef K2_TORCH_CSRC_FSA_ALGO_H_
#define K2_TORCH_CSRC_FSA_ALGO_H_

#include "k2/csrc/fsa.h"
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/nbest.h"

namespace k2 {

/* Create a CTC topology.

   Note:
     A standard CTC topology is the conventional one, where there
     is a mandatory blank between two repeated neighboring symbols.
     A non-standard, i.e., modified CTC topology, imposes no such constraint.

   @param max_token  The maximum token ID (inclusive). We assume that token IDs
                     are contiguous (from 1 to `max_token`). 0 represents blank.
   @param modified  If False, create a standard CTC topology. Otherwise, create
                    a modified CTC topology.
   @param device  A torch.device indicating what device the returned Fsa will
                  be. Default torch::CPU.
   @return  Return either a standard or a modified CTC topology as an FSA
            depending on whether `modified` is false or true.
 */
FsaClass CtcTopo(int32_t max_token, bool modified = false,
                 torch::Device device = torch::kCPU);

/*
  Create a trivial graph which has only two states. On state 0, there are
  `max_token` self loops(i.e. a loop for each symbol from 1 to max_token), and
  state 1 is the final state.

    @param [in] max_token  The maximum token ID (inclusive). We assume that
                           token IDs are contiguous (from 1 to `max_token`).
    @param device  A torch.device indicating what device the returned Fsa will
                   be. Default torch::CPU.
    @return    Returns the expected trivial graph on the given device.
               Note the returned graph does not contain arcs with label being 0.
 */
FsaClass TrivialGraph(int32_t max_token, torch::Device device = torch::kCPU);

/* Intersect a DenseFsaVec constructed from nnet_output with an FsaClass, i.e.,
   decoding graphs.

     @param graphs Input FsaClass containing decoding graphs and the associated
                   attributes. The decoding graph might just be a linear
                   sequence of phones, or might be something more complicated.
                   Must have either `graph.fsa.shape[0] == dense.dim0()`, or
                   `graphs.fsa.shape[0] == 1` in which case the graph is shared.
     @param dense Input FSAs that correspond to neural network output.
     @param search_beam Decoding beam, e.g. 20.  Smaller is faster, larger is
                        more exact (less pruning). This is the default value; it
                        may be modified by `min_active_states` and
                       `max_active_states`.
     @param output_beam Pruning beam for the output of intersection (vs. best
                        path); equivalent to kaldi's lattice-beam. E.g. 8.
     @param max_active_states Maximum number of FSA states that are allowed to
                              be active on any given frame for any given
                              intersection/composition task. This is advisory,
                              in that it will try not to exceed that but may not
                              always succeed. You can use a very large number if
                              no constraint is needed.
     @param min_active_states Minimum number of FSA states that are allowed to
                              be active on any given frame for any given
                              intersection/composition task. This is advisory,
                              in that it will try not to have fewer than this
                              number active. Set it to zero if there is no
                              constraint.
   @return  Returns an FsaClass containing the intersection of DenseFsaVec and
            decoding graphs with the attributes propagated.
 */
FsaClass IntersectDensePruned(FsaClass &graphs, DenseFsaVec &dense,
                              float search_beam, float output_beam,
                              int32_t min_activate_states,
                              int32_t max_activate_states);

/* Return the shortest paths as linear FSAs from the start state
   to the final state in the tropical semiring.

   Note:
     It uses the opposite sign. That is, It uses `max` instead of `min`.

   @param lattice The input FsaClass.
   @return An FsaClass containing the best paths as linear FSAs with the
           attributes propagated.
 */
FsaClass ShortestPath(FsaClass &lattice);


/*
  Return array of total scores (one per FSA)
      @param [in] fsas   Input FsaVec (must have 3 axes)
      @param [in] log_semiring   If true, combine path with LogAdd
                  (i.e., mathematically, `log(exp(a)+exp(b))`); if false,
                   combine as `max(a,b)`.
      @return  Returns array of total scores, of dimension fsas.fsa.Dim0(),
                which will contain the scores in the final-states of
                `forward_scores`, or -infinity for FSAs that had no
                states.
*/
template <typename FloatType>
Array1<FloatType> GetTotScores(FsaClass &fsa, bool log_semiring = true);


/** Swap the labels and aux labels of a lattice.

   Caution: This is an in-place operation.

   @param lattice The input/output lattice. It has to have
                  an attribute "aux_labels".
 */
void Invert(FsaClass *lattice);

/** Arc sort an FSA in place.

  @param lattice The input/output lattice.
 */
void ArcSort(FsaClass *lattice);

/** Trim an FSA in place.

  @param lattice The input/output lattice.
 */
void Connect(FsaClass *lattice);

/** TopSort an FSA in place.

  @param lattice The input/output lattice.
 */
void TopSort(FsaClass *lattice);

/** Sample num_paths from the given lattice.
    @param lattice The input lattice to be sampled from.
    @param num_paths  Number of paths to sample

    @return Return a nbest object containing the sampled paths.
 */
Nbest RandomPaths(FsaClass &lattice, int32_t num_paths);

/// Wrapper for k2::IntersectDevice() in k2/csrc/fsa_algo.h
/// to support attribute propagation.
FsaClass IntersectDevice(FsaClass &a_fsas, FsaClass &b_fsas,
                         const Array1<int32_t> &b_to_a_map,
                         bool sorted_match_a);

/** Create a linear FSA with epsilon self-loops by first removing epsilon
    transitions from the input linear FSA.

      @param [in] fsas  An FSA or an FsaVec. It MUST be a linear FSA or a vector
                        of linear FSAs.
      @return  Return an FSA or FsaVec, where each FSA contains epsilon
               self-loops but contains no epsilon transitions for arcs that are
               not self-loops.
 */
FsaClass LinearFsaWithSelfLoops(FsaClass &fsas);
}  // namespace k2

#define IS_IN_K2_TORCH_CSRC_FSA_ALGO_H_
#include "k2/torch/csrc/fsa_algo_inl.h"
#undef IS_IN_K2_TORCH_CSRC_FSA_ALGO_H_

#endif  // K2_TORCH_CSRC_FSA_ALGO_H_
