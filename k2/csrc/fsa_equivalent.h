// k2/csrc/fsa_equivalent.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include <cstdint>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/weights.h"

#ifndef K2_CSRC_FSA_EQUIVALENT_H_
#define K2_CSRC_FSA_EQUIVALENT_H_

namespace k2 {

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if the
  paths exist in the other one.
 */
bool IsRandEquivalent(const Fsa &a, const Fsa &b, std::size_t npath = 100);

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if each path
  exists in the other one and the sum of weights along that path are the same.

  @param [in]  a          One of the FSAs to be checked the equivalence
  @param [in]  a_weights  Arc weights of `a`
  @param [in]  b          The other FSA to be checked the equivalence
  @param [in]  b_weights  Arc weights of `b`
  @param [in]  beam       beam > 0 that affects pruning; the algorithm
                          will only check paths within `beam` of the
                          best path(for tropical semiring, it's max
                          weight over all paths from start state to
                          final state; for log semiring, it's log-sum probs
                          over all paths) in `a` or `b`. That is,
                          any symbol sequence, whose total weights
                          over all paths are within `beam` of the best
                          path (either in `a` or `b`), must have
                          the same weights in `a` and `b`.
                          There is no any requirement on symbol sequences
                          whose total weights over paths are outside `beam`.
                          Just keep `kFloatInfinity` if you don't want pruning.
  @param [in]  delta      Tolerance for path weights to check the equivalence.
                          If abs(weights_a, weights_b) <= delta, we say the two
                          paths are equivalent.
  @param [in]  top_sorted The user may set this to true if both `a` and `b` are
                          topologically sorted; this makes this function faster.
                          Otherwise it must be set to false.
  @param [in]  npath      The number of paths will be generated to check the
                          equivalence of `a` and `b`
 */
template <FbWeightType Type>
bool IsRandEquivalent(const Fsa &a, const float *a_weights, const Fsa &b,
                      const float *b_weights, float beam = kFloatInfinity,
                      float delta = 1e-6, bool top_sorted = true,
                      std::size_t npath = 100);

/*
  This version of `IsRandEquivalent` will be used to check the equivalence
  between the input FSA `a` and the output FSA `b` of `RmEpsilonPrunedLogSum`.
  We need this version because `RmEpsilonPrunedLogSum` does pruning on the input
  FSA instead of on the output FSA (compared with other algorithm such as
  determinization), thus it will goes against the symmetric assumption of
  IsRandEquivalent as `a` may always has greater weights than `b` for any symbol
  sequence.

  Specifically, we implement the algorithm as below:
    1. any path in `b` that is within `beam` of the total weight of `b` should
       be present in `a` with a weight that is not greater than its weight in
       `b`;
    2. any path (without epsilon arcs) in `a` that is within `beam` of the total
       weight of `a` should be present in `b` with a weight that is not less
       than its weight in `a`.

  The function returns true if `a` is stochastically equivalent to `b` by
  randomly generating `npath` paths from one of them and then checking if each
  path exists in the other one and if the relationship of two weights satisfies
  above rules.

  @param [in]  a          The FSA before epsion-removal
                          (input FSA of `RmEpsilonPrunedLogSum`)
  @param [in]  a_weights  Arc weights of `a`
  @param [in]  b          The FSA after epsion-removal
                          (output FSA of `RmEpsilonPrunedLogSum`)
  @param [in]  b_weights  Arc weights of `b`
  @param [in]  beam       beam > 0 that affects pruning; the algorithm
                          will only check paths within `beam` of the
                          total weight of `a` or `b`. The value of `beam`
                          should also be not greater than the `beam` value
                          used in `RmEpsilonPrunedLogSum`.
  @param [in]  top_sorted The user may set this to true if both `a` and `b` are
                          topologically sorted; this makes this function faster.
                          Otherwise it must be set to false.
  @param [in]  npath      The number of paths will be generated to check the
                          equivalence of `a` and `b`
 */
bool IsRandEquivalentAfterRmEpsPrunedLogSum(
    const Fsa &a, const float *a_weights, const Fsa &b, const float *b_weights,
    float beam, bool top_sorted = true, std::size_t npath = 100);

/*
  Gets a random path from an Fsa `a`, returns true if we get one path
  successfully.

  @param [in]  a         The input fsa from which we will generate a random path
  @param [out] b         The output path
  @param [out] state_map If non-NULL, this function will output a map from the
                         state-index in `b` to the corresponding state-index in
  `a`.
*/
bool RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32_t> *state_map = nullptr);

// Version of RandomPath that requires that there's no epsilon arc in the
// returned path.
bool RandomPathWithoutEpsilonArc(const Fsa &a, Fsa *b,
                                 std::vector<int32_t> *state_map = nullptr);
/*
  Computes the intersection of two FSAs where one FSA has weights on arc.

  @param [in] a    One of the FSAs to be intersected.  Must satisfy
                   ArcSorted(a)
  @param [in] a_weights Arc weights of `a`
  @param [in] b    The other FSA to be intersected  Must satisfy
                   ArcSorted(b) and IsEpsilonFree(b). It is usually a path
                   generated from `RandomNonEpsilonPath`
  @param [out] c   The composed FSA will be output to here.
  @param [out] c_weights Arc weights of output FSA `c`.
  @param [out] arc_map_a   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `a` was, `-1` represents
                   there is no corresponding source arc in `a`.
  @param [out] arc_map_b   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `b` was, `-1` represents
                   there is no corresponding source arc in `b`.
 */
void Intersect(const Fsa &a, const float *a_weights, const Fsa &b, Fsa *c,
               std::vector<float> *c_weights,
               std::vector<int32_t> *arc_map_a = nullptr,
               std::vector<int32_t> *arc_map_b = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_EQUIVALENT_H_
