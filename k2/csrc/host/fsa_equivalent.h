/**
 * @brief
 * fsa_equivalent
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cstdint>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/log.h"

#ifndef K2_CSRC_HOST_FSA_EQUIVALENT_H_
#define K2_CSRC_HOST_FSA_EQUIVALENT_H_

namespace k2host {

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if the
  paths exist in the other one.
  Noted we'll pass `treat_epsilons_specially` to Intersection in
  `host/intersect.h` to do the intersection between the random path and the
  input Fsas. Generally, if it's true, we will treat epsilons as epsilon when
  doing intersection; Otherwise, epsilons will just be treated as any other
  symbol. See `host/intersect.h` for details.
 */
bool IsRandEquivalent(const Fsa &a, const Fsa &b,
                      bool treat_epsilons_specially = true,
                      std::size_t npath = 100);

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if each path
  exists in the other one and the sum of weights along that path are the same.

  @param [in]  a          One of the FSAs to be checked the equivalence
  @param [in]  b          The other FSA to be checked the equivalence
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
  @param [in]  treat_epsilons_specially We'll pass `treat_epsilons_specially`
                          to Intersection in `host/intersect.h` to do the
                          intersection between the random path and the
                          input Fsas. Generally, if it's true, we will treat
                          epsilons as epsilon when doing intersection;
                          Otherwise, epsilons will just be treated as any other
                          symbol. See `host/intersect.h` for details.
  @param [in]  delta      Tolerance for path weights to check the equivalence.
                          If abs(weights_a, weights_b) <= delta, we say the two
                          paths are equivalent.
  @param [in]  top_sorted The user may set this to true if both `a` and `b` are
                          topologically sorted; this makes this function faster.
                          Otherwise it must be set to false.
                          Caution: we only support true for now and may remove
                          this parameter later if we find we only need to
                          support top-sorted cases.
  @param [in]  npath      The number of paths will be generated to check the
                          equivalence of `a` and `b`
 */
template <FbWeightType Type>
bool IsRandEquivalent(const Fsa &a, const Fsa &b, float beam = kFloatInfinity,
                      bool treat_epsilons_specially = true, float delta = 1e-6,
                      bool top_sorted = true, std::size_t npath = 100);

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
  @param [in]  b          The FSA after epsion-removal
                          (output FSA of `RmEpsilonPrunedLogSum`)
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
bool IsRandEquivalentAfterRmEpsPrunedLogSum(const Fsa &a, const Fsa &b,
                                            float beam, bool top_sorted = true,
                                            std::size_t npath = 100);

/*
  Gets a random path from the input FSA, returns true if we get one path
  successfully.
*/
class RandPath {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input fsa from which we will generate a random path
     @param [in] no_eps_arc  If true, the generated path must be epsilon-free,
                             i.e. there must be no arc with
                             arc.label == kEpsilon.
     @param [in] eps_arc_tries  Will be used when `no_eps_arc` is true. If all
                                leaving arcs from a particular state have no
                                labels (all arcs' labels are epsilons), we may
                                fail to find a epsilon-free path (even if
                                `fsa_in` is connected). `eps_arc_tries` controls
                                how many times we'll try to generate a labeled
                                arc from any state. Must be greater than 1.
  */
  RandPath(const Fsa &fsa_in, bool no_eps_arc, int32_t eps_arc_tries = 50)
      : fsa_in_(fsa_in),
        no_epsilon_arc_(no_eps_arc),
        eps_arc_tries_(eps_arc_tries) {
    K2_CHECK_GT(eps_arc_tries_, 1);
  }

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the path to `path` and
    arc mapping information to `arc_map` (if provided).
    @param [out]  path    Output path.
                          Must be initialized; search for 'initialized
                          definition' in class Array2 in array.h for meaning.
    @param [out] arc_map   If non-NULL, will output a map from the arc-index
                           in `fsa_out` to the corresponding arc-index in
                           `fsa_in`.
                           If non-NULL, at entry it must be allocated with
                           size num-arcs of `fsa_out`, e.g. `fsa_out->size2`.

    @return true if it succeeds; will be false if it fails,
            `fsa_out` will be empty when it fails.
   */
  bool GetOutput(Fsa *fsa_out, int32_t *arc_map = nullptr);

 private:
  const Fsa &fsa_in_;
  const bool no_epsilon_arc_;
  const int32_t eps_arc_tries_;

  bool status_;
  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out
  std::vector<int32_t> arc_map_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_FSA_EQUIVALENT_H_
