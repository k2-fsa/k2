/**
 * @brief host_shim  Wrapper functions so we can use our older
 *                 CPU-only code, in host/, with the newer interfaces
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_SHIM_H_
#define K2_CSRC_HOST_SHIM_H_

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_equivalent.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/ragged.h"

namespace k2 {

/*
   Convert an Fsa (CPU only!) to k2host::Fsa.

      @param [in] fsa  FSA to convert
      @return  Returns the k2host::Fsa referring to the contents of `fsa`

   Don't let `fsa` go out of scope while you are still accessing
   the return value.
*/
k2host::Fsa FsaToHostFsa(Fsa &fsa);

/*
  Convert one element of an FsaVec (CPU only!) to a k2host::Fsa.

   @param [in] fsa_vec  The FsaVec to convert the format of one
                        element of.  Must be on CPU.
   @param [in] index    The FSA index to access, with 0 <= index <
                        fsa_vec.Dim0().
   @return   Returns a k2host::Fsa referring to the memory in `fsa_vec`.

  Warning: don't let `fsa_vec` go out of scope while you are still accessing
  the return value.
*/
k2host::Fsa FsaVecToHostFsa(FsaVec &fsa_vec, int32_t index);

class FsaCreator {
 public:
  FsaCreator() = default;
  /*
    Initialize Fsa with host::Array2size, search for 'initialized definition' in
    class Array2 in array.h for meaning. Note that we don't fill data in
    `indexes` and `data` here, the caller is responsible for this.

    `Array2Storage` is for this purpose as well, but we define this version of
    constructor here to make test code simpler.
  */
  explicit FsaCreator(const k2host::Array2Size<int32_t> &size) { Init(size); }

  void Init(const k2host::Array2Size<int32_t> &size) {
    arc_indexes_ = Array1<int32_t>(GetCpuContext(), size.size1 + 1);
    // just for case of empty Array2 object, may be written by the caller
    arc_indexes_.Data()[0] = 0;
    arcs_ = Array1<Arc>(GetCpuContext(), size.size2);
  }

  /*
    Create an Fsa from a vector of arcs.  This was copied and modified from
    similar code in host/.
    TODO(dan): maybe delete this if not needed.

     @param [in, out] arcs   A vector of arcs as the arcs of the generated Fsa.
                             The arcs in the vector should be sorted by
                             src_state.
     @param [in] final_state Will be as the final state id of the generated Fsa.
   */
  explicit FsaCreator(const std::vector<Arc> &arcs, int32_t final_state)
      : FsaCreator() {
    if (arcs.empty())
      return;  // has created an empty Fsa in the default constructor
    arcs_ = Array1<Arc>(GetCpuContext(), arcs);
    std::vector<int32_t> arc_indexes;  // == row_splits.
    int32_t curr_state = -1;
    int32_t index = 0;
    for (const auto &arc : arcs) {
      K2_CHECK_LE(arc.src_state, final_state);
      K2_CHECK_LE(arc.dest_state, final_state);
      K2_CHECK_LE(curr_state, arc.src_state);
      while (curr_state < arc.src_state) {
        arc_indexes.push_back(index);
        ++curr_state;
      }
      ++index;
    }
    // noted that here we push two `final_state` at the end, the last element is
    // just to avoid boundary check for some FSA operations.
    for (; curr_state <= final_state; ++curr_state)
      arc_indexes.push_back(index);

    arc_indexes_ = Array1<int32_t>(GetCpuContext(), arc_indexes);
  }

  Fsa GetFsa() {
    RaggedShape shape = RaggedShape2(&arc_indexes_, nullptr, arcs_.Dim());
    Fsa ans(shape, arcs_);
    return ans;
  }

  // This will be used to write the data to, i.e. the host code will
  // use the pointers in k2host::Fsa to write data to, then the user
  // will call GetFsa() to get it as an FSA.
  k2host::Fsa GetHostFsa() {
    return k2host::Fsa(arc_indexes_.Dim() - 1, arcs_.Dim(), arc_indexes_.Data(),
                       reinterpret_cast<k2host::Arc *>(arcs_.Data()));
  }

 private:
  Array1<int32_t> arc_indexes_;  // == row_splits
  Array1<Arc> arcs_;
};

class FsaVecCreator {
 public:
  /*
    Initialize FsaVecCreator with vector of host::Array2size, search for
    'initialized definition' in class Array2 in array.h for meaning. Note that
    this only sets up some metadata; most of the data will be written by the
    caller into objects returned by GetHostFsa().

    `Array2Storage` is for this purpose as well, but we define this version of
    constructor here to make test code simpler.

    `sizes` must be nonempty.
  */
  explicit FsaVecCreator(
      const std::vector<k2host::Array2Size<int32_t>> &sizes) {
    Init(sizes);
  }

  int32_t NumArcs() { return static_cast<int32_t>(arcs_.Dim()); }

  void Init(const std::vector<k2host::Array2Size<int32_t>> &sizes);

  FsaVec GetFsaVec();

  int32_t GetArcOffsetFor(int32_t fsa_idx) {
    return row_splits12_.Data()[fsa_idx];
  }

  /* This will be used to write the data to, i.e. the host code will use the
     pointers in k2host::Fsa to write data for the FSA numbered `fsa_idx`.  When
     they have all been written to, the user will call GetFsaVec() to get them
     all as one FsaVec.

     CAUTION: these host FSAs must be written to in order, i.e. for 0, 1, 2 and
     so on.  This is necessary because overlapping elements of row_splits2_ are
     written to, and writing them in order assures that the later FSA always
     'wins' the conflict.
  */
  k2host::Fsa GetHostFsa(int32_t fsa_idx);

 private:
  void FinalizeRowSplits2();

  // The row-splits1 of the result, is the exclusive-sum of the size1 elements
  // of `sizes` passed to the constructor
  Array1<int32_t> row_splits1_;
  // The row_splits2[row_splits1] of the result; it is the exclusive-sum of the
  // size2 elements of `sizes` passed to the constructor.
  Array1<int32_t> row_splits12_;

  Array1<int32_t> row_splits2_;

  Array1<Arc> arcs_;

  // We set this to true
  bool finalized_row_splits2_;
  int32_t next_fsa_idx_;
};

/*
  Wrap some property test function in `host/properties.h`, they are generally
  for test purpose for now and work only for CPU. Users usually would not call
  these functions. Instead, Call `GetFsaBasicProperties` or
  `GetFsaVecBasicProperties` in fsa.h if you want to check Fsa/FsaVec's
  properites in production code.

  Noted all below functions work for both Fsa and FsaVec,
  If `fsas` is FsaVec, the function will return an array on CPU which has
    ans[i] = true if `fsas.Index(0,i)` has the corresponding property,
                  for 0 <= i < fsa_vec.Dim0()
  else
    the function will return an array with size 1 on CPU which has
    ans[0] = true if the fsa has the corresponding property.
*/

/*

  ans[i]=true if `fsas[i]` is valid. An Fsa is valid if:
  1. It is a valid Ragged array.
  2. If it's not empty, it contains at least two states.
     Noted empty fsa is valid.
  3. The final state is numerically greater than any other state.
  4. Only arcs with symbol==-1 enter the final state.
  5. There are no arcs leaving final_state.
  6. All arcs leaving a state have a same src_state, which is the corresponding
     idx0 in Fsa (or idx1 in FsaVec), see index naming convention explained
     in utils.h.
  TODO(haowen): Implement it as the version in host/properties.h doesn't
                address all requirements above, but this is not so urgent
                as we may not really need it for now.
 */
Array1<bool> IsValid(FsaOrVec &fsas);

/*
  ans[i]=true if `fsa[i]` is empty.
 */
Array1<bool> IsEmpty(FsaOrVec &fsas);

/*
  ans[i]=true if the states in `fsas[i]` are topologically sorted.
*/
Array1<bool> IsTopSorted(FsaOrVec &fsas);

/*
  ans[i]=true if arcs leaving each state in `fsas[i]` are sorted on
  label first and then on dest_state.
*/
Array1<bool> IsArcSorted(FsaOrVec &fsas);

/*
  ans[i]=true if `fsa[i]` has any self-loops
*/
Array1<bool> HasSelfLoops(FsaOrVec &fsas);

/*
  ans[i]=true if `fsas[i]` is acyclic.
*/
Array1<bool> IsAcyclic(FsaOrVec &fsas);

/*
  ans[i]=true if `fsas[i]` is deterministic; an Fsa is deterministic if
  it has no state that has multiple arcs leaving it with the same label on
  them.
*/
Array1<bool> IsDeterministic(FsaOrVec &fsas);

/*
  ans[i]=true if `fsas[i]` is free of epsilons, i.e. if there are no
  arcs for which `label` is kEpsilon == 0.
*/
Array1<bool> IsEpsilonFree(FsaOrVec &fsas);

/*
  ans[i]=true if all states in `fsas[i]` are both reachable from the
  start state (accessible) and can reach the final state (coaccessible).  Note:
  an FSA can be both connected and empty, because the empty FSA has no states
  (neither start state nor final state exist).  So you may sometimes want to
  check IsConnected() && IsNonempty().
 */
Array1<bool> IsConnected(FsaOrVec &fsas);

/*
  Wraps function IsRandEquivalent in `host/fsa_equivalent.h` for test purpose.
  Works for CPU only.
*/

/*
  (for Fsas):

    Returns true if the Fsa `a` is equivalent to `b` (ignoring the weights!),
    tested by randomly generating `npath` paths from one of them and then
    checking if the paths exist in the other one. Noted we only check the paths
    existence, the weights on the paths will not be checked.

   (for FsaVec):
     Returns true if IsRandEquivalentByCheckPathSymbols is true for all of
     the corresponding elements of a and b.

   The algorithm is done on CPU; the FSAs will be copied to CPU if needed.

   Noted we'll pass `treat_epsilons_specially` to Intersection in
   `host/intersect.h` to do the intersection between the random path and the
   input Fsas. Generally, if it's true, we will treat epsilons as epsilon when
   doing intersection; Otherwise, epsilons will just be treated as any other
   symbol. See `host/intersect.h` for details.
 */
bool IsRandEquivalentUnweighted(FsaOrVec &a, FsaOrVec &b,
                                bool treat_epsilons_specially = true,
                                std::size_t npath = 100);

/*
  Returns true if the Fsa `a` appears to be equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if the symbol
  sequence exists in the other one and if the total weight for that symbol
  sequence is the same in both FSAs.

  @param [in]  a          One of the FSAs to be checked the equivalence.
                          Must be top-sorted and have NumAxes() == 2 and on CPU.
  @param [in]  b          The other FSA to be checked the equivalence.
                          Must be top-sorted and have NumAxes() == 2 and on CPU.
  @param [in]  log_semiring The semiring to be used for all weight measurements;
                          if false then we use 'max' on alternative paths; if
                          true we use 'log-add'.
  @param [in]  beam       beam > 0 that affects pruning; the algorithm
                          will only check paths within `beam` of the
                          total score of the lattice (for tropical semiring,
                          it's max weight over all paths from start state to
                          final state; for log semiring, it's log-sum probs over
                          all paths) in `a` or `b`. That is, any symbol
                          sequence, whose total weights over all paths are
                          within `beam` of the total score of the lattice
                          (either in `a` or `b`), must have the same weights in
                          `a` and `b` (within `delta`).  There is no requirement
                          on symbol sequences whose total weights over paths are
                          outside `beam`.  Leave this as infinity you don't want
                          pruning.
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
  @param [in]  npath      The number of paths will be generated to check the
                          equivalence of `a` and `b`
 */
bool IsRandEquivalent(Fsa &a, Fsa &b, bool log_semiring,
                      float beam = k2host::kFloatInfinity,
                      bool treat_epsilons_specially = true, float delta = 1e-6,
                      std::size_t npath = 100);

/*
  Wrap forward and backward scores computation in `host/weights.h` (noted
  `score` is called `weight` there), they are generally for test purpose for now
  and work only for CPU. Users usually would not call these functions. Instead,
  Call `GetForwadScores` or `GetBackwardScores` in fsa_utils.h if you want to
  get FsaVec's forward or backward scores in production code.
*/

/*
   Compute and return forward scores per state (like alphas in Baum-Welch),
   or forward best-path scores if log_semiring == false.

      @param [in] fsas  Input Fsa or FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties. Must be on CPU.
      @param [in] log_semiring   If true, combine path with LogAdd
                  (i.e., mathematically, `log(exp(a)+exp(b))`); if false,
                   combine as `max(a,b)`.
      @return   Returns vector indexed by state-index (idx01 into fsas), i.e.
               `ans.Dim()==fsas.TotSize(1)`, containing forward scores.
                (these will be zero for the start-states).
*/
template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, bool log_semiring);

/*
   Compute and return backward scores per state (like betas in Baum-Welch),
   or backward best-path scores if log_semiring == false.
      @param [in] fsas  Input Fsa or FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties. Must be on CPU.
       @param [in] tot_scores  Must be on CPU. If provided, we'll treat
                  the backward scores of final-states as the negative
                  of these tot_scores (which must have
                  `tot_scores->Dim() == fsas.Dim0())`; otherwise
                  as zero.
       @param [in] log_semiring  If true, use LogAdd to combine
                  scores; if false, use max.
       @return  Returns a vector indexed by state-index (idx01 in fsas), with
               `ans.Dim() == fsas.TotSize(1)`, containing backward
               scores.
 */
template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, const Array1<FloatType> *tot_scores = nullptr,
    bool log_semiring = true);

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
