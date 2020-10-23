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
*/

/*
  Check properties of FsaOrVec (which must be on CPU) with property test
  function `f` which is one of property test functions for k2host::Fsa in
  host/properties.h, e.g. IsValid(const k2::host Fsa&), IsTopSorted(const
  k2host::Fsa&).

  If `fsas` is FsaVec, the function will return an array on CPU which has
    ans[i] = f(FsaVecToHostFsa(fsa_vec, i)) for 0 <= i < fsa_vec.Dim0();
  else
    returns a CPU array with size 1 and ans[0] = f(FsaToHostFsa(fsas))
*/
Array1<bool> CheckProperties(FsaOrVec &fsas, bool (*f)(const k2host::Fsa &));

/*
  ans[i]=true if `fsas[i]` is valid. An Fsa is valid if:
  1. It is a valid Ragged array.
  2. If it's not empty, it contains at least two states.
     Noted empty fsa is valid.
  3. Only arcs with symbol==-1 enter the final state.
  4. There's no arcs leaving final_state.
  5. All arcs leaving a state have a same src_state, which is the corresponding
     idx0 in fsa, see index naming convention explained in utils.h.
  TODO(haowen): Implement it as the version in host/properties.h doesn't
                address all requirements above, but this is not so urgent
                as we may not really need it for now.
 */
inline Array1<bool> IsValid(FsaOrVec &fsas);

/*
  ans[i]=true if `fsa[i]` is empty.
 */
inline Array1<bool> IsEmpty(FsaOrVec &fsas);

/*
  ans[i]=true if the states in `fsas[i]` are topologically sorted.
*/
inline Array1<bool> IsTopSorted(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::IsTopSorted);
}

/*
  ans[i]=true if arcs leaving each state in `fsas[i]` are sorted on
  label first and then on dest_state.
*/
inline Array1<bool> IsArcSorted(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::IsArcSorted);
}

/*
  ans[i]=true if `fsa[i]` has any self-loops
*/
inline Array1<bool> HasSelfLoops(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::HasSelfLoops);
}

// As k2host::IsAcyclic has two input arguments, we create a wrapper function
// here so we can pass it to CheckProperties
inline bool IsAcyclicWapper(const k2host::Fsa &fsa) {
  return k2host::IsAcyclic(fsa, nullptr);
}
/*
  ans[i]=true if `fsas[i]` is acyclic.
*/
inline Array1<bool> IsAcyclic(FsaOrVec &fsas) {
  return CheckProperties(fsas, IsAcyclicWapper);
}

/*
  ans[i]=true if `fsas[i]` is deterministic; an Fsa is deterministic if
  it has is no state that has multiple arcs leaving it with the same label on
  them.
*/
inline Array1<bool> IsDeterministic(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::IsDeterministic);
}

/*
  ans[i]=true if `fsas[i]` is free of epsilons, i.e. if there are no
  arcs for which `label` is kEpsilon == 0.
*/
inline Array1<bool> IsEpsilonFree(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::IsEpsilonFree);
}

/*
  ans[i]=true if all states in `fsas[i]` are both reachable from the
  start state (accessible) and can reach the final state (coaccessible).  Note:
  an FSA can be both connected and empty, because the empty FSA has no states
  (neither start state nor final state exist).  So you may sometimes want to
  check IsConnected() && IsNonempty().
 */
inline Array1<bool> IsConnected(FsaOrVec &fsas) {
  return CheckProperties(fsas, k2host::IsConnected);
}

/*
  Wraps function IsRandEquivalent in `host/fsa_equivalent.h` for test purpose.
  Works for CPU only.
*/

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if the
  paths exist in the other one. `a` and `b` must have NumAxes() == 2 and on CPU.
 */
bool IsRandEquivalent(Fsa &a, Fsa &b, std::size_t npath = 100);

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if each path
  exists in the other one and the sum of weights along that path are the same.

  @param [in]  a          One of the FSAs to be checked the equivalence.
                          Must have NumAxes() == 2 and on CPU.
  @param [in]  b          The other FSA to be checked the equivalence.
                          Must have NumAxes() == 2 and on CPU.
  @param [in]  top_sorted The user may set this to true if both `a` and `b` are
                          topologically sorted; this makes this function faster.
                          Otherwise it must be set to false.
  @param [in]  log_semiring If true, the algorithm will only check paths
                          within `beam` of the of the log-sum probs over
                          all pahts;
                          If false, the algorithm will only check paths
                          within `beam` of the best path (it's the max weight
                          over all paths from start state to final state;
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
                          Just keep `k2host::kFloatInfinity` if you don't want
                          pruning.
  @param [in]  delta      Tolerance for path weights to check the equivalence.
                          If abs(weights_a, weights_b) <= delta, we say the two
                          paths are equivalent.
  @param [in]  npath      The number of paths will be generated to check the
                          equivalence of `a` and `b`
 */
bool IsRandEquivalent(Fsa &a, Fsa &b, bool top_sorted, bool log_semiring,
                      float beam = k2host::kFloatInfinity, float delta = 1e-6,
                      std::size_t npath = 100);

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
