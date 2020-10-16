/**
 * @brief host_shim  Wrapper functions so we can use our older
 *                 CPU-only code, in host/, with the newer interfaces
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_SHIM_H_
#define K2_CSRC_HOST_SHIM_H_

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa.h"
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

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
