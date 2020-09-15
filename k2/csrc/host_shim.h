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

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa.h"

namespace k2 {

/*
  Convert k2 Fsa to k2host::Fsa (this is our older version of this codebase,
  that only works on CPU).

  This only works of `fsa` was on the CPU.  Note: the k2host::Fsa
  refers to memory inside `fsa`, so making a copy doesn't work.
  Be careful with the returned k2host::Fsa as it does not own its
  own memory!
*/
k2host::Fsa FsaToHostFsa(Fsa &fsa) {
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  K2_CHECK_EQ(fsa.Context().DeviceType(), kCpu);
  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).
  return k2host::Array2(fsa.Dim0(), fsa.TotSize1(),
                        fsa.RowSplits1().Data(),
                        reinterpret_cast<k2host::Arc*>(fsa.values.Data()));
}

class FsaCreator {
 public:
  FsaCreator() = default;
  /*
    Initialize Fsa with host::Array2size, search for 'initialized definition' in class
    Array2 in array.h for meaning. Note that we don't fill data in `indexes` and
    `data` here, the caller is responsible for this.

    `Array2Storage` is for this purpose as well, but we define this version of
    constructor here to make test code simpler.
  */
  explicit FsaCreator(const host::Array2Size<int32_t> &size) { Init(size); }

  void Init(const host::Array2Size<int32_t> &size) {
    arc_indexes_ = Array1<int32_t>(CpuContext(), size.size1 + 1);
    // just for case of empty Array2 object, may be written by the caller
    arc_indexes_.data[0] = 0;
    arcs_ = Array1<Arc>(CpuContext(), size.size2);
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
    arcs_ = Array1<Arc>(CpuContext(), arcs);
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

    arc_indexes_ = Array1<int32_t>(CpuContext(), arc_indexes);
  }

  Fsa GetFsa() const {
    return Fsa(Ragged2Shape(&arc_indexes_, nullptr, arcs_.Dim()),
               arcs_);
  }

  k2host::Fsa GetHostFsa() const {
    return k2host::Fsa(arc_indexes_.Dim() - 1, arcs_.Dim(), arc_indexes_.Data(),
                       reinterpret_cast<k2host::Arc*>(arcs_.Data()));

  }

 private:
  Array1<int32_t> arc_indexes_; // == row_splits
  Array1<Arc> arcs_;
};



}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
