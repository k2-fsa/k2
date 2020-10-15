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


k2host::Fsa FsaToHostFsa(Fsa &fsa) {
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  K2_CHECK_EQ(fsa.Context()->GetDeviceType(), kCpu);
  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).
  return k2host::Fsa(fsa.shape.Dim0(), fsa.shape.TotSize(1),
                     fsa.shape.RowSplits(1).Data(),
                     reinterpret_cast<k2host::Arc *>(fsa.values.Data()));
}

k2host::Fsa FsaVecToHostFsa(FsaVec &fsa_vec, int32_t index) {
  K2_CHECK_EQ(fsa_vec.NumAxes(), 3);
  K2_CHECK_LT(static_cast<uint32_t>(index), static_cast<uint32_t>(fsa_vec.Dim0()));
  K2_CHECK_EQ(fsa_vec.Context()->GetDeviceType(), kCpu);

  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).

  int32_t *row_splits1_data = fsa.RowSplits(1).Data(),
      *row_splits2_data = fsa.RowSplits(1).Data();
  Arc *arcs_data = fsa.values.Data();
  int32_t start_state_idx01 = row_splits1_data[index],
      end_state_idx01 = row_splits1_data[index+1],
      size1 = end_state01 - start_state01,
      start_arc_idx012 = row_splits2_data[start_state_idx01],
      end_arc_idx012 = row_splits2_data[end_state_idx01],
      size2 = end_arc_idx012 - start_arc_idx012;

  return k2host::Fsa(size1, size2,
                     row_splits2_data + start_state_idx01,
                     arcs_data + start_arc_idx012);
}


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

  k2host::Fsa GetHostFsa() {
    return k2host::Fsa(arc_indexes_.Dim() - 1, arcs_.Dim(), arc_indexes_.Data(),
                       reinterpret_cast<k2host::Arc *>(arcs_.Data()));
  }

 private:
  Array1<int32_t> arc_indexes_;  // == row_splits
  Array1<Arc> arcs_;
};

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
