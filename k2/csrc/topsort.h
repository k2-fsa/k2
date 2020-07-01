// k2/csrc/topsort.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_TOPSORT_H_
#define K2_CSRC_TOPSORT_H_

#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"

namespace k2 {

/**
    Sort the input fsa topologically.
 */
class TopSorter {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA
  */
  explicit TopSorter(const Fsa &fsa_in) : fsa_in_(fsa_in) {}

  /*
    Do enough work that know now much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the top-sorted FSA to `fsa_out` and
    state mapping information to `state_map` (if provided).
    @param [out]  fsa_out Output fsa. It is set to empty if the input fsa is not
                          acyclic or is not connected; otherwise it contains the
                          top-sorted fsa.
                          Must be initialized; search for 'initialized
                          definition' in class Array2 in array.h for meaning.
    @param [out]  state_map   If non-NULL, Maps from state indexes in the output
                              fsa to state indexes in input fsa.
                              If non-NULL, at entry it must be allocated with
                              size num-states of `fsa_out`,
                              e.g. `fsa_out->size1`.

    @return true if the input fsa is acyclic and connected,
            or if the input is empty; return false otherwise.
   */
  bool GetOutput(Fsa *fsa_out, int32_t *state_map = nullptr);

 private:
  const Fsa &fsa_in_;

  bool is_connected_;  // if the input FSA is connected
  bool is_acyclic_;    // if the input FSA is acyclic
  // map order to state.
  // state 0 has the largest order, i.e., num_states - 1
  // final_state has the least order, i.e., 0
  std::vector<int32_t> order_;
};

}  // namespace k2

#endif  // K2_CSRC_TOPSORT_H_
