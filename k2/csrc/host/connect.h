/**
 * @brief
 * connect
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_CONNECT_H_
#define K2_CSRC_HOST_CONNECT_H_

#include <vector>

#include "k2/csrc/host/fsa.h"

namespace k2host {
/*
  Removes states that are not accessible (from the start state) or are not
  co-accessible (i.e. that cannot reach the final state), and ensures that
  if the FSA admits a topological sorting (i.e. it contains no cycles except
  self-loops), the version that is output is topologically sorted.  This
  is not a stable sort, i.e. states may be renumbered even for top-sorted
  input.

  Notes:
    - If the input FSA (`fsa_in`) admitted a topological sorting, the output
      FSA (`fsa_out`) will be topologically sorted. If `fsa_in` is not
      topologically sorted but is acyclic, `fsa_out` will also be topologically
      sorted. TODO(Dan): maybe just leave in the same order as `fsa_in`??
      (Current implementation may **renumber** the state)
    - If `fsa_in` was deterministic, `fsa_out` will be deterministic; same for
      epsilon free, obviously.
    - `fsa_out` will be arc-sorted (arcs sorted by label). TODO(fangjun): this
       has not be implemented.
 */
class Connection {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA
  */
  explicit Connection(const Fsa &fsa_in) : fsa_in_(fsa_in) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the connected FSA to `fsa_out` and
    arc mapping information to `arc_map` (if provided).
    @param [out] fsa_out   The output FSA; Will be connected;
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
    @param [out] arc_map   If non-NULL, will output a map from the arc-index
                           in `fsa_out` to the corresponding arc-index in
                           `fsa_in`.
                           If non-NULL, at entry it must be allocated with
                           size num-arcs of `fsa_out`, e.g. `fsa_out->size2`.

    @return   The return status indicates whether topological sorting
        was successful; if true, the result is top-sorted.  The only situation
        it might return false is when the input had cycles that were not self
        loops; such FSAs do not admit a topological sorting.

        Caution: true return status does not imply that the returned FSA is
        nonempty.
   */
  bool GetOutput(Fsa *fsa_out, int32_t *arc_map = nullptr);

 private:
  const Fsa &fsa_in_;

  bool is_acyclic_;  // if the input FSA is acyclic
  // if true, there is no state in the input FSA that is both
  // accessible and co-accessible.
  bool no_accessible_state_;
  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out
  std::vector<int32_t> arc_map_;
};

/*
  The core part of `Connection`which removes states that are not accessible
  or are not coaccessible, i.e. not reachable from start state or cannot reach
  the final state.

  If the resulting Fsa is empty, `state_map` will be empty at exit and
  it returns true.

     @param [in]  fsa         The FSA to be connected.
     @param [out] state_map   Maps from state indexes in the output fsa to
                              state indexes in `fsa`. If the input fsa is
                              acyclic, the output fsa is topologically sorted.

      Returns true on success (i.e. the output will be topsorted).
      The only failure condition is when the input had cycles that were not self
  loops.

      Caution: true return status does not imply that the returned FSA is
  nonempty.

 */
bool ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map);

}  // namespace k2host

#endif  // K2_CSRC_HOST_CONNECT_H_
