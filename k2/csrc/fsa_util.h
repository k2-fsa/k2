// k2/csrc/fsa_util.h

// Copyright (c)  2020  Daniel Povey
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_UTIL_H_
#define K2_CSRC_FSA_UTIL_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"

namespace k2 {

/*
  Computes lists of arcs entering each state (needed for algorithms that
  traverse the Fsa in reverse order).

  Requires that `fsa` be valid and top-sorted, i.e.  CheckProperties(fsa,
  KTopSorted) == true.

    @param [out] arc_index   A list of arc indexes.
                             For states 0 < s < fsa.NumStates(),
                             the elements arc_index[i] for
                             end_index[s-1] <= i < end_index[s] contain the
                             arc-indexes in fsa.arcs for arcs that enter
                             state s.
    @param [out] end_index   For each state, the `end` index in `arc_index`
                             where we can find arcs entering this state, i.e.
                             one past the index of the last element in
                             `arc_index` that points to an arc entering
                             this state.
*/
void GetEnteringArcs(const Fsa &fsa, std::vector<int32_t> *arc_index,
                     std::vector<int32_t> *end_index);

/*
  Gets arc weights for an FSA (output FSA) according to `arc_map` which
  maps each arc in the FSA to a sequence of arcs in the other FSA (input FSA).

    @param [in] arc_weights_in  Arc weights of the input FSA. Indexed by
                                arc in the input FSA.
    @param [in] arc_map         Indexed by arc in the output FSA. `arc_map[i]`
                                lists the sequence of arcs in the input FSA
                                that arc `i` in the output FSA corresponds to.
                                The weight of arc `i` will be equal to the
                                sum of those input arcs' weights.
    @param [out] arc_weights_out Arc weights of the output FSA. Indexed by arc
                                 in the output FSA. It should have the same size
                                 with arc_map at entry.
*/
void GetArcWeights(const float *arc_weights_in,
                   const std::vector<std::vector<int32_t>> &arc_map,
                   float *arc_weights_out);

// Version of GetArcWeights where arc_map maps each arc in the output FSA to one
// arc (instead of a sequence of arcs) in the input FSA; see its documentation.
void GetArcWeights(const float *arc_weights_in,
                   const std::vector<int32_t> &arc_map, float *arc_weights_out);
/*
  Convert indexes (typically arc-mapping indexes, e.g. as output by Compose())
  from int32 to int64; this will be needed for conversion to LongTensor.
 */
void ConvertIndexes1(const std::vector<int32_t> &arc_map, int64_t *indexes_out);

/*
  Convert indexes (typically arc-mapping indexes, e.g. as output by
  RmEpsilonPruned())
  from int32 to long int; this will be needed for conversion to LongTensor.

  This version is for when each arc of the output FSA may correspond to a
  sequence of arcs in the input FSA.

       @param [in] arc_map   Indexed by arc-index in the output FSA, the
                            sequence of arc-indexes in the input FSA that
                            it corresponds to
       @param [out] indexes1  This vector, of length equal to the
                           total number of int32's in arc_map, will contain
                           arc-indexes in the input FSA
       @param [out] indexes2  This vector, also of length equal to the
                           total number of int32's in arc_map, will contain
                           arc-indexes in the output FSA
 */
void GetArcIndexes2(const std::vector<std::vector<int32_t>> &arc_map,
                    std::vector<int64_t> *indexes1,
                    std::vector<int64_t> *indexes2);

void Swap(Fsa *a, Fsa *b);

/** Build an FSA from a string.

  The input string is a transition table with the following
  format (same with OpenFST):

  from_state  to_state  label
  from_state  to_state  label
  ... ...
  final_state

  K2 requires that the final state has the largest state number. The above
  format requires the last line to be the final state, whose sole purpose is
  to be compatible with OpenFST.

  @param [in] s Input string representing the transition table.

  @return an FSA.
 */
std::unique_ptr<Fsa> StringToFsa(const std::string &s);

std::string FsaToString(const Fsa &fsa);

struct RandFsaOptions {
  std::size_t num_syms;
  std::size_t num_states;
  std::size_t num_arcs;
  bool allow_empty;
  bool acyclic;  // generate a cyclic fsa in a best effort manner if it's false
  int32_t seed;  // for random generator. Set it to non-zero for reproducibility

  RandFsaOptions();
};

void GenerateRandFsa(const RandFsaOptions &opts, Fsa *fsa);

// move-copy an array to output, reordering it according to given indexes,
// where`index[i]` tells us what value (i.e. `src[index[i]`) we should copy to
// `dest[i]`
template <class InputIterator, class Size, class RandomAccessIterator,
          class OutputIterator>
void ReorderCopyN(InputIterator index, Size count, RandomAccessIterator src,
                  OutputIterator dest) {
  if (count > 0) {
    for (Size i = 0; i != count; ++i) {
      *dest++ = std::move(src[*index++]);
    }
  }
}

}  // namespace k2

#endif  // K2_CSRC_FSA_UTIL_H_
