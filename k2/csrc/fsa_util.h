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

// Version of GetArcWeights where arc_map maps each arc in the output FSA to
// one arc (instead of a sequence of arcs) in the input FSA; see its
// documentation.
void GetArcWeights(const float *arc_weights_in,
                   const std::vector<int32_t> &arc_map, float *arc_weights_out);

/* Reorder a list of arcs to get a valid FSA. This function will be used in a
   situation that the input list of arcs is not sorted by src_state, we'll
   reorder the arcs and generate the corresponding valid FSA. Note that we don't
   remap any state index here, it is supposed that the start state is 0 and the
   final state is the largest state number in the input arcs.

   @param [in] arcs  A list of arcs.
   @param [out] fsa  Output fsa. Must be initialized; search for
                     'initialized definition' in class Array2 in
                     array.h for meaning.
   @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `fsa` to
                            the corresponding arc-index in input `arcs`.
*/
void ReorderArcs(const std::vector<Arc> &arcs, Fsa *fsa,
                 std::vector<int32_t> *arc_map = nullptr);

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

// Create Fsa for test purpose.
class FsaCreator {
 public:
  // Create an empty Fsa
  FsaCreator() {
    // TODO(haowen): remove below line and use `FsaCreator() = default`
    // we need this for now as we reset `indexes = nullptr` in the constructor
    // of Fsa
    fsa_.indexes = &fsa_.size1;
  }

  /*
    Initialize Fsa with Array2size, search for 'initialized definition' in class
    Array2 in array.h for meaning. Note that we don't fill data in `indexes` and
    `data` here, the caller is responsible for this.

    `Array2Storage` is for this purpose as well, but we define this version of
    constructor here to make test code simpler.
  */
  explicit FsaCreator(const Array2Size<int32_t> &size) {
    arc_indexes_.resize(size.size1 + 1);
    // just for case of empty Array2 object, may be written by the caller
    arc_indexes_[0] = 0;
    arcs_.resize(size.size2);
    fsa_.Init(size.size1, size.size2, arc_indexes_.data(), arcs_.data());
  }

  /*
    Create an Fsa from a vector of arcs
     @param [in, out] arcs   A vector of arcs as the arcs of the generated Fsa.
                             The arcs in the vector should be sorted by
                             src_state.
     @param [in] final_state Will be as the final state id of the generated Fsa.
   */
  explicit FsaCreator(const std::vector<Arc> &arcs, int32_t final_state)
      : FsaCreator() {
    if (arcs.empty())
      return;  // has created an empty Fsa in the default constructor
    arcs_ = arcs;
    int32_t curr_state = -1;
    int32_t index = 0;
    for (const auto &arc : arcs_) {
      CHECK_LE(arc.src_state, final_state);
      CHECK_LE(arc.dest_state, final_state);
      CHECK_LE(curr_state, arc.src_state);
      while (curr_state < arc.src_state) {
        arc_indexes_.push_back(index);
        ++curr_state;
      }
      ++index;
    }
    // noted that here we push two `final_state` at the end, the last element is
    // just to avoid boundary check for some FSA operations.
    for (; curr_state <= final_state; ++curr_state)
      arc_indexes_.push_back(index);

    fsa_.Init(static_cast<int32_t>(arc_indexes_.size()) - 1,
              static_cast<int32_t>(arcs_.size()), arc_indexes_.data(),
              arcs_.data());
  }

  const Fsa &GetFsa() const { return fsa_; }
  Fsa &GetFsa() { return fsa_; }

 private:
  Fsa fsa_;
  std::vector<int32_t> arc_indexes_;
  std::vector<Arc> arcs_;
};

}  // namespace k2

#endif  // K2_CSRC_FSA_UTIL_H_
