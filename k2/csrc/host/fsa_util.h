/**
 * @brief
 * fsa_util
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_FSA_UTIL_H_
#define K2_CSRC_HOST_FSA_UTIL_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/fsa.h"

namespace k2host {

namespace dfs {

constexpr int8_t kNotVisited = 0;  // a node that has not been visited
constexpr int8_t kVisiting = 1;    // a node that is under visiting
constexpr int8_t kVisited = 2;     // a node that has been visited
// depth first search state
struct DfsState {
  int32_t state;      // state number of the visiting node
  int32_t arc_begin;  // arc index of the visiting arc
  int32_t arc_end;    // end of the arc index of the visiting node
};
}  // namespace dfs

/*
  Computes lists of arcs entering each state (needed for algorithms that
  traverse the Fsa in reverse order).

  Requires that `fsa` be valid and top-sorted, i.e.  CheckProperties(fsa,
  KTopSorted) == true.

    @param [in]  fsa         The input FSA.
    @param [out] arc_indexes For each state i in `fsa`,
                             `arc_indexes.data[arc_indexes.indexes[i]] through
                             `arc_indexes.data[arc_indexes.indexes[i+1] - 1]`
                             will be the arc-indexes of those arcs entering
                             state `i` in `fsa`.  Must be initialized;
                             search for 'initialized definition' in class
                             Array2 in array.h for meaning. Specifically,
                             at entry there should be
                             `arc_indexes.size1 == fsa.size1` and
                             `arc_indexes.size2 == fsa.size2`.
*/
void GetEnteringArcs(const Fsa &fsa, Array2<int32_t *, int32_t> *arc_indexes);

/*
  TODO(dan): remove this, should no longer be needed.

  Gets arc weights for an FSA (output FSA) according to `arc_map` which
  maps each arc in the FSA to a sequence of arcs in the other FSA (input FSA).

    @param [in] arc_weights_in  Arc weights of the input FSA. Indexed by
                                arc in the input FSA.
    @param [in] arc_map  An `Array2` that can be interpreted as the arc
                         mappings from arc-indexes in the output FSA to
                         arc-indexes in the input FSA. Generally,
                         `arc_map.data[arc_map.indexes[i]]` through
                         `arc_map.data[arc_map.indexes[i+1] - 1]` is the
                         sequence of arc-indexes in the input FSA that
                         arc `i` in the output FSA corresponds to.
                         The weight of arc `i` will be equal to the sum of
                         those input arcs' weights.
    @param [out] arc_weights_out Arc weights of the output FSA. Indexed by arc
                                 in the output FSA. At entry it must be
                                 allocated with size `arc_map.size1`.
*/
void GetArcWeights(const float *arc_weights_in,
                   const Array2<int32_t *, int32_t> &arc_map,
                   float *arc_weights_out);

// TODO(dan): remove this, should no longer be needed.
//
// Version of GetArcWeights where arc_map maps each arc in the output FSA to
// one arc (instead of a sequence of arcs) in the input FSA; see its
// documentation.
// Note that `num_arcs` is the number of arcs in the output FSA,
// at entry `arc_map` should have size `num_arcs` and `arc_weights_out` must
// be allocated with size `num_arcs`.
void GetArcWeights(const float *arc_weights_in, const int32_t *arc_map,
                   int32_t num_arcs, float *arc_weights_out);

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
   @param [in] arc_map  Indexed by arc-index in the output FSA, the
                        arc-index in the input FSA that it corresponds to.
   @param [in] num_arcs The size of `arc_map`
   @param [out] indexes_out At entry it must be allocated with size `num_arcs`;
                            will contain arc-indexes in the input FSA.
 */
void ConvertIndexes1(const int32_t *arc_map, int32_t num_arcs,
                     int64_t *indexes_out);

/*
  Convert indexes (typically arc-mapping indexes, e.g. as output by
  RmEpsilonPruned()) from int32 to long int; this will be needed for conversion
  to LongTensor.

  This version is for when each arc of the output FSA may correspond to a
  sequence of arcs in the input FSA. For example,
  Suppose arc_map ==  [ [ 1, 2 ], [ 6, 8, 9 ] ], we'd form
  indexes1 = [ 1, 2, 6, 8, 9 ], and indexes2 = [ 0, 0, 1, 1, 1 ]

       @param [in] arc_map  An `Array2` that can be interpreted as the arc
                            mappings from arc-indexes in the output FSA to
                            arc-indexes in the input FSA. Generally,
                            `arc_map.data[arc_map.indexes[i]]` through
                            `arc_map.data[arc_map.indexes[i+1] - 1]` is the
                            sequence of arc-indexes in the input FSA that
                            arc `i` in the output FSA corresponds to.
       @param [out] indexes1  At entry it must be allocated with size
                              `arc_map.size2`; will contain arc-indexes
                              in the input FSA.
       @param [out] indexes2  At entry it must be allocated with size
                              `arc_map.size2`; will contain arc-indexes
                              in the output FSA.
 */
void GetArcIndexes2(const Array2<int32_t *, int32_t> &arc_map,
                    int64_t *indexes1, int64_t *indexes2);

// Create Fsa for test purpose.
class FsaCreator {
 public:
  // Create an empty Fsa
  FsaCreator() = default;

  /*
    Initialize Fsa with Array2size, search for 'initialized definition' in class
    Array2 in array.h for meaning. Note that we don't fill data in `indexes` and
    `data` here, the caller is responsible for this.

    `Array2Storage` is for this purpose as well, but we define this version of
    constructor here to make test code simpler.
  */
  explicit FsaCreator(const Array2Size<int32_t> &size) { Init(size); }

  void Init(const Array2Size<int32_t> &size) {
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
      K2_CHECK_LE(arc.src_state, final_state);
      K2_CHECK_LE(arc.dest_state, final_state);
      K2_CHECK_LE(curr_state, arc.src_state);
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

/* Create an acyclic FSA from a list of arcs. The returned Fsa is top-sorted and
   acyclic.

   Arcs do not need to be pre-sorted by src_state.
   If there is a cycle, it aborts.

   The start state MUST be 0. The final state will be automatically determined
   by topological sort.

   @param [in] arcs  A list of arcs.
   @param [out] fsa  Output fsa which is top-sorted. Must be initialized;
                     search for 'initialized definition' in class Array2
                     in array.h for meaning.
   @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `fsa` to
                            the corresponding arc-index in input `arcs`.
*/
void CreateTopSortedFsa(const std::vector<Arc> &arcs, Fsa *fsa,
                        std::vector<int32_t> *arc_map = nullptr);

/* Create an FSA from a list of arcs.

   Arcs do not need to be pre-sorted by src_state.

   The start state MUST be 0. There must be only one state whose all entering
   arcs have label -1 and there's no arc leaving this state, this state will
   be the final state; otherwise, if we cannot find such a state,
   the program will abort with an error.

   @param [in] arcs  A list of arcs.
   @param [out] fsa  Output fsa. Must be initialized; search for 'initialized
                     definition' in class Array2 in array.h for meaning.
   @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `fsa` to
                            the corresponding arc-index in input `arcs`.
*/
void CreateFsa(const std::vector<Arc> &arcs, Fsa *fsa,
               std::vector<int32_t> *arc_map = nullptr);

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
 */
class StringToFsa {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] s Input string representing the transition table.
  */
  explicit StringToFsa(const std::string &s) : s_(s) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the FSA to `fsa_out`
    @param [out] fsa_out   The output FSA;
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
   */
  void GetOutput(Fsa *fsa_out);

 private:
  const std::string &s_;

  // `arcs_[i]` will be the arcs leaving state `i`
  std::vector<std::vector<Arc>> arcs_;
};

std::string FsaToString(const Fsa &fsa);

struct RandFsaOptions {
  std::size_t num_syms;
  std::size_t num_states;
  std::size_t num_arcs;
  bool allow_empty;
  bool acyclic;  // generate a cyclic fsa in a best effort manner if it's false
  int32_t seed;  // for random generator. Set it to non-zero for reproducibility
  bool nonzero_weights;  // allow weights to be nonzero (default: false)

  RandFsaOptions();
};

/**
    Generate a random FSA.
 */
class RandFsaGenerator {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] opts   Options that control the properties of the generated
                        FSA.
  */
  explicit RandFsaGenerator(const RandFsaOptions &opts) : opts_(opts) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the generated FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the generated FSA to `fsa_out`
    @param [out]  fsa_out Output fsa.
                          Must be initialized; search for 'initialized
                          definition' in class Array2 in array.h for meaning.
   */
  void GetOutput(Fsa *fsa_out);

 private:
  const RandFsaOptions opts_;

  FsaCreator fsa_creator_;
};

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

}  // namespace k2host

#endif  // K2_CSRC_HOST_FSA_UTIL_H_
