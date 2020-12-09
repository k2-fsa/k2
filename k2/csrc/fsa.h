/**
 * @brief
 * fsa
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Guoguo Chen
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_FSA_H_
#define K2_CSRC_FSA_H_

#include <ostream>
#include <string>

#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

struct Arc {
  int32_t src_state;
  int32_t dest_state;
  int32_t label;
  float score;

  Arc() = default;
  Arc(int32_t src_state, int32_t dest_state, int32_t label, float score)
      : src_state(src_state),
        dest_state(dest_state),
        label(label),
        score(score) {}

  __host__ __device__ __forceinline__ bool operator==(const Arc &other) const {
    return src_state == other.src_state && dest_state == other.dest_state &&
           label == other.label && fabsf(score - other.score) < 1e-5;
  }

  __host__ __device__ __forceinline__ bool operator!=(const Arc &other) const {
    return !(*this == other);
  }

  __host__ __device__ __forceinline__ bool operator<(const Arc &other) const {
    // Compares `label` first, then `dest_state`;
    // compare label as unsigned so -1 comes after other symbols, since some
    // algorithms may require epsilons to be first.
    if (label != other.label)
      return static_cast<uint32_t>(label) < static_cast<uint32_t>(other.label);
    else
      return dest_state < other.dest_state;
  }
};

std::ostream &operator<<(std::ostream &os, const Arc &arc);
std::istream &operator>>(std::istream &os, Arc &arc);

using FsaProperties = uint32_t;

/*
  The FSA properties need to be computed via a reduction over arcs, and we use
  '&' to reduce.  These properties are all things that can be computed locally,
  using an arc and adjacent arcs (and the structural info).
 */
enum FsaBasicProperties {
  kFsaPropertiesValid = 0x01,      // Valid from a formatting perspective
  kFsaPropertiesNonempty = 0x02,   // Nonempty as in, has at least one arc.
  kFsaPropertiesTopSorted = 0x04,  // FSA is top-sorted, but possibly with
                                   // self-loops, dest_state >= src_state
  kFsaPropertiesTopSortedAndAcyclic =
      0x08,  // Top-sorted and acyclic (no self-loops), dest_state > src_state
  kFsaPropertiesArcSorted =
      0x10,  // Arcs leaving a given state are sorted by label first and then on
             // `dest_state`, see operator< in struct Arc above.
             // (Note: labels are treated as uint32 for purpose of sorting!)
  kFsaPropertiesArcSortedAndDeterministic = 0x20,  // Arcs leaving a given state
                                                   // are *strictly* sorted by
                                                   // label, i.e. no duplicates
                                                   // with the same label.

  kFsaPropertiesEpsilonFree = 0x40,  // Label zero (epsilon) is not present..
  kFsaPropertiesMaybeAccessible = 0x80,  // True if there are no obvious signs
                                         // of states not being accessible or
                                         // co-accessible, i.e. states with no
                                         // arcs entering them
  kFsaPropertiesMaybeCoaccessible =
      0x0100,  // True if there are no obvious signs of
               // states not being co-accessible, i.e.
               // i.e. states with no arcs leaving them
  kFsaAllProperties = 0x01FF
};

/* Convert FSA properties to a string.

   @param [in] properties
   @return A string consisting of the names of FsaBasicProperties'
           member separated by |. For example, if properties == 3,
           it will return kFsaPropertiesValid|kFsaPropertiesNonempty.
           If properties == 0, it returns an empty string.
 */
std::string FsaPropertiesAsString(int32_t properties);

using Fsa = Ragged<Arc>;  // 2 axes: state,arc

using FsaVec = Ragged<Arc>;  // 3 axes: fsa,state,arc.  Note, the src_state
                             // and dest_state in the arc are *within the
                             // FSA*, i.e. they are idx1 not idx01.

using FsaOrVec = Ragged<Arc>;  // for when we don't know if it will have 2 or
                               // 3 axes.  (i.e. Fsa or FsaVec)

/*
  Vector of FSAs that actually will come from neural net log-softmax outputs (or
  similar).

  Conceptually this is a 3-dimensional tensor of log-probs with the second
  dimension ragged, i.e.  the shape would be [ num_fsas, None, num_symbols+1 ],
  e.g. if this were a TF ragged tensor.  The indexing would be
  [fsa_idx,t,symbol+1], where the "+1" after the symbol is so that we have
  somewhere to put the output for symbol == -1 (remember, -1 is kFinalSymbol,
  used on the last frame).

  Also, if a particular FSA has T frames of neural net output, we actually
  have T+1 potential indexes, 0 through T, so there is space for the terminating
  final-symbol on frame T.  (On the last frame, the final symbol has
  logprob=0, the others have logprob=-inf).
 */
struct DenseFsaVec {
  RaggedShape shape;  // has 2 axes; indexed first by FSA-index (this object
                      // represents a list of FSAs!); and then for each FSA,
                      // the state-index (actually the state-index from which
                      // the arcs leave).

  // TODO: construct from a regular matrix containing the nnet output, plus some
  // meta-info saying where the supervisions are.

  // The following variable was removed and can be obtained as scores.Dim1().
  // int32_t num_cols;

  // `scores` is a contiguous matrix of dimension shape.TotSize1()
  // by num_cols (where num_cols == num_symbols+1); the indexes into it are
  // [row_idx, symbol+1], where row_ids is an ind_01 w.r.t. `shape` (see naming
  // convention explained in utils.h).
  //
  //  You can access scores[row_idx,symbol+1] as
  //  scores.Data()[row_ids*scores.Dim1() + symbol+1]
  //
  // `scores` contains -infinity in certain locations: in scores[j,0] where
  // j is not the last row-index for a given FSA-index, and scores[j,k] where
  // j is the last row-index for a given FSA-index and k>0.  The remaining
  // locations contain the neural net output, except scores[j,0] where j
  // is the last row-index for a given FSA-index; this contains zero.
  // (It's the final-transition).
  Array2<float> scores;

  // NOTE: our notion of "arc-index" / arc_idx is an index into scores.Data().
  int32_t NumArcs() const { return scores.Dim0() * scores.Dim1(); }

  DenseFsaVec() {}
  DenseFsaVec(const RaggedShape &shape, const Array2<float> &scores)
      : shape(shape), scores(scores) {
    K2_CHECK(IsCompatible(shape, scores));
    K2_CHECK_EQ(shape.NumElements(), scores.Dim0());
    K2_CHECK_EQ(shape.NumAxes(), 2);
  }
  ContextPtr &Context() const { return shape.Context(); }
  DenseFsaVec To(ContextPtr c) const {
    return DenseFsaVec(shape.To(c), scores.To(c));
  }
  /* Indexing operator that rearranges the sequences, analogous to: RaggedShape
     Index(RaggedShape &src, const Array1<int32_t> &indexes).  Currently just
     used for testing.
   */
  DenseFsaVec operator[](const Array1<int32_t> &indexes);
};

std::ostream &operator<<(std::ostream &os, const DenseFsaVec &dfsavec);

/*
  Create an FSA from a Tensor.  The Tensor is expected to be an N by 4 tensor of
  int32_t, where N is the number of arcs (the format is src_state, dest_state,
  symbol, cost).  The cost is not really an int32_t, it is a float.  This code
  will print an error message and output 'true' to 'error', and return an empty
  FSA (with no states or arcs) if 't' was not interpretable as a valid FSA.
  These requirements for a valid FSA are:

    - src_state values on the arcs must be non-decreasing
    - all arcs with -1 as the label must be to a single state (call this
      final_state) which has no arcs leaving it
    - final_state, if it exists, must be numerically greater than any state
      which has arcs leaving it, and >= any state that has arcs entering it.

  If there are no arcs with -1 on the label, here is how we determine the final
  state:
     - If there were no arcs at all in the FSA we'll return the empty FSA (with
  no states).
     - Otherwise, we'll let `final_state` be the highest-numbered state
       that has any arcs leaving or entering it, plus one.  (This FSA
       has no successful paths but still has states.)

    @param [in] t   Source tensor.  Caution: the returned FSA will share
                    memory with this tensor, so don't modify it afterward!
    @param [out] error   Error flag.  On success this function will write
                        'false' here; on error, it will print an error
                        message to the standard error and write 'true' here.
    @return         The resulting FSA will be returned.

*/
Fsa FsaFromTensor(Tensor &t, bool *error);


Fsa FsaFromArray1(Array1<Arc> &arc, bool *error);

/*
  Returns a single Tensor that represents the FSA; this is just the vector of
  Arc reinterpreted as a (num_arcs by 4) Tensor of int32_t.  It can be converted
  back to an equivalent FSA using `FsaFromTensor`.  Notice: this is not the same
  format as we use to serialize FsaVec.  Also the round-trip conversion to
  Tensor and back may not preserve the number of states for FSAs that had no
  arcs entering the final-state, since we have to guess the number of states in
  this case.

  It is an error if `fsa.NumAxes() != 2`.
 */
Tensor FsaToTensor(const Fsa &fsa);

/*
  Returns a single Tensor that represents a vector of FSAs.  It is a vector of
  int32_t's (on the same device as `fsa_vec`).  The format is as follows:

       - 1st element is num_fsas
       - 2nd element is currently zero (included for word-alignment purposes)
       - Next `num_fsas + 1` elements are the row_splits1 of the FsaVec,
         i.e. 0, num_states1, num_states1+num_states2, ...  [the exclusive-sum
         of the num-states of all the FSAs]
       - Next `num_fsas + 1` elements are combined row_splits1 and row_splits2
         of the FsaVec, which are the exclusive sum of the total number of arcs
         in the respective FSAs.
       - Next `num_arcs * 4` elements are the arcs in the FSAs (note: the
         float-valued will be reinterpreted as int32_t's but are still floats).

  If it is really a transducer you can just store the olabels as a separate
  tensor; the num-arcs and the arc layout will survive the round-trip to
  serialization so this will work.

  You can convert this back to an FSA using `FsaVecFromTensor`.

  It is an error if `fsa_vec` does not have 3 axes.  Empty FsaVec's are allowed,
  though (i.e. num_fsas == 0 is allowed).
*/
Tensor FsaVecToTensor(const FsaVec &fsa_vec);

/*
  Create an FsaVec (vector of FSAs) from a Tensor which is an array of
  int32_t's. This tensor is interpreted as follows: First 2 elements: num_fsas 0
     Next num_fsas + 1 elements:  row_splits1 of the FsaVec, which is
                                the cumulative sum of num_states
     Next num_fsas + 1 elements:  row_splits12 of the FsaVec, i.e. its
                                row_splits2[row_splits1], which is the
                                cumulative sum of num_arcs for those FSAs
     Next num_arcs * 4 elements:  the arcs.  The scores in the arcs are really
                                of type float, not int32_t.


    @param [in] t   Source tensor.  Must have dtype == kInt32Dtype and have one
                    axis.  Caution: the returned FSA will share
                    memory with this tensor if the FSA was originally created by
                    FsaVecFromTensor().
    @param [out] error   Error flag.  On success this function will write
                      'false' here; on error, it will print an error
                       message to the standard error and write 'true' here.
    @return         The resulting FsaVec (vector of FSAs) will be returned;
                    this is a Ragged<Arc> with 3 axes.  Caution, it will not
                    have been fully validated; you might want to check the
                    kFsaPropertiesValid property once you compute the
                    properties.

*/
FsaVec FsaVecFromTensor(Tensor &t, bool *error);

/*
  Return one Fsa in an FsaVec.  Note, this has to make copies of the
  row offsets and strides but can use a sub-array of the arcs array
  directly.

     @param [in] vec   Input FsaVec to make a copy of
     @param [in] i     Index within the FsaVec to select
     @return           Returns the FSA.  Its `values` array will
                       refer to a part of the `values` array of
                       the input `vec`.
 */
inline Fsa GetFsaVecElement(FsaVec &vec, int32_t i) { return vec.Index(0, i); }

/*
  Create an FsaVec from a list of Fsas.  Caution: Fsa and FsaVec are really
  the same type, just with different expectations on the number of axes!
 */
inline FsaVec CreateFsaVec(int32_t num_fsas, Fsa **fsas) {
  // Implementation goes to this template:
  //  template <typename T>
  //  Ragged<T> Stack(int32_t axis, int32_t src_size, const Ragged<T> *src);
  K2_CHECK_EQ(fsas[0]->NumAxes(), 2);
  return Stack(0, num_fsas, fsas);
}

// Returns FSA with no arcs and no states, which is just an empty Ragged<Arc>
// with 2 axes.
Fsa EmptyFsa();

/*
   If the input was an Fsa (2 axes) then converts it to an FsaVec with
   one element (note: will share the same underlying
   memory, just add an extra axis, increasing NumAxes() from 2 to 3).
   Otherwise just return `fsa` unchanged, so it will pass through an FsaVec
   unchanged.
   `fsa` non-const because the FSA's row-ids are populated on-demand.
*/
FsaVec FsaToFsaVec(const Fsa &fsa);

// Compute and return basic properties for Fsa.
// Returns 0 if fsa.NumAxes() != 2.
int32_t GetFsaBasicProperties(const Fsa &fsa);

/*
  Compute basic properties for an FsaVec, with their `and` in `properties_tot`.

     @param [in] fsa_vec   FSAs to compute the properties of.  It is an
                           error if fsa_vec.NumAxes() != 3 (will crash).
     @param [out] properties_out  The properties per FSA will be written to
                   here, on the same device as `fsa_vec`.  This array
                   will be assigned to (does not have to be correctly sized at
                   entry).
     @param [out] tot_properties_out  The `and` of all properties in
                  `properties_out` will be written to this host
                  (i.e. CPU-memory) pointer.
*/
void GetFsaVecBasicProperties(FsaVec &fsa_vec, Array1<int32_t> *properties_out,
                              int32_t *tot_properties_out);

// Return weights of `arcs` as a Tensor that shares the same memory
// location
Tensor WeightsOfArcsAsTensor(const Array1<Arc> &arcs);

// Return weights of `arcs` as an Array1<float>; this will not share the same
// memory location because Array1 does not support a stride.  However
// it would be possible to get it as an Array2.
inline Array1<float> WeightsOfArcsAsArray1(const Array1<Arc> &arcs) {
  return Array1<float>(WeightsOfArcsAsTensor(arcs));
}

inline Array1<float> WeightsOfFsaAsArray1(const Ragged<Arc> &fsa) {
  return Array1<float>(WeightsOfArcsAsTensor(fsa.values));
}

}  // namespace k2

#endif  // K2_CSRC_FSA_H_
