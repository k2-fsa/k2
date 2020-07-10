// k2/csrc/intersect.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_INTERSECT_H_
#define K2_CSRC_INTERSECT_H_

#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"

namespace k2 {
/**
   Compute the intersection of two FSAs; this is the equivalent of composition
   for automata rather than transducers, and can be used as the core of
   composition.
 */
class Intersection {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] a    One of the FSAs to be intersected.  Must satisfy
                      CheckProperties(a, kArcSorted)
     @param [in] b    The other FSA to be intersected  Must satisfy
                      CheckProperties(b, kArcSorted), and either a or b
                      must be epsilon-free (c.f. IsEpsilonFree); this
                      ensures that epsilons do not have to be treated
                      differently from any other symbol.
  */
  Intersection(const Fsa &a, const Fsa &b) : a_(a), b_(b) {}

  /*
    Do enough work that know now much memory will be needed, and output
    that information
     @param [out] fsa_size   The num-states and num-arcs of the output FSA
                             will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the composed FSA to `c` and
    arc mapping information to `arc_map_a` and `arc_map_b` (if provided).

    @param [out] c         The composed FSA will be output to here.
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
    @param [out] arc_map_a If non-NULL, at exit will be a vector of
                           size c->size2, saying for each arc in
                           `c` what the source arc in `a` was, `-1` represents
                           there is no corresponding source arc in `a`.
                           If non-NULL, at entry it must be allocated with
                           size num-arcs of `c`, e.g. `c->size2`.
    @param [out] arc_map_b If non-NULL, at exit will be a vector of
                           size c->size2, saying for each arc in
                           `c` what the source arc in `b` was, `-1` represents
                           there is no corresponding source arc in `b`.
                           If non-NULL, at entry it must be allocated with
                           size num-arcs of `c`, e.g. `c->size2`.

    @return false if `a` or `b` is not arc-sorted, or if both `a` and `b` are
                  not epsilon-free; return true otherwise.
   */
  bool GetOutput(Fsa *c, int32_t *arc_map_a = nullptr,
                 int32_t *arc_map_b = nullptr);

 private:
  const Fsa &a_;
  const Fsa &b_;

  bool status_;
  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out

  std::vector<int32_t> arc_map_a_;
  std::vector<int32_t> arc_map_b_;
};

/**
   Intersection of two weighted FSA's: the same as Intersect(), but it prunes
   based on the sum of two costs.  Note: although these costs are provided per
   arc, they would usually be a sum of forward and backward costs, that is
   0 if this arc is on a best path and otherwise is the distance between
   the cost of this arc and the best-path cost.

  @param [in] a    One of the FSAs to be intersected.  Must satisfy
                   CheckProperties(a, kArcSorted)
  @param [in] a_cost  Pointer to array containing a cost per arc of a
  @param [in] b    The other FSA to be intersected  Must satisfy
                   CheckProperties(b, kArcSorted), and either a or b
                   must be epsilon-free (c.f. IsEpsilonFree).
  @param [in] b_cost  Pointer to array containing a cost per arc of b
  @param [in] cutoff  Cutoff, such that we keep an arc in the output
                   if its cost_a + cost_b is less than this cutoff,
                   where cost_a and cost_b are elements of
                   `a_cost` and `b_cost`.
  @param [out] c   The output FSA will be written to here.
  @param [out] state_map_a  Maps from arc-index in c to the corresponding
                   arc-index in a
  @param [out] state_map_b  Maps from arc-index in c to the corresponding
                   arc-index in b
 */
void IntersectPruned2(const Fsa &a, const float *a_cost, const Fsa &b,
                      const float *b_cost, float cutoff, Fsa *c,
                      std::vector<int32_t> *state_map_a,
                      std::vector<int32_t> *state_map_b);

}  // namespace k2

#endif  // K2_CSRC_INTERSECT_H_
