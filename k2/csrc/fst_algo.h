#include <k2/fst.h>


namespace k2 {

/*
  The core part of Connect().  Connect() removes states that are not accessible
  or are not coaccessible, i.e. not reachable from start state or cannot reach the
  final state.

  If the resulting Fsa is empty, `state_map` will be empty at exit.

     @param [in] fsa  The FSA to be connected.  Requires
     @param [out] state_map   Maps from state indexes in the output fsa to
                      state indexes in `fsa`.  Will retain the original order,
                      so the output will be topologically sorted if the input
                      was.
 */
void ConnectCore(const Fsa &fsa,
                 std::vector<int32> *state_map);


/*
  Removes states that are not accessible (from the start state) or are not
  coaccessible (i.e. that cannot reach the final state), and ensures that if the
  FSA admits a topological sorting (i.e. if contains no cycles except
  self-loops), the version that is output is topologically sorted.

     @param [in] a    Input FSA
     @param [out] b   Output FSA, that will be trim / connected (there are
                     two terminologies).

     @param [out] state_map   If non-NULL, to here this function will
                     output a map from state-index in `b` to the corresponding
                     state-index in `a`.

  Notes:
    - If `a` admitted a topological sorting, b will be topologically
      sorted.
    - If `a` was deterministic, `b` will be deterministic; same for
      epsilon free, obviously.
    - `b` will be arc-sorted (arcs sorted by label)
    - `b` will (obviously) be connected

  Also if `a`
 */
void Connect(const Fsa &a,
             Fsa *b,
             std::vector<int32> *state_map = NULL);




/*
  Compute the intersection of two FSAs; this is the equivalent of composition
  for automata rather than transducers, and can be used as the core of
  composition.

  @param [in] a    One of the FSAs to be intersected.  Must satisfy
                   CheckProperties(a, kArcSorted)
  @param [in] b    The other FSA to be intersected  Must satisfy
                   CheckProperties(b, kArcSorted), and either a or b
                   must be epsilon-free (c.f. IsEpsilonFree); this
                   ensures that epsilons do not have to be treated
                   differently from any other symbol.
  @param [out] c   The composed FSA will be output to here.
  @param [out] src_a   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `a` was.
  @param [out] src_b   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `b` was.
 */
void Intersect(const Fsa &a,
               const Fsa &b,
               Fsa *c,
               std::vector<int32> *src_a = NULL,
               std::vector<int32> *src_b = NULL);

/**
   Intersection of two weighted FSA's: the same as Intersect(), but it prunes
   based on the sum of two costs.  Note: although these costs are provided per
   arc, they would usually be a sum of forward and backward costs, that's
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




   as composition, but for finite state automata rather than transducers

   This version is for where they both have weights... can also make one for
   where just one has weights.
 */
void IntersectPrunedWW(const Fsa &a, const float *a_cost,
                       const Fsa &b, const float *b_cost,
                       float cutoff,
                       Fsa *c,
                       std::vector<int32> *deriv_a,
                       std::vector<int32> *deriv_b);


void RandomPath(const Fsa &a,
                const float *a_cost,
                Fsa *b,
                std::vector<int32> *state_map = NULL);


} // namespace k2
