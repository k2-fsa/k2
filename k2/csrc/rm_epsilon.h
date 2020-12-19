/**
 * @brief
 * remove epsilon
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RM_EPSILON_H_
#define K2_CSRC_RM_EPSILON_H_

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

/*
  Notes on our iterative epsilon removal algorithm (for tropical semiring),
  RemoveEpsilonsIterativeTropical().

  We first separate epsilon and non-epsilon arcs (while leaving the
  state numbering fixed); doing a closure on the epsilon part so we have direct
  epsilon arcs between pairs of states that can reach each other by epsilons;
  then combining the epsilon and non-epsilon arcs to produce non-epsilon arcs.

  We keep the original numbering of the states.  We'll us capital letters for
  sets of arcs.  Note: epsilon self-loops can be discarded whenever they appear
  if their score is <= 0; if there is scores is >0 we can abort the algorithm
  because it woudl imply that the FSA has some paths with infinite score but
  finite number of real symbols.


     N = set of non-epsilon arcs in thw input
     E = set of epsilon-arcs e in input
     C = closure of set of epsilon-arcs e in input, i.e. for distinct states
         a,b,c, if C contains epsilon arc from a->b and b->c, it will alos
         contain one from a->c.
     C_f = subset of C that we decide, based on heuristics with the goal of
         minimizing the size of the output, to combine with the following
         non-epsilon arcs ,
    C_p = subset of C, C \ C_f (C minus C_f), that we decide to combine with the
         preceding non-epsilon arc, based on a heuristic that we choose, with
         the constraint that arcs that leave the start-state cannot be in P.
     F = set of arcs that arises from combination of arcs e in C_f with all arcs in N
         that leave the destination-state of e, i.e. arc-combinations [e,n];
         we'll use square brackets for sequences of arcs to distinguish from other
         uses of parentheses.
     P = set of arcs that arises from combination of arcs e in C_p with all arcs in N
         that enter the source-state of e, i.e. arc-combinations [n,e].
     Q = set of arcs that arises from combination of arcs e in C_p with all arcs in F
         that enter the source-state of e.  i.e. from arc-combinations [f,e]

  At the start of the algorithm we have the set of arcs
     N u E    (u means union).
  After epsilon-closure we have:
     N u C     ...  this is equivalent to N u E by properties of epsilon-closure.
  Then we divide C in to two subsets: we have
     N u C_f u C_p   ... equivalent to N u C for obvious reasons.
  Then we add F:
     N u C_f u C_p u F .. equivalent to the above because for each arc in F
                      there is a path of two arcs in (C_f, N) that has the same
                      weight and symbol
  Then we add P:
     N u C_f u C_p u F u P  .. equivalent to the above because for each arc in P
                       there is a path of two arcs in (N, C_p) that has the same
                       weight and symbol
  Then we add Q:
     N u C_f u C_p u F u P u Q .. equivalent to the above because for each arc in Q
                       there is a path of two arcs in (P, C_p) that has the same
                       weight and symbol

 The final stage is to remove C_f and C_p, leaving us with
     U u F u P u Q
 which is epsilon-free.  We need to demonstrate that we can remove C_f and C_p
 while preserving equivalence.  This is the not-quite-so-trivial part of the proof.
 Let A be the FSA containing arcs

     N u C_f u C_p u F u P u Q

 and B be the FSA containing arcs

     U u F u P u Q.

 We show that B and A are equivalent by showing that for any path in A that has
 at least one epsilon arc, there is a path in B with the same "real" symbol sequence
 (i.e. after removing epsilons) and a score that is at least as large.  We
 show this inductively by demonstrating that for any path P in A that has at
 least one epsilon, there is a path P' in A that satisifes the following properties:

    - The score of P' is >= the score of P
    - P' has the same real symbol-sequence as P (i.e. the same sequence after
      removing epsilons)
    - Either:
       - (a) P' has one fewer epsilon arc than P, or
       - (b) P' has the same number of epsilons as P, but one fewer
             "implicit epsilons", where we consider that arcs in F or P
             have one "implicit epsilon" and arcs in Q have two
             "implicit epsilons".

 The above lets us argue that we will eventually reach a path in A with no epsilons,
 which must also be in B.

===

 Consider the path P in A; we want to show there is a path P' with the properties
 mentioned above.

   (i) First imagine that P has two successive epsilon arcs, from states a->b and
  b->c.  These arcs must both be in either C_f or C_p and hence in C.  But because C is the
  result of epsilon closure there must be an epsilon from a->c with a score at
  least as great as the sum of the original two paths' scores; so we can reduce the
  number of epsilons by one.

  [[Note regarding epsilon-loops: if a and c are the same state, there are two
  choices: if the score is <=0 then we can remove the epsilon from our path
  entirely while having the same symnbol sequence and at least as large a score;
  if the score is >0 then the graph contained positive-score epsilon cycles
  which is equivalent to infinite score, which is invalid.]]

  We can use (i) as needed until there are no successive epsilons, so the following
  arguments will consider only isolated epsilons.  Note on terminology: we use
  letters a,b,c,d for epsilon arcs below.

   (ii) Consider an epsilon-arc a in C_f.  This must be followed by a non-epsilon
     arc (assuing we already reduced via (i)), and this non-epsilon arc must
     be in N, F, P or Q.  Briefly the cases are as follows:
        - following arc n is in N -> we can replace [a,n] by the
                          appropriate arc f in F, leaving us with one fewer epsilon.
        - following arc f is in F -> there was originally a pair [b,n]
                           from which f was constructed; so we can expand f to
                           [b,n] and then reduce [a,b] to a single epsilon c via
                           argument (i); this leaves us with [c,n], which is
                           case (b) above, i.e. the same number of epsilons
                           but one fewer implicit epsilon.
        - following arc p is in P -> there was originally a pair [n,b] from which
                           p was constructed, where b is in C_p; and there is an
                           arc f in F to which [a,n] was expanded;
                           and an arc q in Q to which [f,b] was expanded; so
                           we can reduce [a,p] to q.
        - following arc q is in Q -> arc q was expanded from arcs [f,b] with f in F
                           and b in C_p; and F was expanded from [c,n] with c
                           in C_f and n in N, so we can reduce sequence  [a,c,n,b]
                           to [d,n,b] by reducing [a,c] to d via (i), and further
                           to [d,p] by combining [n,b] to the arc in P that was
                           constructed from that pair; so we have reduced [a,q]
                           to [d,p] which has one fewer implicit epsilon.

  (iii) Consider an epsilon-arc a in C_p.   (Note: by construction, this cannot
     leave the start state so there must be a preceding arc).  We are assuming we
     already reduced via (i) so the preceding arc is non-epsilon; it must be in
     N, F, P or Q.
        - preceding arc n is in N -> we can replace [n,a] by the
                          appropriate arc p in F, leaving us with one fewer epsilon.
        - preceding arc f is in F -> we can replace [f,a] by the arc q in Q
                          that was constructed from it.
        - preceding arc p is in P -> there was originally a pair
                           [n,b] from which p was expanded, where b is in C_p;
                           we reduce [b,a] to c using argument (i), leaving us
                           with [n,c] which has one fewer implicit epsilon.
        - preceding arc q is in Q -> arc q was expanded from arcs [f,b] with f in F
                          and b in C_p. [b,a] to c via argument (i), leaving
                          us with [f,c] which has one fewer implicit epsilon than
                          [a,q].
*/



/*
  Extract just the epsilon arcs and the states with epsilons entering and
  leaving them (plus the start and final states).  For use inside
  epsilon-removal algorithm.
     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest  Output FsaVec; will contain all the states that had
                       epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
                       Noted if src[i] has no arc (but has some states),
                       we would not keep the start state and final state of it
                       in the corresponding output Fsa dest[i], i.e. dest[i]
                       will be an empty Fsa.
    @param [out] state_map  Will be set to a new Array1 mapping from the
                       state_idx01's in `dest` to the corresponding
                       state_idx01's in `src`.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx012's in `dest` to the corresponding arc_idx012's
                       in `src`.
*/
void ComputeEpsilonSubset(FsaVec &src, FsaVec *dest, Array1<int32_t> *state_map,
                          Array1<int32_t> *arc_map);

/*
  Extract just the non-epsilon arcs and the states that have non-epsilons
  leaving or entering them (plus the start and final states).  For use inside
  epsilon-removal algorithm.

     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest  Output FsaVec; will contain all the states that had
                       non-epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
                       Noted if src[i] has no arc (but has some states),
                       we would not keep the start state and final state of it
                       in the corresponding output Fsa dest[i], i.e. dest[i]
                       will be an empty Fsa.
     @param [out] state_map  Will be set to the renumbering object from the old
                       to new state indexes.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx012's in `dest` to the corresponding arc_idx012's
                       in `src`.
*/
void ComputeNonEpsilonSubset(FsaVec &src, FsaVec *dest, Renumbering *state_map,
                             Array1<int32_t> *arc_map);

/*
   Map some states of an FsaVec, with the arcs entering and leaving those states
      @param [in] src   Source FsaVec, to be mapped
      @param [in] state_row_splits   The row_splits vector for `dest`, which
                         determines the number of states for each output FSA
      @param [in] state_row_ids  The row_ids vector corresponding to
                        `state_row_splits`
      @param [in] state_map   Map from state_idx01's in `src` to state_idx01's
                        in `dest`, with -1 for states that are to be removed.
                        Note: the number of states in `src` may be smaller or
                        larger than state_row_ids.Dim().
                        Must have state_map.Dim() == src.TotSize(1).
      @param [out] dest  Destination FsaVec; at exit, will contain all arcs in
                        `src` whose src_state and dest_state are both kept
                        (i.e. not mapped to -1).
      @param [out] arc_map Will be set to a new Array1 that maps from arc_idx012
                        in `dest` to original arc_idx012 in `src`.

*/
void MapFsaVecStates(FsaVec &src, Array1<int32_t> &state_row_splits,
                     Array1<int32_t> &state_row_ids,
                     const Array1<int32_t> &state_map, FsaVec *dest,
                     Array1<int32_t> *arc_map);

/*
  Compute the closure of an FSA containing just epsilon arcs (as output
  by ComputeEpsilonSubset()).  This means adding epsilon arcs from
  each state s1 to each state s2 which is reachable indirectly by epsilons
  from s1 to s2.  Note: this implicitly assumes the tropical semiring, because
  we are taking only the best epsilon path from any state to any other
  state.

     @param [in] epsilon_fsa   FSA containing only epsilon arcs, as output
                           by ComputeEpsilonSubset()
     @param [out] closure_fsa  FSA containing the closure of the epsilon arcs.
                           Will be arc-sorted, and no state will have more than
                           one arc to any other state.

    CAUTION: For any epsilon cycle, e.g. s1->s1, if its score is negative or
    zero, we'll delete this arc; if its score is positive, we'll abort the
    program as positive score means we'll get infinity weight under tropical
    semiring.
*/
void ComputeEpsilonClosure(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                           Ragged<int32_t> *arc_map);

/*
 One iteration of the algorithm in ComputeEpsilonClosure().
   @param [in] epsilon_fsa  The input FSA containing only epsilon arcs
                   (possibly already passed through one or more iterations
                   of closure). Must have 3 axes.
   @param [out] closure_fsa   FSA that is the result of one iteration of
                    closure. Will contain an arc from state s1 to s2 if there
                    was already such an arc in `epsilon_fsa` or if there was
                    a state s3 such that there was an arc from s1 to s2 and
                    one from s2 to s3. Will contain at most one arc from one
                    state to any other state.
    @param [out] arc_map   For each arc in closure_fsa, contains the sequence of
                  arc_idx012's in epsilon_fsa that was the source (the sequence
                  length is 1 or 2 depending on the arc is just copying from
                  `epsilon_fsa` (s1->s2) or it's an expanded arc (s1->s3).
*/
void ComputeEpsilonClosureOneIter(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                                  Ragged<int32_t> *arc_map);

/*
  Remove epsilons from FsaOrVec in `src_fsa`, producing an FsaOrVec `dest_fsa`
  which is equivalent (in tropical semiring).  Uses an iterative algorithm which
  tries to minimize the number of arcs in the resulting FSA (epsilons are
  combined with either preceding or following arcs).

    @param [in] src_fsa    FSA to remove epsilons from.  It is an error if
                        src_fsa has epsilon loops with score greater than zero.
    @param [out] dest_fsa  Result will be written to here; will be equivalent
                          to `src_fsa` in the tropical semiring, and will be
                          epsilon-free.
    @param [out] arc_map  If not nullptr, a map from arc in `dest_fsa` to the
                          corresponding sequence of arcs in `src_fsa` will be
                          written here.

   For an explanation of how this algorithm works and a proof-sketch, see the
   comment at the top of this file.
*/
void RemoveEpsilonsIterativeTropical(FsaOrVec &src_fsa, FsaOrVec *dest_fsa,
                                     Ragged<int32_t> *arc_map = nullptr);
}  // namespace k2

#endif  // K2_CSRC_RM_EPSILON_H_
