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



/**
   This version is for where they both have weights... can also make one for
   where just one has weights.
 */
void PrunedIntersection(const Fsa &a, float *weights_a,
                        const Fsa &b, float *weights_b,
                        Fsa *c,
                        std::vector<int32> *deriv_a,
                        std::vector<int32> *deriv_b);



} // namespace k2
