#include <k2/fsa.h>


namespace k2 {


/*
  Computes lists of arcs entering each state (needed for algorithms that
  traverse the Fsa in reverse order).

  Requires that `fsa` be valid and top-sorted, i.e.
  CheckProperties(fsa, KTopSorted) == true.
*/
void GetEnteringArcs(const Fsa &fsa,
                     VecOfVec *entering_arcs);




} // namespace k2
