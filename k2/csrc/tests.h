#include "k2/fsa.h"

namespace k2 {

/*
  Returns true if the Fsa `a` is equivalent to `b`.
  CAUTION: this one will be quite hard to implement.
 */
bool IsEquivalent(const Fsa &a, const Fsa &b);


/* Gets a random path from an Fsa `a`
 */
void RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32> *state_map = NULL);



}

