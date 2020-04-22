#include <vector>
#include <assert.h>

#include "../fsa.h"
#include "../properties.h"

using namespace k2;

//TODO: create Fsa examples with more elegant way (add methods addState, addArc, etc.)
Fsa CreateNonTopSortedFsaExample(){
    std::vector<Arc> arcs = {
        {0, 2, 0},
        {2, 1, 0},
        {0, 1, 0},
    };
    std::vector<Range> leaving_arcs = {
        {0, 1},
        {1, 2},
        {2, 3},
    };
    Fsa fsa;
    fsa.leaving_arcs = leaving_arcs;
    fsa.arcs = arcs;
    return fsa;
}

int main(int argc, char *argv[]){
    
    Fsa fsa = CreateNonTopSortedFsaExample();
    bool sorted = IsTopSorted(fsa);
    assert(!sorted);

    std::vector<Arc> arcs = {
        {0, 1, 0},
        {1, 2, 0},
        {0, 2, 0},
    };
    fsa.arcs = arcs;
    sorted = IsTopSorted(fsa);
    assert(sorted);
    return 0;
}
    
