#include "fsa.h"
#include "properties.h"

namespace k2 {
    
    bool IsTopSorted(const Fsa &fsa){
        for(auto &range: fsa.leaving_arcs){
            for(std::size_t arc_idx = range.begin; arc_idx < range.end; ++arc_idx){
                const Arc &arc = fsa.arcs[arc_idx];
                if(arc.dest_state < arc.src_state){
                    return false;
                }
            }
        }
        return true;
    }

    bool HasSelfLoops(const Fsa &fsa){
        //TODO: refactor code below as we have so many for-for-loop structures 
        for(auto &range: fsa.leaving_arcs){
            for(std::size_t arc_idx = range.begin; arc_idx < range.end; ++arc_idx){
                const Arc &arc = fsa.arcs[arc_idx];
                if(arc.dest_state == arc.src_state){
                    return false;
                }
            }
        }
        return true;
    }

} // namespace k2
