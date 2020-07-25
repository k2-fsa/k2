

// Note: b is FsaVec<Arc>.
void Intersect(const DenseFsa &a, const FsaVec &b, Fsa *c,
               Array<int32_t> *arc_map_a = nullptr,
               Array<int32_t> *arc_map_b = nullptr);



// compose/intersect array of FSAs (multiple streams decoding or training in
// parallel, in a batch)... basically composition with frame-synchronous beam pruning,
// like in speech recognition.
//
// This code is intended to run on GPU (but should also work on CPU).
void IntersectDensePruned(Array3<Arc> &a_fsas,
                          DenseFsaVec &b_fsas,
                          float beam,
                          int32_t max_states,
                          FsaVec *ofsa,
                          Array<int> *arc_map_a,
                          Array<int> *arc_map_b);
