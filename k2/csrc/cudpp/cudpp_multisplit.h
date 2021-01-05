// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef   __MULTISPLIT_H__
#define   __MULTISPLIT_H__

#include "cudpp.h"
#include "cudpp_plan.h"


extern "C"
void allocMultiSplitStorage(CUDPPMultiSplitPlan* plan);

extern "C"
void freeMultiSplitStorage(CUDPPMultiSplitPlan* plan);

extern "C"
void cudppMultiSplitDispatch(unsigned int *d_keys,
                             unsigned int *d_values,
                             size_t numElements,
                             size_t numBuckets,
                             BucketMappingFunc bucketMappingFunc,
                             const CUDPPMultiSplitPlan *plan);

#endif // __MULTISPLIT_H__
