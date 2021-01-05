// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_PLAN_H__
#define __CUDPP_PLAN_H__

#include "cudpp.h"

typedef void* KernelPointer;




/** @brief Plan class for MultiSplit
*
*/
class CUDPPMultiSplitPlan
{
public:
 CUDPPMultiSplitPlan(CUDPPConfiguration config, size_t numElements,
                     size_t numBuckets);
 ~CUDPPMultiSplitPlan();
 CUDPPConfiguration m_config;  //!< @internal Options structure

 unsigned int m_numElements;
 unsigned int m_numBuckets;
 unsigned int *m_d_mask;
 unsigned int *m_d_out;
 unsigned int *m_d_fin;
 unsigned int *m_d_temp_keys;
 unsigned int *m_d_temp_values;
 unsigned long long int *m_d_key_value_pairs;
};

#endif // __CUDPP_PLAN_H__
