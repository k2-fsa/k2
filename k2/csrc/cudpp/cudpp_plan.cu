// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp_plan.h"
#include "cudpp_multisplit.h"

/** @brief CUDPP MultiSplit Plan Constructor
  *
  * @param[in] config The configuration struct specifying options
  * @param[in] The number of elements to be split
  * @param[in] The number of buckets
  *
  */
CUDPPMultiSplitPlan::CUDPPMultiSplitPlan(CUDPPConfiguration config,
                                         size_t numElements, size_t numBuckets)
    : m_config(config) {
  m_numElements = numElements;
  m_numBuckets = numBuckets;

  allocMultiSplitStorage(this);

  // use the allocated array for temporary storage of keys and values
  m_d_temp_keys = (unsigned int *)m_d_key_value_pairs;
  m_d_temp_values = (unsigned int *)m_d_key_value_pairs + numElements;
}

/** brief MultiSplit Plan Destructor*/
CUDPPMultiSplitPlan::~CUDPPMultiSplitPlan()
{
    freeMultiSplitStorage(this);
}
