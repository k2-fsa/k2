// this file is copied/modified from
// https://github.com/cudpp/cudpp/blob/master/src/cudpp/cudpp_plan.cpp
#include "k2/csrc/cudpp/cudpp.h"

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

CUDPPMultiSplitPlan::~CUDPPMultiSplitPlan() { freeMultiSplitStorage(this); }
