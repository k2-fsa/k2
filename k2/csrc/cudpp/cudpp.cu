// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * cudpp.cpp
 *
 * @brief Main library source file.  Implements wrappers for public
 * interface.
 *
 * Main library source file.  Implements wrappers for public
 * interface.  These wrappers call application-level operators.
 * As this grows we may decide to partition into multiple source
 * files.
 */

/**
 * \defgroup publicInterface CUDPP Public Interface
 * The CUDA public interface comprises the functions, structs, and enums
 * defined in cudpp.h.  Public interface functions call functions in the
 * \link cudpp_app Application-Level\endlink interface. The public
 * interface functions include Plan Interface functions and Algorithm
 * Interface functions.  Plan Interface functions are used for creating
 * CUDPP Plan objects that contain configuration details, intermediate
 * storage space, and in the case of cudppSparseMatrix(), data.  The
 * Algorithm Interface is the set of functions that do the real work
 * of CUDPP, such as cudppScan() and cudppSparseMatrixVectorMultiply().
 *
 * @{
 */

/** @name Algorithm Interface
 * @{
 */

#include "cudpp.h"
#include "cudpp_multisplit.h"
#include <stdio.h>


/**
 * @brief Splits an array of keys and an optional 
 * array of values into a set of buckets.
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs an arrays of keys and (optionally) values in place,
 * where the keys and values have been split into ordered buckets.
 * Key-value or key-only multisplit is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY or
 * CUDPP_OPTION_KEY_VALUE_PAIRS. The function used to map a key to a bucket
 * is selected through the configuration option 'bucket_mapper'. 
 * The current options are:
 *
 * ORDERED_CYCLIC_BUCKET_MAPPER (default):
 * bucket = (key % numElements) / ((numElements + numBuckets - 1) / numBuckets);
 *
 * MSB_BUCKET_MAPPER:
 * bucket = (key >> (32 - ceil(log2(numBuckets)))) % numBuckets;
 *
 * Currently, the only supported key and value type is CUDPP_UINT.
 *
 *
 * @param[in] planHandle Handle to plan for CUDPPMultiSplitPlan
 * @param[in,out] d_keys keys by which key-value pairs will be split
 * @param[in,out] d_values values to be split
 * @param[in] numElements number of elements in d_keys and d_values
 * @param[in] numBuckets Number of buckets
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMultiSplit(CUDPPMultiSplitPlan *plan,
                            unsigned int      *d_keys,
                            unsigned int      *d_values,
                            size_t            numElements,
                            size_t            numBuckets)
{
  cudppMultiSplitDispatch(d_keys, d_values, numElements, numBuckets, NULL,
                          plan);
  return CUDPP_SUCCESS;
}

/**
 * @brief Splits an array of keys and an optional array of values into 
 * a set of buckets using a custom function to map elements to buckets.
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs an arrays of keys and (optionally) values in place,
 * where the keys and values have been split into ordered buckets.
 * Key-value or key-only multisplit is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY or
 * CUDPP_OPTION_KEY_VALUE_PAIRS. To use this function, the 
 * configuration option 'bucket_mapper' must be set to CUSTOM_BUCKET_MAPPER.
 * This option lets the library know to use the custom function pointer,
 * specified in the last argument, when assigning an element to a bucket.
 * The user specified bucket mapper must be a function pointer to a device
 * function that takes one unsigned int argument (the element) and returns 
 * an unsigned int (the bucket). 
 *
 *
 * Currently, the only supported key and value type is CUDPP_UINT.
 *
 * @param[in] planHandle Handle to plan for BWT
 * @param[in,out] d_keys  Input data
 * @param[in,out] d_values Output data
 * @param[in] numElements Number of elements
 * @param[in] numBuckets Number of buckets
 * @param[in] bucketMappingFunc function that maps an element to a bucket
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
#if 0
CUDPP_DLL
CUDPPResult cudppMultiSplitCustomBucketMapper(const CUDPPHandle planHandle,
                                              unsigned int      *d_keys,
                                              unsigned int      *d_values,
                                              size_t            numElements,
                                              size_t            numBuckets,
                                              BucketMappingFunc bucketMappingFunc)
{
    CUDPPMultiSplitPlan *plan =
        (CUDPPMultiSplitPlan*)getPlanPtrFromHandle<CUDPPMultiSplitPlan>(planHandle);

    if (plan != NULL)
    {
      cudppMultiSplitDispatch(d_keys, d_values, numElements, numBuckets,
        bucketMappingFunc, plan);
          return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_INVALID_HANDLE;
    }
}

#endif

/** @} */ // end Algorithm Interface
/** @} */ // end of publicInterface group

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
