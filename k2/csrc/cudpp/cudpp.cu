#include "k2/csrc/cudpp/cudpp_multisplit.h"

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
 * @param[in] plan An instance of CUDPPMultiSplitPlan
 * @param[in,out] d_keys  Input data
 * @param[in,out] d_values Output data
 * @param[in] numElements Number of elements
 * @param[in] numBuckets Number of buckets
 * @param[in] bucketMappingFunc function that maps an element to a bucket
 */
void cudppMultiSplitCustomBucketMapper(CUDPPMultiSplitPlan *plan,
                                       unsigned int *d_keys,
                                       unsigned int *d_values,
                                       size_t numElements, size_t numBuckets,
                                       BucketMappingFunc bucketMappingFunc) {
  cudppMultiSplitDispatch(d_keys, d_values, numElements, numBuckets,
                          bucketMappingFunc, plan);
}
