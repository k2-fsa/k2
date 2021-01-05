// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp.h
 * 
 * @brief Main library header file.  Defines public interface.
 *
 * The CUDPP public interface is a C-only interface to enable 
 * linking with code written in other languages (e.g. C, C++, 
 * and Fortran).  While the internals of CUDPP are not limited 
 * to C (C++ features are used), the public interface is 
 * entirely C (thus it is declared "extern C").
 */

#ifndef __CUDPP_H__
#define __CUDPP_H__

#include <stdlib.h> // for size_t

class CUDPPMultiSplitPlan;

/**
 * @brief CUDPP Result codes returned by CUDPP API functions.
 */
enum CUDPPResult
{
    CUDPP_SUCCESS = 0,                 /**< No error. */
    CUDPP_ERROR_INVALID_HANDLE,        /**< Specified handle (for example, 
                                            to a plan) is invalid. **/
    CUDPP_ERROR_ILLEGAL_CONFIGURATION, /**< Specified configuration is
                                            illegal. For example, an
                                            invalid or illogical
                                            combination of options. */
    CUDPP_ERROR_INVALID_PLAN,          /**< The plan is not configured properly.
                                            For example, passing a plan for scan
                                            to cudppSegmentedScan. */
    CUDPP_ERROR_INSUFFICIENT_RESOURCES,/**< The function could not complete due to
                                            insufficient resources (typically CUDA
                                            device resources such as shared memory)
                                            for the specified problem size. */
    CUDPP_ERROR_UNKNOWN = 9999         /**< Unknown or untraceable error. */
};

#ifdef __cplusplus
extern "C" {
#endif


/** 
 * @brief Options for configuring CUDPP algorithms.
 * 
 * @see CUDPPConfiguration, cudppPlan, CUDPPAlgorithm
 */
enum CUDPPOption
{
    CUDPP_OPTION_FORWARD   = 0x1,  /**< Algorithms operate forward:
                                    * from start to end of input
                                    * array */
    CUDPP_OPTION_BACKWARD  = 0x2,  /**< Algorithms operate backward:
                                    * from end to start of array */
    CUDPP_OPTION_EXCLUSIVE = 0x4,  /**< Exclusive (for scans) - scan
                                    * includes all elements up to (but
                                    * not including) the current
                                    * element */
    CUDPP_OPTION_INCLUSIVE = 0x8,  /**< Inclusive (for scans) - scan
                                    * includes all elements up to and
                                    * including the current element */
    CUDPP_OPTION_CTA_LOCAL = 0x10, /**< Algorithm performed only on
                                    * the CTAs (blocks) with no
                                    * communication between blocks.
                                    * @todo Currently ignored. */
    CUDPP_OPTION_KEYS_ONLY = 0x20, /**< No associated value to a key 
                                    * (for global radix sort) */
    CUDPP_OPTION_KEY_VALUE_PAIRS = 0x40, /**< Each key has an associated value */
};


/** 
 * @brief Datatypes supported by CUDPP algorithms.
 *
 * @see CUDPPConfiguration, cudppPlan
 */
enum CUDPPDatatype
{
    CUDPP_CHAR,     //!< Character type (C char)
    CUDPP_UCHAR,    //!< Unsigned character (byte) type (C unsigned char)
    CUDPP_SHORT,    //!< Short integer type (C short)
    CUDPP_USHORT,   //!< Short unsigned integer type (C unsigned short)
    CUDPP_INT,      //!< Integer type (C int)
    CUDPP_UINT,     //!< Unsigned integer type (C unsigned int)
    CUDPP_FLOAT,    //!< Float type (C float)
    CUDPP_DOUBLE,   //!< Double type (C double)
    CUDPP_LONGLONG, //!< 64-bit integer type (C long long)
    CUDPP_ULONGLONG,//!< 64-bit unsigned integer type (C unsigned long long)
    CUDPP_DATATYPE_INVALID,  //!< invalid datatype (must be last in list)
};

/** 
 * @brief Operators supported by CUDPP algorithms (currently scan and
 * segmented scan).
 *
 * These are all binary associative operators.
 *
 * @see CUDPPConfiguration, cudppPlan
 */
enum CUDPPOperator
{
    CUDPP_ADD,      //!< Addition of two operands
    CUDPP_MULTIPLY, //!< Multiplication of two operands
    CUDPP_MIN,      //!< Minimum of two operands
    CUDPP_MAX,      //!< Maximum of two operands
    CUDPP_OPERATOR_INVALID, //!< invalid operator (must be last in list)
};

/**
* @brief Algorithms supported by CUDPP.  Used to create appropriate plans using
* cudppPlan.
* 
* @see CUDPPConfiguration, cudppPlan
*/
enum CUDPPAlgorithm
{
    CUDPP_SCAN,              //!< Scan or prefix-sum
    CUDPP_SEGMENTED_SCAN,    //!< Segmented scan
    CUDPP_COMPACT,           //!< Stream compact
    CUDPP_REDUCE,            //!< Parallel reduction
    CUDPP_SORT_RADIX,        //!< Radix sort
    CUDPP_SORT_MERGE,        //!< Merge Sort
    CUDPP_SORT_STRING,       //!< String Sort
    CUDPP_SPMVMULT,          //!< Sparse matrix-dense vector multiplication
    CUDPP_RAND_MD5,          //!< Pseudorandom number generator using MD5 hash algorithm
    CUDPP_TRIDIAGONAL,       //!< Tridiagonal solver algorithm
    CUDPP_COMPRESS,          //!< Lossless data compression
    CUDPP_LISTRANK,          //!< List ranking
    CUDPP_BWT,               //!< Burrows-Wheeler transform
    CUDPP_MTF,               //!< Move-to-Front transform
    CUDPP_SA,                //!< Suffix Array algorithm
    CUDPP_MULTISPLIT,        //!< Multi-Split algorithm
    CUDPP_ALGORITHM_INVALID, //!< Placeholder at end of enum
};

/**
 * @brief Operators supported by CUDPP algorithms (currently scan and
 * segmented scan).
 *
 * BLAH
 *
 * @see CUDPPConfiguration, cudppPlan
 */
enum CUDPPBucketMapper
{
    CUDPP_LSB_BUCKET_MAPPER,      //!< The bucket is determined by the element's LSBs.
    CUDPP_MSB_BUCKET_MAPPER,      //!< The bucket is determined by the element's MSBs.
    CUDPP_DEFAULT_BUCKET_MAPPER,  //!< The bucket is determined by the element's value.
    CUDPP_CUSTOM_BUCKET_MAPPER,   //!< The bucket mapping is a user-specified function.
};

/**
* @brief Configuration struct used to specify algorithm, datatype,
* operator, and options when creating a plan for CUDPP algorithms.
*
* @see cudppPlan
*/
struct CUDPPConfiguration
{
    CUDPPAlgorithm    algorithm;     //!< The algorithm to be used
    CUDPPOperator     op;            //!< The numerical operator to be applied
    CUDPPDatatype     datatype;      //!< The datatype of the input arrays
    unsigned int      options;       //!< Options to configure the algorithm
    CUDPPBucketMapper bucket_mapper; //!< The bucket mapping function for multisplit
};

typedef unsigned int (*BucketMappingFunc)(unsigned int);

#define CUDPP_INVALID_HANDLE 0xC0DABAD1
typedef size_t CUDPPHandle;


#ifdef WIN32
    #if defined(CUDPP_STATIC_LIB)
        #define CUDPP_DLL
    #elif defined(cudpp_EXPORTS) || defined(cudpp_hash_EXPORTS)
        #define CUDPP_DLL __declspec(dllexport)
    #else
        #define CUDPP_DLL __declspec(dllimport)
    #endif    
#else
    #define CUDPP_DLL
#endif

// MultiSplit
CUDPP_DLL
CUDPPResult cudppMultiSplit(CUDPPMultiSplitPlan *plan,
                            unsigned int      *d_keys,
                            unsigned int      *d_values,
                            size_t            numElements,
                            size_t            numBuckets);

// MultiSplit with user-specified bucket mapping function
CUDPP_DLL
// CUDPPResult cudppMultiSplitCustomBucketMapper(const CUDPPHandle planHandle,
//                                               unsigned int      *d_keys,
//                                               unsigned int      *d_values,
//                                               size_t            numElements,
//                                               size_t            numBuckets,
//                                               BucketMappingFunc bucketMappingFunc);

#ifdef __cplusplus
}
#endif


#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
