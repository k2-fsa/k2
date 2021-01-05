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
 * cudpp_globals.h
 *
 * @brief Global declarations defining machine characteristics of GPU target.
 * These are currently set for best performance on G8X GPUs.  The optimal 
 * parameters may change on future GPUs. In the future, we hope to make
 * CUDPP a self-tuning library.
 */

#ifndef __CUDPP_GLOBALS_H__
#define __CUDPP_GLOBALS_H__

const int SORT_CTA_SIZE = 256;                   /**< Number of threads per CTA for radix sort. Must equal 16 * number of radices */
const int SCAN_CTA_SIZE = 128;                   /**< Number of threads in a CTA */
const int REDUCE_CTA_SIZE = 256;                 /**< Number of threads in a CTA */

const int LOG_SCAN_CTA_SIZE = 7;                 /**< log_2(CTA_SIZE) */

const int WARP_SIZE = 32;                        /**< Number of threads in a warp */

const int LOG_WARP_SIZE = 5;                     /**< log_2(WARP_SIZE) */
const int LOG_SIZEOF_FLOAT = 2;                  /**< log_2(sizeof(float)) */

const int SCAN_ELTS_PER_THREAD = 8;              /**< Number of elements per scan thread */
const int SEGSCAN_ELTS_PER_THREAD = 8;           /**< Number of elements per segmented scan thread */

// BWT
#define BWT_NUMPARTITIONS 1024
#define BWT_CTA_BLOCK 128
#define BWT_BLOCKSORT_SIZE 1024
#define BWT_CTASIZE_simple 256
#define BWT_DEPTH_simple 2
#define BWT_CTASIZE_multi 256
#define BWT_DEPTH_multi 2
#define BWT_INTERSECT_B_BLOCK_SIZE_simple 1024

#define BWT_SIZE                            BWT_BLOCKSORT_SIZE*BWT_NUMPARTITIONS
#define BWT_INTERSECT_A_BLOCK_SIZE_simple   BWT_DEPTH_simple*BWT_CTASIZE_simple
#define BWT_INTERSECT_A_BLOCK_SIZE_multi    BWT_DEPTH_multi*BWT_CTASIZE_multi
#define BWT_INTERSECT_B_BLOCK_SIZE_multi    2*BWT_DEPTH_multi*BWT_CTASIZE_multi

// MTF
#define MTF_PER_THREAD      64
#define MTF_THREADS_BLOCK   64
#define MTF_LIST_SIZE       256

// Huffman
#define HUFF_THREADS_PER_BLOCK_HIST     64
#define HUFF_WORK_PER_THREAD_HIST       512
#define HUFF_THREADS_PER_BLOCK          128
#define HUFF_WORK_PER_THREAD            32
#define HUFF_NUM_CHARS                  257

#define HUFF_BLOCK_CHARS (HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD)
#define HUFF_CODE_BYTES ( ((HUFF_BLOCK_CHARS*12)%32 == 0) ? (HUFF_BLOCK_CHARS*12/32) : (HUFF_BLOCK_CHARS*12/32+1) )

#define HUFF_FALSE   0
#define HUFF_TRUE    1
#define HUFF_NONE    -1
#define HUFF_COUNT_T_MAX     UINT_MAX    /* based on count_t being unsigned int */
#define HUFF_COMPOSITE_NODE      -1      /* node represents multiple characters */
#define HUFF_EOF_CHAR    (HUFF_NUM_CHARS - 1) /* index used for EOF */

#define CUDPP_CHAR_BIT 8
#define BITS_TO_CHARS(bits)   ((((bits) - 1) / 8) + 1)
#define MS_BIT                (1 << (8 - 1))
#define BIT_CHAR(bit)         ((bit) / CUDPP_CHAR_BIT)
#define BIT_IN_CHAR(bit)      (1 << (8 - 1 - ((bit)  % 8)))

struct encoded {
    unsigned int block_size;            // Block size in words
    unsigned int code[HUFF_CODE_BYTES]; // Blocks are stored as unsigned ints
};

struct huffman_code {
    unsigned char code[32];
    unsigned char codeLen;
    unsigned int numBits;
};

struct my_huffman_node_t
{
    int value;          /* character(s) represented by this entry */
    unsigned int count;      /* number of occurrences of value (probability) */

    char ignore;        /* TRUE -> already handled or no need to handle */
    int level;          /* depth in tree (root is 0) */
        unsigned int iter;
        int left, right, parent;
};

#endif // __CUDPP_GLOBALS_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
