/**
 * @brief
 * Put cuda runtime headers here.
 * With this file be included, code static analysis works.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_CUDA_HEADERS_H_
#define K2_CSRC_CUDA_HEADERS_H_

#ifdef __CUDACC__
#define K2_CUDA_HOSTDEV __host__ __device__
#else
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#define K2_CUDA_HOSTDEV
#endif

#endif  // K2_CSRC_CUDA_HEADERS_H_
