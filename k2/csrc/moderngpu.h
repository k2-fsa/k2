/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_MODERN_GPU_H_
#define K2_CSRC_MODERN_GPU_H_

#ifdef K2_WITH_CUDA
#include "moderngpu/context.hxx"
#include "moderngpu/kernel_load_balance.hxx"
#include "moderngpu/kernel_mergesort.hxx"
#include "moderngpu/kernel_segsort.hxx"
#include "moderngpu/kernel_sortedsearch.hxx"
#endif

#endif  // K2_CSRC_MODERN_GPU_H_
