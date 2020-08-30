// k2/csrc/cuda/arch.h

// Copyright (c) 2020 Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors
#ifndef K2_CSRC_CUDA_ARCH_H_
#define K2_CSRC_CUDA_ARCH_H_

namespace k2 {

/**
 * @def K2_PTX_ARCH
 *
 * @brief
 *  K2_PTX_ARCH get the target PTX version through the active compiler.
 *
 * @details
 *  - for host, it's 0.
 *  - for device, it's `__CUDA_ARCH__`, which indicates the
 *                compute compatibility, should >= 200.
 */
#ifndef __CUDA_ARCH__
  #define K2_PTX_ARCH 0
#else
  #define K2_PTX_ARCH __CUDA_ARCH__
#endif

}  // namespace k2

#endif  // K2_CSRC_CUDA_ARCH_H_
