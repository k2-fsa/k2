/**
 * @brief
 * nvtx.cu
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/nvtx.h"
#include "nvToolsExt.h"

namespace k2 {

#ifdef K2_ENABLE_NVTX
NvtxRange::NvtxRange(const char *name) { nvtxRangePushA(name); }
NvtxRange::~NvtxRange() { nvtxRangePop(); }
#else
NvtxRange::NvtxRange(const char *) {}
NvtxRange::~NvtxRange() = default;
#endif

}  // namespace k2
