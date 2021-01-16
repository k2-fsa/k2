/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 *
 * Test for NVTX.
 *
 * Usage:
 *
 * 1. Go to https://developer.nvidia.com/gameworksdownload
 * to download "Nsight Systems".
 *
 *   Example: Download
 *
 * https://developer.nvidia.com/rdp/assets/nsight-systems-2020-4h-linux-installer
 *
 * 2. Install it:
 *
 *    chmod +x nsight-systems-2020-4h-linux-installer
 *    ./nsight-systems-2020-4h-linux-installer
 *
 *    The default installation path is `/opt/nvidia/nsight-systems/2020.4.1`
 *
 * 3. Add `/opt/nvidia/nsight-systems/2020.4.1/bin` to `PATH`.
 *
 * 4. There are various subcommands of `nsys`. One example usage is:
 *
 *      nsys nvprof ./bin/cu_nvtx_test
 *
 * 5. References:
 *
 *    - https://developer.nvidia.com/nsight-systems
 *    - https://docs.nvidia.com/nsight-systems/index.html
 *
 */

#include <unistd.h>

#include "gtest/gtest.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

TEST(Nvtx, Sleep) {
  {
    NVTX_RANGE("Sleep 2s");
    sleep(2);
  }
  {
    NVTX_RANGE("Sleep 1s");
    sleep(1);

    NVTX_RANGE("Sleep 1s");
    sleep(1);
  }
}

}  // namespace k2
