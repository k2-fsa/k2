/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

using namespace std::chrono_literals;  // NOLINT

TEST(Nvtx, Sleep) {
  {
    NVTX_RANGE("Sleep 2s");
    std::this_thread::sleep_for(2000ms);
  }
  {
    NVTX_RANGE("Sleep 1s");
    std::this_thread::sleep_for(1000ms);

    NVTX_RANGE("Sleep 1s");
    std::this_thread::sleep_for(1000ms);
  }
}

}  // namespace k2
