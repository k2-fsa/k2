// this file is modified from
// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

#ifndef K2_CSRC_BENCHMARK_HELPER_CUDA_H_
#define K2_CSRC_BENCHMARK_HELPER_CUDA_H_

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

#endif  // K2_CSRC_BENCHMARK_HELPER_CUDA_H_
