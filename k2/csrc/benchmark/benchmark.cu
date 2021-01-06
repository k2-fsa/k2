/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <ctime>
#include <iostream>
#include <regex>  // NOLINT
#include <sstream>
#include <string>

#include "k2/csrc/benchmark/benchmark.h"
#include "k2/csrc/benchmark/helper_cuda.h"
#include "k2/csrc/log.h"
#include "k2/csrc/math.h"
#include "k2/csrc/version.h"

namespace k2 {

std::string DeviceInfo::ToString() const {
  std::ostringstream os;
  os << kPrefix << "CUDA device name: " << device_name << "\n";
  os << kPrefix << "Compute capability: " << compute_capability_major << "."
     << compute_capability_minor << "\n";
  os << kPrefix << "GPU clock freq: " << gpu_clock_freq << " GHz"
     << "\n";
  os << kPrefix << "Driver version: " << driver_version_major << "."
     << driver_version_minor << "\n";
  os << kPrefix << "Runtime version: " << runtime_version_major << "."
     << runtime_version_minor << "\n";
  os << kPrefix << "Warp size: " << warp_size << "\n";
  os << kPrefix << "L2 cache size: " << l2_cache_size << " MB"
     << "\n";
  os << kPrefix << "Total global memory: " << total_global_mem << " GB"
     << "\n";
  os << kPrefix << "Total const memory: " << total_const_mem << " KB"
     << "\n";
  os << kPrefix
     << "Total shared memory per block: " << total_shared_mem_per_block << " KB"
     << "\n";
  os << kPrefix
     << "Total shared memory per multiprocessor: " << total_shared_mem_per_mp
     << " KB"
     << "\n";
  os << kPrefix << "ECC enabled: " << ecc_enabled << "\n";
  os << kPrefix << "Number of multiprocessors: " << num_multiprocessors << "\n";
  os << kPrefix << "Number of CUDA cores: " << num_cuda_cores << "\n";
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const DeviceInfo &info) {
  return os << info.ToString();
}

DeviceInfo GetDeviceInfo() {
  DeviceInfo info;

  int32_t driver_version;
  K2_CUDA_SAFE_CALL(cudaDriverGetVersion(&driver_version));
  info.driver_version_major = driver_version / 1000;
  info.driver_version_minor = (driver_version % 1000) / 10;

  int32_t runtime_version;
  K2_CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime_version));
  info.runtime_version_major = runtime_version / 1000;
  info.runtime_version_minor = (runtime_version % 1000) / 10;

  int32_t current_device;
  K2_CUDA_SAFE_CALL(cudaGetDevice(&current_device));

  cudaDeviceProp prop;
  K2_CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, current_device));
  info.device_name = prop.name;
  info.compute_capability_major = prop.major;
  info.compute_capability_minor = prop.minor;
  info.gpu_clock_freq = prop.clockRate / 1000. / 1000.;
  info.warp_size = prop.warpSize;
  info.l2_cache_size = prop.l2CacheSize / 1024. / 1024.;
  info.total_global_mem = prop.totalGlobalMem / 1024. / 1024. / 1024.;
  info.total_const_mem = prop.totalConstMem / 1024.;
  info.total_shared_mem_per_block = prop.sharedMemPerBlock / 1024.;
  info.total_shared_mem_per_mp = prop.sharedMemPerMultiprocessor / 1024.;
  info.ecc_enabled = prop.ECCEnabled;
  info.num_multiprocessors = prop.multiProcessorCount;
  info.num_cuda_cores = _ConvertSMVer2Cores(prop.major, prop.minor);
  return info;
}

std::string BenchmarkRun::GetFieldsName() {
  std::ostringstream os;
  os << "op_name,dtype,device,problem_size,"
        "number_of_iterations,elapsed_us_per_iteration";
  return os.str();
}

std::string BenchmarkRun::ToString() const {
  std::ostringstream os;
  os << stat.op_name << "," << stat.dtype_name << "," << stat.device_type << ","
     << stat.problem_size << "," << stat.num_iter << "," << std::fixed
     << stat.eplased_per_iter;
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const BenchmarkRun &run) {
  return os << run.ToString();
}

std::vector<std::unique_ptr<BenchmarkInstance>> *GetRegisteredBenchmarks() {
  static std::vector<std::unique_ptr<BenchmarkInstance>> instances;
  return &instances;
}

void RegisterBenchmark(const std::string &name, BenchmarkFunc func) {
  auto benchmark_inst = std::make_unique<BenchmarkInstance>(name, func);
  GetRegisteredBenchmarks()->emplace_back(std::move(benchmark_inst));
}

std::vector<BenchmarkRun> RunBechmarks() {
  auto &registered_benchmarks = *GetRegisteredBenchmarks();
  std::vector<BenchmarkRun> results;
  for (const auto &b : registered_benchmarks) {
    BenchmarkRun run;
    run.name = b->name;
    run.stat = b->func();
    results.push_back(run);
  }
  return results;
}

void FilterRegisteredBenchmarks(const std::string &pattern) {
  std::regex regex(pattern);
  std::smatch match;
  auto &benchmarks = *GetRegisteredBenchmarks();

  std::vector<std::unique_ptr<BenchmarkInstance>> kept;
  for (auto &b : benchmarks) {
    if (std::regex_search(b->name, match, regex)) {
      kept.emplace_back(std::move(b));
    }
  }
  std::swap(kept, benchmarks);
}

void PrintRegisteredBenchmarks() {
  auto &benchmarks = *GetRegisteredBenchmarks();
  for (const auto &b : benchmarks) {
    std::cout << kPrefix << b->name << "\n";
  }
}

std::string GetCurrentDateTime() {
  std::time_t t = std::time(nullptr);
  return std::string(kPrefix) + std::ctime(&t);
}

void PrintEnvironmentInfo() {
  std::ostringstream os;
  os << GetCurrentDateTime();
  os << GetDeviceInfo() << "\n";

  os << kPrefix << "cuDNN version: " << kCudnnVersion << "\n";
  os << kPrefix << "Build type: " << kBuildType << "\n";
  os << kPrefix << "OS: " << kOS << "\n";
  os << kPrefix << "CMake version: " << kCMakeVersion << "\n";
  os << kPrefix << "GCC version: " << kGCCVersion << "\n";
  os << kPrefix << "CMake CUDA flags: " << kCMakeCudaFlags << "\n";
  os << kPrefix << "CMake CXX flags: " << kCMakeCxxFlags << "\n";
  os << kPrefix << "torch version: " << kTorchVersion << "\n";
  os << kPrefix << "torch CUDA version: " << kTorchCudaVersion << "\n";
  os << kPrefix << "NVTX enabled: " << kEnableNvtx << "\n";
  os << kPrefix << "Debug disabled: " << internal::kDisableDebug << "\n";
  os << kPrefix
     << "cuda device sync enabled: " << internal::EnableCudaDeviceSync()
     << "\n";
  os << kPrefix << "Checks disabled: " << internal::DisableChecks() << "\n";
  os << kPrefix << "Seed: " << GetSeed() << "\n";

  // print it to stderr so that it can be redirected
  std::cerr << os.str() << "\n";
}

}  // namespace k2
