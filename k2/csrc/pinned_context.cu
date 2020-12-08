/**
 * @brief PinnedContext that allocates pinned memory.
 *
 * The implementation is modified from
 * https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCCachingHostAllocator.cpp
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <deque>
#include <functional>
#include <mutex>  // NOLINT
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

namespace {

struct BlockSize {
  size_t size;  // size of this memory block in bytes
  void *ptr;    // pointer to the beginning of this memory block

  explicit BlockSize(size_t size, void *ptr = nullptr) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize {
  bool allocated;  // true if the block is currently allocated
                   // false if the block is available for allocation

  int event_count;  // number of outstanding cuda events
  std::unordered_set<cudaStream_t> streams;

  Block(size_t size, void *ptr, bool allocated)
      : BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize &a, const BlockSize &b) {
  NVTX_RANGE(K2_FUNC);
  // sort by size, break ties with pointer
  if (a.size != b.size) return a.size < b.size;

  return std::less<void *>()(a.ptr, b.ptr);
}

/* Allocate pinned memory using cudaMallocHost with caching.

  WARNING: Once memory is allocated, it is not returned to the system.
  Use it with care!
 */
class PinnedAllocator {
 public:
  PinnedAllocator() : available_(&BlockComparator) {}

  /* Allocate a block of memory.

     If we can find a free block that is large enough (first fit or best fit
     as free blocks are sorted by size) for the requested size, the free block
     is marked as allocated and returned to the user.

     If no free blocks are available, a new block is allocated by
     using `cudaMallocHost`.

     @param  [in]   size        Number of bytes to be allocated.
     @param  [out]  ptr         On return, it contains the starting address of
                                the allocated memory.

     @return Return cudaSuccess on success. Return a CUDA error code on failure.
   */
  cudaError_t Malloc(size_t size, void **ptr) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_NE(ptr, nullptr);
    std::lock_guard<std::mutex> lock(mutex_);

    // If the number of outstanding cuda events is larger
    // than 100, we will invoke `ProcessEvents()`, which
    // may make some blocks available for reuse.
    //
    // If the number of cuda_events_ is small,
    // we first try to find a block from the pool. If it
    // cannot find one, we then invoke `ProcessEvents`.
    //
    // The purpose is to reduce the time of waiting for
    // the pending events.
    for (int32_t iter = 0; iter < 2; ++iter) {
      if (cuda_events_.size() > 100 || iter > 0) {
        // ProcessEvents may free blocks
        cudaError_t err = ProcessEvents();
        if (err != cudaSuccess) return err;
      }

      // search for the smallest block which can hold this allocation
      BlockSize search_key(size);
      auto it = available_.lower_bound(search_key);
      if (it != available_.end()) {
        // we find an unused block
        Block &block = blocks_.at(it->ptr);
        K2_CHECK(!block.allocated && block.event_count == 0);
        block.allocated = true;
        *ptr = block.ptr;
        available_.erase(it);
        return cudaSuccess;
      }
    }

    // we need to allocate a new block.
    // note that cudaMallocHost may not touch pointer if size is 0
    *ptr = 0;
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) return err;

    blocks_.insert({*ptr, Block(size, *ptr, true)});
    return cudaSuccess;
  }

  /* Free memory allocated by `Malloc`.

     @param [in] ptr  Pointer to the starting address of a block
                      allocated by `Malloc`.

     @return Return cudaSuccess on success. Return a CUDA error code on failure.
   */
  cudaError_t Free(void *ptr) {
    NVTX_RANGE(K2_FUNC);
    if (ptr == nullptr) return cudaSuccess;
    std::lock_guard<std::mutex> lock(mutex_);

    // process outstanding cuda events which may have occurred
    cudaError_t err = ProcessEvents();
    if (err != cudaSuccess) return err;

    auto it = blocks_.find(ptr);
    K2_CHECK(it != blocks_.end())
        << "The passed pointer is not allocated by Malloc!";

    Block &block = it->second;
    K2_CHECK(block.allocated);

    block.allocated = false;

    // insert CUDA events for each stream on which this block was used.
    err = InsertEvents(block);
    if (err != cudaSuccess) return err;

    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding cuda events
      available_.insert(block);
    }

    return cudaSuccess;
  }

  /* Record an event of a ptr with a stream.

     @param [in]  stream  A CUDA stream.
     @param [in]  ptr     It is a pointer returned by `Malloc`.
   */
  void RecordEvent(cudaStream_t stream, void *ptr) {
    NVTX_RANGE(K2_FUNC);
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(ptr);

    if (it == blocks_.end()) {
      // this pointer is not returned by `Malloc`, ignore it.
      return;
    }

    Block &block = it->second;
    K2_CHECK(block.allocated)
        << "RecordEvent is called with a block that has not been allocated!";
    block.streams.insert(stream);
  }

 private:
  cudaError_t InsertEvents(Block &block) {
    NVTX_RANGE(K2_FUNC);
    // InsertEvents is called from `Free`, which has already held the mutex.
    std::unordered_set<cudaStream_t> streams(std::move(block.streams));
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      cudaEvent_t event;
      cudaError_t err =
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (err != cudaSuccess) return err;

      err = cudaEventRecord(event, *it);
      if (err != cudaSuccess) return err;

      ++block.event_count;
      cuda_events_.emplace_back(event, block.ptr);
    }
    return cudaSuccess;
  }

  /* Process events in `cuda_events_`.

     If the events of a block have all been processed, this block
     is put into `available_` and is ready for reuse.

     If `cudaEventQuery()` returns `cudaErrorNotReady`, it
     returns immediately.

     @return `cudaSuccess` on success; on error, it returns a
              cuda error code.
   */
  cudaError_t ProcessEvents() {
    NVTX_RANGE(K2_FUNC);
    // InsertEvents is called from `Malloc` and `Free`,
    // which has already held the mutex.
    while (!cuda_events_.empty()) {
      auto &e = cuda_events_.front();
      cudaEvent_t event = e.first;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) break;

      if (err != cudaSuccess) return err;

      err = cudaEventDestroy(event);
      if (err != cudaSuccess) return err;

      Block &block = blocks_.at(e.second);
      --block.event_count;

      if (block.event_count == 0 && !block.allocated) available_.insert(block);

      cuda_events_.pop_front();
    }
    return cudaSuccess;
  }

 private:
  // It contains all blocks allocated by Malloc.
  std::unordered_map<void *, Block> blocks_;

  using Compare = bool (*)(const BlockSize &, const BlockSize &);
  // It contains all free blocks **sorted** by block size in increasing order.
  std::set<BlockSize, Compare> available_;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, void *>> cuda_events_;

  // to protect `blocks_`, `available_` and `cuda_events_` being accessed
  // from multiple threads
  std::mutex mutex_;
};

static PinnedAllocator *GetPinnedAllocator() {
  static std::once_flag init_flag;
  static PinnedAllocator *allocator = nullptr;

  std::call_once(init_flag, []() {
    // it is never freed.
    allocator = new PinnedAllocator;
  });

  return allocator;
}

}  // namespace

class PinnedContext : public Context {
 public:
  PinnedContext() { allocator_ = GetPinnedAllocator(); }

  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    cudaError_t err = allocator_->Malloc(bytes, &p);
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  void Deallocate(void *data, void *deleter_context) override {
    (void)deleter_context;
    allocator_->Free(data);
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

  void CopyDataTo(size_t num_bytes, const void *src, ContextPtr dst_context,
                  void *dst) override {
    DeviceType device_type = dst_context->GetDeviceType();
    switch (device_type) {
      case kCpu:
        // we assume that src and dst do not overlap
        memcpy(dst, src, num_bytes);
        break;
      case kCuda: {
        cudaStream_t stream = dst_context->GetCudaStream();
        cudaError_t ret = cudaMemcpyAsync(dst, src, num_bytes,
                                          cudaMemcpyHostToDevice, stream);
        K2_CHECK_CUDA_ERROR(ret);

        allocator_->RecordEvent(stream, const_cast<void *>(src));
        break;
      }
      default:
        K2_LOG(FATAL) << "Unsupported device type: " << device_type;
        break;
    }
  }

 private:
  PinnedAllocator *allocator_;  // NOT owned here
};

ContextPtr GetPinnedContext() {
  static std::once_flag has_cuda_init_flag;
  static bool has_cuda = false;

  std::call_once(has_cuda_init_flag, []() {
    int32_t count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
      K2_LOG(WARNING) << "cudaGetDeviceCount() failed: "
                      << cudaGetErrorString(err) << "\n."
                      << "Return a CPU context";
    } else if (count == 0) {
      K2_LOG(WARNING)
          << "No CUDA capable devices are found. Return a CPU context.";
    } else {
      has_cuda = true;
    }
  });

  if (has_cuda) return std::make_shared<PinnedContext>();

  return GetCpuContext();
}

ContextPtr GetContextForTransfer(DeviceType device_type) {
  switch (device_type) {
    case kCpu:
      return GetCpuContext();
    case kCuda:
      return GetPinnedContext();
    default:
      K2_LOG(FATAL) << "Unsupported device type: " << device_type;
      return nullptr;
  }
}

}  // namespace k2
