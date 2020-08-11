// k2/csrc/cuda/context.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_CONTEXT_H_
#define K2_CSRC_CUDA_CONTEXT_H_

#include <cassert>
#include <memory>

namespace k2 {

enum class DeviceType {
  kUnk,
  kCuda,
  kCpu,
};

constexpr DeviceType kUnk = DeviceType::kUnk;
constexpr DeviceType kCuda = DeviceType::kCuda;
constexpr DeviceType kCpu = DeviceType::kCpu;

class Context;
using ContextPtr = std::shared_ptr<Context>;

/**
   class Context is the main surface of interaction with external
   tensor libraries like PyTorch; it allows us to use their notions
   of device and of memory allocation.
   (in the case of PyTorch it would probably contain a Device and an Allocator).

   We will sub-class this in several ways: with versions that wrap external
   toolkits like PyTorch, and also a "native" version that's mostly for
   testing purposes.

   This object should be allocated with std::shared_ptr, as that's how we store
   pointers to it.
*/
class Context: public std::enable_shared_from_this<Context> {
 public:
  virtual ~Context() = default;


  /*
    Return a 'child context' of this.  Think of this as like a sub-process, useful
    for running things in parallel like in multiple threads or CUDA stream.
    Here we give a default implementation that just returns *this (not very useful).
    In the case of CUDA, what this would likely do is:
         create a new stream (child stream)
         create a CUDA event in this stream (the parent stream)
         make the child stream wait on the just-created event in the parent stream
       And then in the destructor of the returned ContextPtr:
         create a CUDA event in the stream we created
         make the parent stream wait on that event.

    In the case of CPU, this would likely:
         wait for a free thread from a thread pool, and put a pointer to it
         in this Context.
     The destructor would:
         tell that thread
         wait for that thread to terminate.

     So the idea is that destroying the child context is like waiting on a child
     process.  We can also later add a function to explicitly wait on a context.

     The idea is to have a device-independent way of "backgrounding" tasks.
   */
  virtual ContextPtr GetChild() { return shared_from_this(); }

  // Returns kCuda if this device is a CUDA device, or kCpu if it's the CPU.
  virtual DeviceType GetDeviceType() const = 0;

  /*
    Allocate memory on this device (raise an exception on failure, which we likely
    won't attempt to catch).

        @param [in] bytes   Number of bytes to allocate; may be zero, in which
                            case NULL will be returned.   Let's assume the alignment
                            of the returned memory is at least as strict as
                            for malloc() with the same values.
        @param [out] decoder_context   If more information than the returned pointer
                            is required in order to deallocat this memory, then
                            that information will be supplied as 'deleter_context'.
                            In some cases this will be zero.
        @return    Returns the allocated block, or NULL if bytes == 0.
  */
  virtual void *Allocate(size_t bytes, void **deleter_context) = 0;

  // Return true if this is the same device as 'other' (essentially: that it
  // lives in the same physical memory space).
  // Must always return true if this == &other.
  virtual bool IsSame(const Context &other) const = 0;

  bool operator==(const Context &other) const {
    return this == &other || this->IsSame(other);
  }

  /*
    Free memory that was allocated by Context::Allocate() from this Context object
    (or, in general, memory obtained from an external toolkit that this Context object
    knows how to delete).
           @param [in] data       The memory to delete (may be NULL)
           @param [in] deleter_context    Some Context objects may require additional
                            context information to delete the memory, like
                            the concept of 'context' in PyTorch's DataPtr.  Or may be
                            NULL in some instances.  In general, whatever was output
                            by Allocate() to deleter_context should be supplied to
                            Deallocate().
  */
  virtual void Deallocate(void *data, void *deleter_context) = 0;
};

template <typename T>
ContextPtr GetContext(const T &t) {
  // suppose T has member method `Context`
  return t.Context();
}

template <typename First, typename... Rest>
ContextPtr GetContext(const First &first, const Rest &... rest) {
  ContextPtr ans1 = GetContext(first), ans2 = GetContext(rest...);
  assert(*ans1 == *ans2 && "Contexts mismatch");
  return ans1;
}

/*
  Note: Region will always be allocated with std::make_shared<Region>(...), and
  holders of a Region will hold it via shared_ptr.

  To enable resizable (extendable) arrays to work when multiple objects may
  point to the same memory: there will be a convention that if an Array covers
  all the bytes used in a memory region it gets to use the remaining bytes
  allocated (i.e. it gets to increased bytes_used up until num_bytes).
  That means only one of the Arrays pointing to that memory region will 'take'
  that memory. Once it gets too large and can't fit in the Region, it allocates
  a new Region.
*/
struct Region: public std::enable_shared_from_this<Region> {
  // note: the inheritance from std::enable_shared_from_this<Region>
  // means that this object has a function
  //  std::shared_ptr<Region> shared_from_this();
  ContextPtr context;
  void *data;         // Pointer to the start of the allocated memory region
  void *deleter_context;  // if non-NULL, this is provided to the context in the
                          // destructor instead of 'data'.  It will be NULL for
                          // some Contexts, non-NULL for others.
  size_t num_bytes;   // number of bytes allocated.
  size_t bytes_used;  // largest number of bytes used/covered by any Array that
                      // points to this Region (this is relevant for things that
                      // behave like resizable vectors).

  // You need template arg to invoke this, e.g. region->GetData<int>();
  // You can also choose to template additionally on the device-type, like
  // region->GetData<int,kGpu>(), to activate a check that it's on the expected
  // device.
  template <typename T = void, DeviceType d = kUnk>
  T *GetData() {
    if (d != kUnk) assert(d == context->GetDeviceType());
    return reinterpret_cast<T*>(data);
  }

  ~Region() {
    context->Deallocate(deleter_context != nullptr ? deleter_context : data);
  }
};

using RegionPtr = std::shared_ptr<Region>;

// Return a basic Context object suitable for work on the CPU.
ContextPtr GetCpuContext();

// Return a basic Context object suitable for work with CUDA,
// with specified GPU-id (or the first one we grab, if gpu_id == -1).
// This will be a *native* context, one that used k2's own memory
// manager.  If you want to use (say) PyTorch's memory manager,  you
// should use a Context passed in from PyTorch
ContextPtr GetCudaContext(int gpu_id = -1);

/**
   Allocate a new Region.

     @param [in] context   Context from which to allocate the region
                          (specifies the device and allocator)
     @param [in] num_bytes  Number of bytes to allocate.  Note: zero bytes
                          is OK and will be handled in the same way as
                          nonzero allocations.
   Returns a new region.   Raises exception (TBD, may be dependent on the
                          context) on error such as allocation failure.
                          The returned
                          region will have bytes_used == num_bytes; if the user
                          wants to change this they can do it afterward.
*/
RegionPtr NewRegion(Context &context, size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  std::shared_ptr<Region> ans = std::make_shared<Region>();
  ans->context = context.shared_from_this();
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

/*
  Convenience wrapper for NewRegion() that takes the context from a provided
  region.
 */
std::shared_ptr<Region> NewRegion(Region &region, size_t num_bytes) {
  return NewRegion(region.context, num_bytes);
}

// Objects from k2 generally have a Context() method, so this template
// will work to get the device-type for pretty arbitrary objects.
template <typename T>
inline DeviceType DeviceOf(const T &t) {
  return t.Context().GetDeviceType();
}

}  // namespace k2

#endif  // K2_CSRC_CUDA_CONTEXT_H_
