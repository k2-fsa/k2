#include <types.h>




/*
  Allocator that allocates memory on GPU.  Likely won't call CUDA's
  malloc for each request, as that's very slow.  We'll probably ensure that we have one
  DeviceAllocator per CPU thread to avoid the need for locks.
  We'll likely
 */
class DeviceAllocator {
  // Allocate memory on device; throws std::bad_alloc if could not.
  // For now assume there's just one device (i.e. GPU), but we may later
  // have to extend this somehow.

  void *Alloc(size_t bytes);

  // Free memory; note, this requires you to know the size of region you
  // originally allocated (this may save some memory in the allocator,
  // but we may later choose to simplify this interface).
  void Free(void *mem, size_t bytes);

 private:
  // ...

};
