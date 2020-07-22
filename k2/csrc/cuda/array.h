#include "k2/csrc/cuda/memory.h"


/*
  Array1* is a 1-dimensional contiguous array (i.e. it doesn't support a stride).
*/


template <typename T> class Array1 {
 public:
  int size;  // dimension of only (1st) axis
  int byte_offset;  // Offset from region->data, in bytes.  (Not storing the
                    // data locally allows us to move the underlying data in
                    // 'region' to handle resizing, without risk of dangling
                    // pointers).
  T *data();  // Gives pointer to 1st elem
  std::shared_ptr<Region> region;  // Region that `data` is a part of.  Device
                                   // type is stored here.  For an Array1 with
                                   // zero size (e.g. created using empty
                                   // constructor), will point to an empty
                                   // Region.


  // generally T will be some kind of lambda.  We'll do src(i) to
  // evaluate element i.  If 'd' is kGpu we'll assume lambda is callable
  // on-device; otherwise we'll assume it's callable on CPU.
  template <typename T>
  Array1(DeviceType d, int size, T lambda) {
    Init(d, size);
    if (d == kGpu) {
      Eval<T,kGpu>(data, size, lambda);
    } else {
      Eval<T,kCpu>(data, size, lambda);
    } else {
      assert("Can't initialize array with unknown device type");
    }
  }

  DeviceType Device() { return region->device; }

  // Resizes, copying old contents if we could not re-use the same memory location.
  // It will always at least double the allocated size if it has to reallocate.
  // See Region::num_bytes vs. Region::bytes_used.
  void Resize(int new_size);

 private:
  void Init(DeviceType d, int size) {
    // .. takes care of allocation etc.
  }

};
