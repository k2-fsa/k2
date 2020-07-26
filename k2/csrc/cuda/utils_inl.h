#include <cub/device/device_scan.cuh>

/* Don't include this file directly; it is included by utils.h.
   It contains implementation code. */



template <typename SrcPtr, typename DestPtr>
void ExclusivePrefixSum(ContextPtr &c, int n, SrcPtr src, DestPtr dest) {
  DeviceType d = c.GetDeviceType();
  using SumType = decltype(dest[0]);
  if (d == kCpu) {
    SumType sum = 0;
    for (int i = 0; i < n; i++) {
      dest[i] = sum;
      sum += src[i];
    }
  } else {
    assert(d != kUnk);
    // TODO: use some kind of cub template here.

  }
}
