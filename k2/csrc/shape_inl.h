/**
 * @brief
 * shape_inl
 *
 * @note
 * to be included only from shape.h
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

namespace k2 {

template <typename T>
T Array1::operator [] (int32_t i) {
  Context *c = Context().get();
  DeviceType t = c->GetDeviceType();
  K2_CHECK_LE(static_cast<uint32_t>(i), static_cast<uint32_t>(size_));
  if (t == kCpu) {
    return data_[i];
  } else {
    // stream associated with this Context; we assume it is not synchronized
    // with the legacy default stream (i.e. was created with
    // cudaStreamNonBlocking flag).
    cudaStream_t stream = c->GetCudaStream();
    cudaEvent_t event;
    // Make the default stream wait for any unfinished kernels in this stream.
    // (Note: the only reason we're involving the default stream is that
    // we haven't got into using pinned memory yet.)

    // TODO: check status of all the following calls via appropriate macro.
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    cudaStreamWaitEvent(CU_DEFAULT_STREAM, event, 0);
    T ans;
    cudaMemcpy((void*)&ans, (void*)(data_ + i), sizeof(T),
               cudaMemcpyDeviceToHost);
    return T;
  }
}

}  // namespace k2
