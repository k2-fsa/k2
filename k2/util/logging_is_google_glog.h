// k2/util/logging_is_google_glog.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Meixu Song)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
#define K2_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

#include <map>
#include <set>
#include <vector>
#include <iomanip>  // std::setw

/*
 * - Include stl_logging.h first to needs to actually use stl_logging (template magic)
 * - In addition, stl logging is disabled for .cu files b/c nvcc has trouble with it.
 * - And stl_logging it's not a good choice for some mobile platforms, so we block it.
*/

#ifdef __CUDACC__
#include <cuda.h>
#endif

#if !defined(__CUDACC__) && !defined(K2_USE_MINIMAL_GLOG)
#include <glog/stl_logging.h>
#else
/*
 * To disable stl logging for nvcc:
 * Just ignore the log message within overloaded operator "<<" template function.
 * Make it into a macro, then apply to "vector,map,set" stl.
*/
namespace std {
#define INSTANTIATE_FOR_CONTAINER(container)                      \
  template <class... Types>                                       \
  ostream& operator<<(ostream& out, const container<Types...>&) { \
    return out;}

INSTANTIATE_FOR_CONTAINER(vector)
INSTANTIATE_FOR_CONTAINER(map)
INSTANTIATE_FOR_CONTAINER(set)
#undef INSTANTIATE_FOR_CONTAINER
}  // namespace std
#endif

#include <glog/logging.h>

#endif  // K2_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
