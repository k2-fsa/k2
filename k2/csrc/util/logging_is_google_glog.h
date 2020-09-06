// k2/csrc/util/logging_is_google_glog.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Meixu Song)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
#define K2_CSRC_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

#include <map>
#include <set>
#include <vector>
#include <iomanip>  // std::setw

/*
 * For usages that actually use stl_logging (template magic),
 * stl_logging.h need included before logging.h.
 * - In addition, stl logging is disabled for .cu files
 *   b/c nvcc has trouble with it.
 *
 * @note
 * stl_logging it's not good at mobile platforms,
 * so block it as you pleasure.
*/

#ifdef __CUDACC__
#include <cuda.h>

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
#else
#include <glog/stl_logging.h>
#endif

#include <glog/logging.h>

#endif  // K2_CSRC_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
