# Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )
# See ../../LICENSE for clarification regarding multiple authors
#
# By default, K2 don't use glog, as glog is not a self-contained dependency,
# to make tools that use k2 as dependency:
#   - not worry about glog-global-init, (If K2 do it, the whole project also get the same setting,
#     thus it's the top caller job, as glog is designed)
#   - avoid problems caused by glog, e.g.:
#       - multi-glog conflicts (https://github.com/bazelbuild/bazel/issues/507)
#       - projects that use K2, have to get involed with glog, about pre-existed checking, glog-global-init..
#         (https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/stubs/logging.h)
#       - all other problems that to use a tool not expected to be used into a library, which would get distributed
#         as other's dependency, reasons:
#           - globle scope flags
#           - leave jobs that hope other do, and landmines that hope other notice.
# Glog should get used when the whold toolchain is under your control, or each dependencies take care of it.
# (As the thruth is, no one serious library (even google's, protobuf, bazel, ..) use glog as its dependency.
#  But to be compatiable with the popular glog, some make a miniglog and use it, instead of real glog).
# That's why I said, either without glog, or glog and glog-mummy both existed.
#
# Otherwise, whether you like or not, you are the one give the trouble, like K2 faced here.
#
# PS: I know why glog is designed so. It gives conviences that we need,
#     and also binded expectation that user know the whole image, as a trade.
#     Please don't asking me same questions again, as I have give my answer here,
#     which has been repeated at plenty places..
#     Seriously, try to make effort till you believe you are right,
#     or at least after several arguments, next time doubt with truth rather than personal taste or intuition.
#     It's more rude than words could be.

option(K2_USE_GLOG "Use GLOG as logger, otherwise one compatible with glog is used (loguru)]" OFF)

# check if use glog
if(TARGET glog::glog)
  # build options
  set(K2_USE_GLOG ON)
  message(STATUS "Glog is pre-existed, use Glog")
else()
  message(STATUS "Glog is not pre-existed, check K2_USE_GLOG ..")
  if(K2_USE_GLOG)
    find_package(Glog)
    if(NOT Glog_FOUND)
      message(WARNING "Glog not found, use one compatible with glog: loguru")
    else()
      message(STATUS "Glog found, use Glog, introduce its as a library glog::glog")
    endif()
  else()
    set(K2_USE_GLOG OFF)
    message(STATUS "Glog is not found, use one compatible with glog: loguru")
  endif()
endif()

# If K2_USE_GLOG,
# - ON: Set the macro "K2_USE_GLOG" in glog_macros.h.
# - OFF: Comment out macros there.
# generate the result glog_macros.h from glog_macros.h.in by cmake
configure_file(
    k2/csrc/util/glog_macros.h.in
    k2/csrc/util/glog_macros.h
    @ONLY
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}) # include glog_macros.h
