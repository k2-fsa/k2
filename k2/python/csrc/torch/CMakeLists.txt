# please keep the list sorted
set(torch_srcs
  arc.cu
  array.cu
  fsa.cu
  ragged.cu
  torch_util.cu
)

set(torch_srcs_with_prefix)
foreach(src IN LISTS torch_srcs)
  list(APPEND torch_srcs_with_prefix "torch/${src}")
endforeach()

set(torch_srcs ${torch_srcs_with_prefix} PARENT_SCOPE)
