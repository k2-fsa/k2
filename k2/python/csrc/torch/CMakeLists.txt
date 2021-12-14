# please keep the list sorted
set(torch_srcs
  arc.cu
  array_ops.cu
  discounted_cum_sum.cu
  fsa.cu
  fsa_algo.cu
  index_add.cu
  index_select.cu
  nbest.cu
  ragged.cu
  ragged_ops.cu
  torch_util.cu

  v2/any.cu
  v2/doc/doc.cu
  v2/k2.cu
  v2/ragged_any.cu
  v2/ragged_shape.cu
)

set(torch_srcs_with_prefix)
foreach(src IN LISTS torch_srcs)
  list(APPEND torch_srcs_with_prefix "torch/${src}")
endforeach()

set(torch_srcs ${torch_srcs_with_prefix} PARENT_SCOPE)
