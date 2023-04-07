# please keep the list sorted
set(torch_srcs
  arc.cu
  fsa.cu
  fsa_algo.cu
  index_add.cu
  index_select.cu
  mutual_information.cu
  mutual_information_cpu.cu
  nbest.cu
  ragged.cu
  ragged_ops.cu
  rnnt_decode.cu

  v2/any.cu
  v2/autograd/swoosh.cu
  v2/doc/doc.cu
  v2/k2.cu
  v2/ragged_any.cu
  v2/ragged_shape.cu
)

if (K2_WITH_CUDA)
  list(APPEND torch_srcs mutual_information_cuda.cu)
endif()

set(torch_srcs_with_prefix)
foreach(src IN LISTS torch_srcs)
  list(APPEND torch_srcs_with_prefix "torch/${src}")
endforeach()

set(torch_srcs ${torch_srcs_with_prefix} PARENT_SCOPE)
