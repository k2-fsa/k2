function(k2_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(k2_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${k2_path}:$<TARGET_FILE_DIR:_k2>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  add_epsilon_self_loops_test.py
  arc_sort_test.py
  cat_test.py
  compose_arc_maps_test.py
  closure_test.py
  compose_test.py
  connect_test.py
  create_sparse_test.py
  ctc_graph_test.py
  ctc_loss_test.py
  ctc_topo_test.py
  dense_fsa_vec_test.py
  determinize_test.py
  expand_ragged_attributes_test.py
  fsa_from_unary_function_ragged_test.py
  fsa_from_unary_function_tensor_test.py
  fsa_test.py
  get_arc_post_test.py
  get_backward_scores_test.py
  get_forward_scores_test.py
  get_best_matching_stats_test.py
  get_tot_scores_test.py
  index_add_test.py
  index_and_sum_test.py
  index_select_test.py
  index_test.py
  intersect_dense_pruned_test.py
  intersect_dense_test.py
  intersect_device_test.py
  intersect_test.py
  invert_test.py
  levenshtein_alignment_test.py
  levenshtein_graph_test.py
  linear_fsa_test.py
  linear_fsa_with_self_loops_test.py
  linear_fst_test.py
  linear_fst_with_self_loops_test.py
  multi_gpu_test.py
  mutual_information_test.py
  mwer_test.py
  nbest_test.py
  numerical_gradient_check_test.py
  online_dense_intersecter_test.py
  ragged_ops_test.py
  ragged_shape_test.py
  ragged_tensor_test.py
  ragged_test.py
  random_paths_test.py
  remove_epsilon_self_loops_test.py
  remove_epsilon_test.py
  reverse_test.py
  rnnt_decode_test.py
  rnnt_loss_test.py
  shortest_path_test.py
  sparse_abs_test.py
  symbol_table_test.py
  top_sort_test.py
  union_test.py
  replace_fsa_test.py
)

foreach(source IN LISTS py_test_files)
  k2_add_py_test(${source})
endforeach()
