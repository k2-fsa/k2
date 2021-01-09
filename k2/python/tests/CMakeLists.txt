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
  closure_test.py
  compose_test.py
  connect_test.py
  ctc_gradients_test.py
  dense_fsa_vec_test.py
  determinize_test.py
  fsa_test.py
  get_tot_scores_test.py
  index_add_test.py
  index_select_test.py
  index_test.py
  intersect_dense_pruned_test.py
  intersect_dense_test.py
  intersect_test.py
  invert_test.py
  linear_fsa_test.py
  numerical_gradient_check_test.py
  ragged_ops_test.py
  ragged_shape_test.py
  ragged_test.py
  remove_epsilon_test.py
  shortest_path_test.py
  symbol_table_test.py
  top_sort_test.py
  union_test.py
)

foreach(source IN LISTS py_test_files)
  k2_add_py_test(${source})
endforeach()
