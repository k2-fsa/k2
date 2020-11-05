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
    PROPERTY ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_k2>:${k2_path}:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  arc_sort_test.py
  get_tot_scores_test.py
  connect_test.py
  dense_fsa_vec_test.py
  fsa_test.py
  intersect_dense_pruned_test.py
  intersect_test.py
  linear_fsa_test.py
  shortest_path_test.py
  top_sort_test.py
)

foreach(source IN LISTS py_test_files)
  k2_add_py_test(${source})
endforeach()
