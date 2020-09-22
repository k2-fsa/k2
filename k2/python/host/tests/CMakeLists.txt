function(k2_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "host_${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(k2host_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_k2host>:${k2host_path}:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
    arcsort_test.py
    array_test.py
    aux_labels_test.py
    connect_test.py
    determinize_test.py
    fsa_equivalent_test.py
    fsa_test.py
    intersect_test.py
    properties_test.py
    rmepsilon_test.py
    topsort_test.py
    weights_test.py
)

foreach(source IN LISTS py_test_files)
  k2_add_py_test(${source})
endforeach()
