
function(k2_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py") # append `_py` to avoid name collisions with cc test

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:k2>"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
    #TODO(haowen): uncomment this
    #fsa_test.py 
)

foreach(source IN LISTS py_test_files)
  k2_add_py_test(${source})
endforeach()
