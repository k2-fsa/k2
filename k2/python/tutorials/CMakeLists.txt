
add_test(NAME lesson_01_basics
  COMMAND
    "${PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/lesson_01_basics.py"
)

set_property(TEST lesson_01_basics
  PROPERTY ENVIRONMENT  "PYTHONPATH=$<TARGET_FILE_DIR:k2>"
)

# TODO(fangjun): add a helper function to add tutorials
