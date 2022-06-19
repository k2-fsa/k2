# Introduction

This directory shows you how to use `k2` in your project managed by CMake.

## Step 1: Install k2

Please refer to <https://k2-fsa.github.io/k2/installation/index.html>
to install k2

## Step 2: Add k2 to your projects

You can use [cmake/k2.cmake](./cmake/k2.cmake) as an example for
how to introduce `k2` into your projects.

Basically, all you need to do is

```cmake
find_package(k2 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${K2_CXX_FLAGS}")
```

and link your executable or library with `${K2_LIBRARIES}`.

One thing you need to note is that you have to tell CMake where to find `k2`.

```bash
python3 -c "import k2; print(k2.cmake_prefix_path)"
```

The above command prints the path that you can use to inform CMake of
the location of `k2`.

Example usage of the above command is

```cmake
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import k2; print(k2.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE K2_CMAKE_PREFIX_PATH
)

message(STATUS "K2_CMAKE_PREFIX_PATH: ${K2_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${K2_CMAKE_PREFIX_PATH}")

find_package(k2 REQUIRED)
```

**Note**: You also have to link with `${TORCH_LIBRARIES}`. You can
find example usage in [./CMakeLists.txt](./CMakeLists.txt).

# Steps to run this project

Assume you have installed k2. The following shows you how to run this
project on Linux, macOS, and Windows.

## Linux/macOS

If you are using Linux or macOS, you can do:

```bash
cd /path/to/k2-torch-api-test
mkdir build
cd build
cmake ..
make -j 6
./bin/torch_api_test
```

or simply use

```bash
cd /path/to/k2-torch-api-test
make
```

## Windows

If you are using Windows, you can do:

```bash
cd /path/to/k2-torch-api-test
mkdir build
cd build
cmake ..
cmake --build . --target ALL_BUILD --config Release
ctest -C Release --output-on-failure
```
