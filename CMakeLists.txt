cmake_minimum_required(VERSION 3.18)
project(dtf_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDES)

# Try to find NCCL
find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS /usr/local/nccl/include)
find_library(NCCL_LIBRARY NAMES nccl HINTS /usr/local/nccl/lib)
if (NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(WARNING "NCCL not found in standard paths. Ensure it is installed and accessible.")
endif()

add_subdirectory(csrc)
