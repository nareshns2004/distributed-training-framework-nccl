set(CSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CSRC_DIR}/include)
if (NCCL_INCLUDE_DIR)
    include_directories(${NCCL_INCLUDE_DIR})
endif()
include_directories(${PYBIND11_INCLUDES})

file(GLOB_RECURSE SRC_FILES 
    "${CSRC_DIR}/core/*.cpp"
    "${CSRC_DIR}/cuda/*.cu"
)

# Pybind bindings
add_library(dtf_C SHARED 
    ${SRC_FILES}
    "${CSRC_DIR}/bindings/pybind.cpp"
)

set_target_properties(dtf_C PROPERTIES PREFIX "")
# Typically extension is .so
set_target_properties(dtf_C PROPERTIES SUFFIX ".so")

target_link_libraries(dtf_C ${NCCL_LIBRARY})
