cmake_minimum_required(VERSION 3.13)
project(zvec_c_wrapper CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Zvec source directory (can be overridden via -DZVEC_SRC_DIR=...)
if(NOT DEFINED ZVEC_SRC_DIR)
    set(ZVEC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/zvec")
endif()

set(ZVEC_INCLUDE_DIR "${ZVEC_SRC_DIR}/src/include")
set(ZVEC_LIB_DIR "${ZVEC_SRC_DIR}/lib")

message(STATUS "ZVEC_SRC_DIR: ${ZVEC_SRC_DIR}")
message(STATUS "ZVEC_INCLUDE_DIR: ${ZVEC_INCLUDE_DIR}")
message(STATUS "ZVEC_LIB_DIR: ${ZVEC_LIB_DIR}")

# Build the wrapper objects
add_library(zvec_c_wrapper STATIC
    src/collection.cpp
    src/doc.cpp
    src/schema.cpp
    src/query.cpp
    src/index_params.cpp
    src/status.cpp
    src/options.cpp
    src/init.cpp
)

target_include_directories(zvec_c_wrapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ZVEC_INCLUDE_DIR}
        ${ZVEC_SRC_DIR}/src
)

set_target_properties(zvec_c_wrapper PROPERTIES
    OUTPUT_NAME zvec_c_wrapper
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
