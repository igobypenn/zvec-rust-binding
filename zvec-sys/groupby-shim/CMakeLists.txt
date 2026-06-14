# Group-by shim for zvec-rust-bindings.
#
# Upstream zvec v0.5.0 C API does not expose group-by query. This tiny static
# library wraps zvec::Collection::GroupByQuery so the Rust bindings can
# preserve the group_by_query feature.
cmake_minimum_required(VERSION 3.13)
project(zvec_groupby_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ZVEC_SRC)
    message(FATAL_ERROR "ZVEC_SRC must point at the upstream zvec source directory")
endif()

add_library(zvec_groupby_shim STATIC
    groupby.cpp
)

target_include_directories(zvec_groupby_shim PRIVATE
    ${ZVEC_SRC}/src/include
    ${ZVEC_SRC}/src
    include
)

set_target_properties(zvec_groupby_shim PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "zvec_groupby_shim"
)

target_compile_options(zvec_groupby_shim PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)
